from __future__ import annotations

import contextlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .config import resolve_package_path


@dataclass(frozen=True)
class PerceptionResult:
    depth: np.ndarray
    sock_mask: np.ndarray
    leg_mask: np.ndarray
    sock_depth: np.ndarray
    leg_depth: np.ndarray
    quality: dict


def assess_masks(
    sock_mask: np.ndarray,
    leg_mask: np.ndarray,
    *,
    min_fraction: float = 0.0001,
    max_fraction: float = 0.95,
    previous_areas: Optional[tuple] = None,
    max_area_change: float = 4.0,
    prompt_points: Optional[Mapping[str, Sequence[Sequence[float]]]] = None,
    renderer_masks: Optional[Mapping[str, np.ndarray]] = None,
    semantic: Optional[Mapping] = None,
) -> dict:
    sock = np.asarray(sock_mask, dtype=bool)
    leg = np.asarray(leg_mask, dtype=bool)
    checks = {
        "same_shape": sock.ndim == 2 and sock.shape == leg.shape,
        "masks_distinct": sock.shape == leg.shape and not np.array_equal(sock, leg),
    }
    fractions = {
        "sock": float(sock.mean()) if sock.size else 0.0,
        "leg": float(leg.mean()) if leg.size else 0.0,
    }
    checks["sock_nontrivial"] = min_fraction <= fractions["sock"] <= max_fraction
    checks["leg_nontrivial"] = min_fraction <= fractions["leg"] <= max_fraction
    areas = (int(sock.sum()), int(leg.sum()))
    checks["area_continuity"] = True
    if previous_areas is not None:
        for current, previous in zip(areas, previous_areas):
            ratio = max(current, previous) / max(1, min(current, previous))
            if ratio > max_area_change:
                checks["area_continuity"] = False
    semantic = semantic or {}
    checks["sock_semantic_area"] = (
        float(semantic.get("sock_min_fraction", min_fraction))
        <= fractions["sock"]
        <= float(semantic.get("sock_max_fraction", max_fraction))
    )
    checks["leg_semantic_area"] = (
        float(semantic.get("leg_min_fraction", min_fraction))
        <= fractions["leg"]
        <= float(semantic.get("leg_max_fraction", max_fraction))
    )
    overlap = np.count_nonzero(sock & leg) / max(1, min(areas))
    checks["limited_overlap"] = overlap <= float(
        semantic.get("maximum_overlap_fraction", 1.0)
    )
    prompt_containment = {}
    if prompt_points:
        for name, mask in (("sock", sock), ("leg", leg)):
            points = prompt_points.get(name, [])
            valid = [
                bool(mask[int(y), int(x)])
                for x, y in points
                if 0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1]
            ]
            prompt_containment[name] = bool(valid and all(valid))
        if semantic.get("require_prompt_containment", False):
            checks["prompt_containment"] = all(prompt_containment.values())
    renderer_agreement = {}
    if renderer_masks:
        for name, mask in (("sock", sock), ("leg", leg)):
            reference = renderer_masks.get(name)
            if reference is None:
                continue
            reference = np.asarray(reference, dtype=bool)
            if reference.shape != mask.shape or not reference.any():
                renderer_agreement[name] = {"available": False}
                continue
            union = np.count_nonzero(mask | reference)
            iou = np.count_nonzero(mask & reference) / max(1, union)
            centroid_distance = _centroid_distance(mask, reference)
            diagonal = float(np.hypot(*mask.shape))
            renderer_agreement[name] = {
                "available": True,
                "iou": float(iou),
                "centroid_distance_fraction": float(centroid_distance / diagonal),
            }
        available = [value for value in renderer_agreement.values() if value["available"]]
        if available:
            checks["renderer_iou"] = all(
                value["iou"] >= float(semantic.get("minimum_iou_with_renderer", 0.0))
                for value in available
            )
            checks["renderer_centroid"] = all(
                value["centroid_distance_fraction"]
                <= float(semantic.get("maximum_centroid_distance_fraction", 1.0))
                for value in available
            )
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "fractions": fractions,
        "areas": list(areas),
        "overlap_fraction": float(overlap),
        "prompt_containment": prompt_containment,
        "renderer_agreement": renderer_agreement,
    }


def _centroid_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_points = np.argwhere(first)
    second_points = np.argwhere(second)
    if not len(first_points) or not len(second_points):
        return float("inf")
    return float(np.linalg.norm(first_points.mean(axis=0) - second_points.mean(axis=0)))


def prompt_components(
    mask: np.ndarray, points: Sequence[Sequence[float]]
) -> np.ndarray:
    """Keep only connected foreground components selected by positive prompts."""
    value = np.asarray(mask, dtype=bool)
    if not points or not value.any():
        return value
    try:
        import cv2
    except ImportError:
        return value
    _, labels = cv2.connectedComponents(value.astype(np.uint8), connectivity=8)
    selected = set()
    for x, y in points:
        row, column = int(y), int(x)
        if 0 <= row < value.shape[0] and 0 <= column < value.shape[1]:
            label = int(labels[row, column])
            if label:
                selected.add(label)
    if not selected:
        return value
    return np.isin(labels, list(selected))


def masked_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth_value = np.asarray(depth)
    mask_value = np.asarray(mask, dtype=bool)
    if depth_value.dtype != np.uint8 or depth_value.shape != mask_value.shape:
        raise ValueError("depth must be uint8 and match the mask")
    return np.where(mask_value, depth_value, 0).astype(np.uint8)


def select_prompt_points(frame: np.ndarray, object_name: str) -> tuple:
    """Collect positive/negative SAM points; left=positive, right=negative, Enter=done."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("interactive prompts require opencv-python") from error
    points, labels = [], []
    window = f"Select {object_name}: left +, right -, Enter done"

    def callback(event, x, y, _flags, _parameter):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            points.append([float(x), float(y)])
            labels.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, callback)
    while True:
        preview = np.asarray(frame).copy()
        for (x, y), label in zip(points, labels):
            cv2.circle(preview, (int(x), int(y)), 5, (0, 255, 0) if label else (255, 0, 0), -1)
        cv2.imshow(window, preview[:, :, ::-1])
        key = cv2.waitKey(20)
        if key in (10, 13) and points:
            break
        if key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError(f"{object_name} prompt selection cancelled")
    cv2.destroyWindow(window)
    return points, labels


class SAMDepthPerception:
    """SAM2 tracking and Depth Anything preprocessing used by the real controller."""

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(
        self,
        config: Mapping,
        *,
        sock_predictor=None,
        leg_predictor=None,
        depth_model=None,
    ):
        self.config = config
        self.settings = config["inference"]
        self.device = self.settings.get("device", "cuda")
        self.sock_predictor = sock_predictor
        self.leg_predictor = leg_predictor
        self.depth_model = depth_model
        self._initialized = False
        self._previous_areas = None
        self._prompt_points = None
        calibration_path = resolve_package_path(
            self.settings["depth_anything"]["calibration"]
        )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        self.depth_min = float(calibration["depth_min"])
        self.depth_max = float(calibration["depth_max"])
        if self.depth_max <= self.depth_min:
            raise ValueError("Depth Anything calibration has an invalid range")

    def load(self) -> None:
        if self.sock_predictor is not None and self.leg_predictor is not None and self.depth_model is not None:
            return
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("SAM2/Depth Anything inference requires torch") from error
        sam = self.settings["sam2"]
        sam_source = resolve_package_path(sam["source"])
        # The vendored camera predictor contains a legacy top-level import from
        # its own package directory. Keep both entries active through Hydra's
        # lazy model instantiation without modifying the read-only ShareSet.
        sam_paths = [str(sam_source), str(sam_source / "sam2")]
        sys.path[0:0] = sam_paths
        try:
            build_module = importlib.import_module("sam2.build_sam")
            builder = build_module.build_sam2_camera_predictor
            model_config = str(resolve_package_path(sam["config"]))
            checkpoint = str(resolve_package_path(sam["checkpoint"]))
            self.sock_predictor = builder(model_config, checkpoint, device=self.device)
            self.leg_predictor = builder(model_config, checkpoint, device=self.device)
        finally:
            for path in sam_paths:
                if path in sys.path:
                    sys.path.remove(path)

        depth = self.settings["depth_anything"]
        source = resolve_package_path(depth["source"])
        sys.path.insert(0, str(source))
        try:
            module = importlib.import_module("depth_anything_v2.dpt")
        finally:
            sys.path.pop(0)
        encoder = depth.get("encoder", "vits")
        self.depth_model = module.DepthAnythingV2(**self.MODEL_CONFIGS[encoder])
        payload = torch.load(
            str(resolve_package_path(depth["checkpoint"])), map_location="cpu"
        )
        self.depth_model.load_state_dict(payload)
        self.depth_model = self.depth_model.to(self.device).eval()

    @staticmethod
    def _points(points: Sequence[Sequence[float]], labels: Optional[Sequence[int]]) -> tuple:
        values = np.asarray(points, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
            raise ValueError("at least one (x, y) prompt point is required per object")
        label_values = (
            np.ones(len(values), dtype=np.int32)
            if labels is None
            else np.asarray(labels, dtype=np.int32)
        )
        if label_values.shape != (len(values),) or not set(label_values.tolist()) <= {0, 1}:
            raise ValueError("prompt labels must contain one 0/1 value per point")
        return values, label_values

    def initialize(
        self,
        frame: np.ndarray,
        *,
        sock_points: Sequence[Sequence[float]],
        leg_points: Sequence[Sequence[float]],
        sock_labels: Optional[Sequence[int]] = None,
        leg_labels: Optional[Sequence[int]] = None,
        renderer_masks: Optional[Mapping[str, np.ndarray]] = None,
    ) -> PerceptionResult:
        self.load()
        sock_points, sock_labels = self._points(sock_points, sock_labels)
        leg_points, leg_labels = self._points(leg_points, leg_labels)
        model_frame = np.asarray(frame)[..., ::-1].copy()
        with self._sam_context():
            self.sock_predictor.load_first_frame(model_frame)
            _, _, sock_logits = self.sock_predictor.add_new_points(
                0, 1, sock_points, sock_labels
            )
            self.leg_predictor.load_first_frame(model_frame)
            _, _, leg_logits = self.leg_predictor.add_new_points(
                0, 1, leg_points, leg_labels
            )
        self._initialized = True
        self._prompt_points = {
            "sock": sock_points[sock_labels == 1].tolist(),
            "leg": leg_points[leg_labels == 1].tolist(),
        }
        return self._result(frame, sock_logits, leg_logits, renderer_masks)

    def track(
        self,
        frame: np.ndarray,
        renderer_masks: Optional[Mapping[str, np.ndarray]] = None,
    ) -> PerceptionResult:
        if not self._initialized:
            raise RuntimeError("initialize prompts before tracking")
        model_frame = np.asarray(frame)[..., ::-1].copy()
        with self._sam_context():
            _, sock_logits = self.sock_predictor.track(model_frame)
            _, leg_logits = self.leg_predictor.track(model_frame)
        return self._result(frame, sock_logits, leg_logits, renderer_masks)

    def _sam_context(self):
        try:
            import torch
        except ImportError:
            return contextlib.nullcontext()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _mask(self, logits) -> np.ndarray:
        threshold = float(self.settings["sam2"].get("threshold", 0.2))
        value = logits[0] if getattr(logits, "ndim", 0) >= 3 else logits
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        value = np.asarray(value)
        value = np.squeeze(value)
        if value.ndim != 2:
            raise ValueError(f"SAM2 mask logits must reduce to HxW, got {value.shape}")
        return value > threshold

    def _depth(self, frame: np.ndarray) -> np.ndarray:
        try:
            import torch
        except ImportError:
            torch = None
        no_grad = torch.no_grad() if torch is not None else contextlib.nullcontext()
        with no_grad:
            raw = self.depth_model.infer_image(
                np.asarray(frame)[..., ::-1].copy(), input_size=518
            )
        normalized = (np.asarray(raw, dtype=np.float32) - self.depth_min) / (
            self.depth_max - self.depth_min
        )
        return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)

    def _result(
        self,
        frame: np.ndarray,
        sock_logits,
        leg_logits,
        renderer_masks: Optional[Mapping[str, np.ndarray]] = None,
    ) -> PerceptionResult:
        sock = self._mask(sock_logits)
        leg = self._mask(leg_logits)
        if self.settings.get("semantic_mask", {}).get(
            "keep_prompt_components", False
        ):
            sock = prompt_components(sock, self._prompt_points["sock"])
            leg = prompt_components(leg, self._prompt_points["leg"])
        if sock.shape != frame.shape[:2] or leg.shape != frame.shape[:2]:
            raise ValueError("SAM2 masks do not match the RGB frame")
        quality = assess_masks(
            sock,
            leg,
            min_fraction=float(self.settings.get("mask_min_fraction", 0.0001)),
            max_fraction=float(self.settings.get("mask_max_fraction", 0.95)),
            previous_areas=self._previous_areas,
            max_area_change=float(self.settings.get("mask_max_area_change", 4.0)),
            prompt_points=self._prompt_points,
            renderer_masks=renderer_masks,
            semantic=self.settings.get("semantic_mask", {}),
        )
        if not quality["ok"]:
            raise RuntimeError("perception mask quality failed: " + json.dumps(quality))
        self._previous_areas = tuple(quality["areas"])
        depth = self._depth(frame)
        return PerceptionResult(
            depth=depth,
            sock_mask=sock,
            leg_mask=leg,
            sock_depth=masked_depth(depth, sock),
            leg_depth=masked_depth(depth, leg),
            quality=quality,
        )
