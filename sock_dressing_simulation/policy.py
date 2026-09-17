from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml
from PIL import Image

from .config import resolve_package_path


def normalize(value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    scale = high - low
    if value.shape[-1] != 36 or low.shape != (36,) or high.shape != (36,):
        raise ValueError("SAMDAMSARNN state and statistics must have 36 channels")
    if np.any(scale <= 0):
        raise ValueError("normalization range contains a non-positive channel")
    return (value - low) / scale


def denormalize(value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    if value.shape[-1] != 36:
        raise ValueError("SAMDAMSARNN output must have 36 channels")
    return value * (high - low) + low


def _resize_rgb(value: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(value, dtype=np.uint8), "RGB")
    result = np.asarray(image.resize((size, size)), dtype=np.float32)
    return result.transpose(2, 0, 1) / 255.0


def _resize_depth(value: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(value, dtype=np.uint8), "L")
    result = np.asarray(image.resize((size, size)), dtype=np.float32)
    return result[None, ...] / 255.0


class SAMDAMSARNNPolicy:
    def __init__(
        self,
        config: Mapping,
        *,
        checkpoint: Path = None,
        stats: Path = None,
        device: str = None,
        model=None,
    ):
        import torch

        settings = config["inference"]
        self.device = torch.device(device or settings.get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.image_size = int(settings.get("image_size", 64))
        self.checkpoint = resolve_package_path(
            str(checkpoint) if checkpoint else settings["checkpoint"]
        )
        stats_path = resolve_package_path(str(stats) if stats else settings["stats"])
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        self.low = np.asarray(payload["joint_min"], dtype=np.float32)
        self.high = np.asarray(payload["joint_max"], dtype=np.float32)
        normalize(np.zeros(36, dtype=np.float32), self.low, self.high)
        self.state = None
        self.model = model or self._load_model(config)
        self.model = self.model.to(self.device).eval()
        self.checkpoint_sha256 = (
            hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
            if self.checkpoint.is_file()
            else "injected-model"
        )

    def _load_model(self, config: Mapping):
        import torch

        settings = config["inference"]
        training_config = yaml.safe_load(
            resolve_package_path(settings["training_config"]).read_text(encoding="utf-8")
        )["model"]
        source = resolve_package_path(settings["shareset_src"])
        sys.path.insert(0, str(source))
        try:
            module = importlib.import_module("models.SAMDAMSARNN")
        finally:
            sys.path.pop(0)
        model = module.SAMDAMSARNN(
            rec_dim=int(training_config["rec_dim"]),
            k_dim=int(training_config["k_dim"]),
            joint_dim=36,
            temperature=float(training_config["temperature"]),
            heatmap_size=float(training_config["heatmap_size"]),
            im_size=[self.image_size, self.image_size],
        )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        payload = torch.load(str(self.checkpoint), map_location=self.device)
        model.load_state_dict(payload.get("model_state_dict", payload))
        return model

    def reset(self) -> None:
        self.state = None

    def step(
        self,
        *,
        rgb: np.ndarray,
        sock_depth: np.ndarray,
        leg_depth: np.ndarray,
        angle: np.ndarray,
        torque: np.ndarray,
        step_index: int = 0,
    ) -> dict:
        import torch

        angle = np.asarray(angle, dtype=np.float32)
        torque = np.asarray(torque, dtype=np.float32)
        if angle.shape != (18,) or torque.shape != (18,):
            raise ValueError("angle and torque must each have 18 channels")
        measured = np.concatenate((angle, torque))
        normalized_state = normalize(measured, self.low, self.high)
        arrays = (
            _resize_rgb(rgb, self.image_size),
            normalized_state,
            _resize_depth(sock_depth, self.image_size),
            _resize_depth(leg_depth, self.image_size),
        )
        tensors = [
            torch.from_numpy(value[None].astype(np.float32)).to(self.device)
            for value in arrays
        ]
        with torch.no_grad():
            predicted_image, predicted_joint, _, _, self.state, _ = self.model(
                *tensors, self.state
            )
        prediction = predicted_joint[0].detach().cpu().numpy()
        # Preserve the warm-up blend used by the real AIREC controller.
        if 5 < step_index < 15:
            measured_rate = 0.8 - 0.08 * (step_index - 6)
            prediction = prediction * (1.0 - measured_rate) + normalized_state * measured_rate
        decoded = denormalize(prediction, self.low, self.high)
        return {
            "action": decoded[:18].astype(np.float64),
            "predicted_torque": decoded[18:].astype(np.float64),
            "normalized_prediction": prediction,
            "predicted_image": predicted_image[0].detach().cpu().numpy(),
        }
