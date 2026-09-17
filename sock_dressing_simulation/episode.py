from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from PIL import Image

SIGNALS = ("angle", "torque", "external_torque")
IMAGE_DIRS = (
    "camera_right",
    "camera_right_mask/sock_mask",
    "camera_right_mask/leg_mask",
    "camera_depth",
    "depth_mask/sock_depth",
    "depth_mask/leg_depth",
)


class EpisodeWriter:
    """Write one frame-aligned episode without importing ShareSet."""

    def __init__(self, path: Path, joint_names: Sequence[str], metadata: Optional[Mapping] = None):
        self.path = Path(path)
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != 18:
            raise ValueError("ShareSet-compatible episodes require 18 joints")
        self.path.mkdir(parents=True, exist_ok=False)
        for directory in IMAGE_DIRS:
            (self.path / directory).mkdir(parents=True)
        self._files = {
            signal: (self.path / f"{signal}.csv").open("w", newline="", encoding="utf-8")
            for signal in SIGNALS
        }
        self._writers = {key: csv.writer(value) for key, value in self._files.items()}
        self._metadata = dict(metadata or {})
        self._metadata.update(
            {
                "schema": "shareset-sock-episode-v1",
                "joint_names": list(self.joint_names),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "depth_encoding": "uint8 linear between configured near/far planes",
            }
        )
        self.frames = 0
        self._closed = False

    def update_metadata(self, values: Mapping) -> None:
        if self._closed:
            raise RuntimeError("episode is closed")
        self._metadata.update(dict(values))

    def append(
        self,
        *,
        rgb: np.ndarray,
        sock_mask: np.ndarray,
        leg_mask: np.ndarray,
        camera_depth: np.ndarray,
        angle: Sequence[float],
        torque: Sequence[float],
        external_torque: Sequence[float],
    ) -> None:
        if self._closed:
            raise RuntimeError("episode is closed")
        rgb = np.asarray(rgb)
        depth = np.asarray(camera_depth)
        sock = np.asarray(sock_mask, dtype=bool)
        leg = np.asarray(leg_mask, dtype=bool)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")
        if depth.shape != rgb.shape[:2] or sock.shape != depth.shape or leg.shape != depth.shape:
            raise ValueError("depth and masks must match RGB height and width")
        if depth.dtype != np.uint8:
            raise ValueError("camera_depth must be a ShareSet-compatible uint8 image")
        index = str(self.frames)
        Image.fromarray(rgb.astype(np.uint8), "RGB").save(self.path / "camera_right" / f"{index}.png")
        Image.fromarray(sock.astype(np.uint8) * 255, "L").save(
            self.path / "camera_right_mask" / "sock_mask" / f"{index}.png"
        )
        Image.fromarray(leg.astype(np.uint8) * 255, "L").save(
            self.path / "camera_right_mask" / "leg_mask" / f"{index}.png"
        )
        Image.fromarray(depth, "L").save(self.path / "camera_depth" / f"{index}.png")
        Image.fromarray(np.where(sock, depth, 0).astype(np.uint8), "L").save(
            self.path / "depth_mask" / "sock_depth" / f"{index}.png"
        )
        Image.fromarray(np.where(leg, depth, 0).astype(np.uint8), "L").save(
            self.path / "depth_mask" / "leg_depth" / f"{index}.png"
        )
        for name, values in (
            ("angle", angle),
            ("torque", torque),
            ("external_torque", external_torque),
        ):
            row = np.asarray(values, dtype=float)
            if row.shape != (18,) or not np.all(np.isfinite(row)):
                raise ValueError(f"{name} must contain 18 finite values")
            self._writers[name].writerow(row.tolist())
            self._files[name].flush()
        self.frames += 1

    def close(self) -> None:
        if self._closed:
            return
        for stream in self._files.values():
            stream.close()
        self._metadata["frames"] = self.frames
        (self.path / "metadata.json").write_text(
            json.dumps(self._metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._closed = True

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
