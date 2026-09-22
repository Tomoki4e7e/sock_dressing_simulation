from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import yaml
from PIL import Image, ImageFilter

from .config import resolve_package_path


@dataclass(frozen=True)
class TrainingEpisode:
    name: str
    split: str
    directory: Path
    start: int
    end: int


def load_training_specs(config: Mapping) -> list:
    settings = config["inference"]
    root = resolve_package_path(settings["training_data_root"])
    manifest_path = resolve_package_path(settings["training_manifest"])
    dataset_name = settings["training_dataset_name"]
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if dataset_name not in payload:
        raise KeyError(f"{dataset_name!r} is absent from {manifest_path}")
    specs = []
    for split in ("train", "test"):
        for name, window in payload[dataset_name].get(split, {}).items():
            specs.append(
                TrainingEpisode(
                    name=str(name),
                    split=split,
                    directory=(
                        resolve_package_path(window["path"])
                        if window.get("path")
                        else root / dataset_name / split / str(name)
                    ),
                    start=int(window["start"]),
                    end=int(window["end"]),
                )
            )
    if not any(spec.split == "train" for spec in specs):
        raise ValueError("training manifest contains no train episodes")
    if not any(spec.split == "test" for spec in specs):
        raise ValueError("training manifest contains no test episodes")
    train_names = {spec.name for spec in specs if spec.split == "train"}
    test_names = {spec.name for spec in specs if spec.split == "test"}
    overlap = train_names & test_names
    if overlap:
        raise ValueError(f"episodes must not cross train/test splits: {sorted(overlap)}")
    return specs


def _numbered_images(directory: Path) -> Dict[int, Path]:
    if not directory.is_dir():
        return {}
    result = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == ".png":
            try:
                result[int(path.stem)] = path
            except ValueError:
                continue
    return result


def audit_training_data(config: Mapping) -> Dict[str, Any]:
    records = []
    ok = True
    for spec in load_training_specs(config):
        errors = []
        if spec.end <= spec.start:
            errors.append("invalid frame window")
        signals = {}
        for name in ("angle.csv", "torque.csv"):
            path = spec.directory / name
            try:
                values = np.loadtxt(str(path), delimiter=",", ndmin=2)
                signals[name] = list(values.shape)
                if values.ndim != 2 or values.shape[1] != 18:
                    errors.append(f"{name} must have 18 columns")
                if values.shape[0] < spec.end:
                    errors.append(f"{name} has fewer than {spec.end} rows")
            except (OSError, ValueError) as error:
                errors.append(f"{name}: {error}")
        modalities = {
            "rgb": spec.directory / "camera_right",
            "camera_depth": spec.directory / "camera_depth",
            "sock_depth": spec.directory / "depth_mask" / "sock_depth",
            "foot_depth": spec.directory / "depth_mask" / "foot_depth",
        }
        # Simulation episodes use leg_depth; original real episodes use foot_depth.
        if not modalities["foot_depth"].is_dir():
            modalities["foot_depth"] = spec.directory / "depth_mask" / "leg_depth"
        counts = {}
        required = set(range(spec.start, spec.end))
        for name, directory in modalities.items():
            available = set(_numbered_images(directory))
            counts[name] = len(available)
            missing = required - available
            if missing:
                errors.append(f"{name} missing {len(missing)} requested frames")
        records.append(
            {
                "episode": spec.name,
                "split": spec.split,
                "window": [spec.start, spec.end],
                "signals": signals,
                "image_counts": counts,
                "errors": errors,
            }
        )
        ok = ok and not errors
    return {"ok": ok, "episodes": records, "n_episodes": len(records)}


def _load_signal(path: Path, start: int, end: int) -> np.ndarray:
    values = np.loadtxt(str(path), delimiter=",", dtype=np.float32, ndmin=2)
    if values.shape[1] != 18:
        raise ValueError(f"{path} must have 18 columns, got {values.shape}")
    return values[start:end]


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        raise ValueError("smooth_torque must be odd")
    if len(values) < window:
        raise ValueError(f"sequence length {len(values)} is shorter than smoothing window {window}")
    try:
        from scipy.signal import savgol_filter
    except ImportError as error:
        raise RuntimeError("training requires scipy for torque smoothing") from error
    return savgol_filter(values, window_length=window, polyorder=2, axis=0).astype(np.float32)


def _load_image(path: Path, size: int, depth: bool) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L" if depth else "RGB")
        if depth:
            image = image.filter(ImageFilter.GaussianBlur(radius=15))
        image = image.crop((0, 0, 1280, 960)).resize((size, size))
        value = np.asarray(image, dtype=np.float32)
    if depth:
        value = value[None, ...]
    else:
        value = value.transpose(2, 0, 1)
    return value / 255.0


def load_episode_arrays(
    spec: TrainingEpisode, *, image_size: int, smooth_torque: int, skip_num: int
) -> tuple:
    angle = _load_signal(spec.directory / "angle.csv", spec.start, spec.end)
    torque = _smooth(
        _load_signal(spec.directory / "torque.csv", spec.start, spec.end),
        smooth_torque,
    )
    foot_dir = spec.directory / "depth_mask" / "foot_depth"
    if not foot_dir.is_dir():
        foot_dir = spec.directory / "depth_mask" / "leg_depth"
    indices = range(spec.start, spec.end, skip_num)
    rgb = np.stack(
        [_load_image(spec.directory / "camera_right" / f"{i}.png", image_size, False) for i in indices]
    )
    sock = np.stack(
        [_load_image(spec.directory / "depth_mask" / "sock_depth" / f"{i}.png", image_size, True) for i in indices]
    )
    leg = np.stack([_load_image(foot_dir / f"{i}.png", image_size, True) for i in indices])
    joints = np.concatenate((angle, torque), axis=1)[::skip_num]
    if not (len(rgb) == len(sock) == len(leg) == len(joints)):
        raise ValueError(f"frame alignment failed for {spec.directory}")
    return rgb, joints.astype(np.float32), sock, leg


class _SequenceDataset:
    def __init__(self, episodes: Iterable[tuple], *, stdev: float, training: bool):
        import torch

        self.episodes = [
            tuple(torch.from_numpy(np.asarray(value, dtype=np.float32)) for value in episode)
            for episode in episodes
        ]
        self.stdev = float(stdev)
        self.training = training

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int):
        import torch

        targets = self.episodes[index]
        if self.training and self.stdev:
            inputs = tuple(value + torch.randn_like(value) * self.stdev for value in targets)
        else:
            inputs = targets
        return inputs, targets


def _import_shareset(shareset_src: Path):
    sys.path.insert(0, str(shareset_src))
    try:
        model_module = importlib.import_module("models.SAMDAMSARNN")
        trainer_module = importlib.import_module("models.fullBPTT")
    finally:
        sys.path.pop(0)
    return model_module.SAMDAMSARNN, trainer_module.fullBPTTtrainerSAM2


def train_policy(
    config: Mapping,
    *,
    epochs: Optional[int] = None,
    device: Optional[str] = None,
    resume: Optional[Path] = None,
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    audit = audit_training_data(config)
    if not audit["ok"]:
        raise RuntimeError("training data audit failed: " + json.dumps(audit, ensure_ascii=False))
    settings = config["inference"]
    training_config_path = resolve_package_path(settings["training_config"])
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    model_config = training_config["model"]
    hyper = training_config["hyperparameters"]
    image_size = int(settings.get("image_size", model_config["img_size"]))
    skip_num = int(hyper["skip_num"])
    specs = load_training_specs(config)
    arrays = {
        split: [
            load_episode_arrays(
                spec,
                image_size=image_size,
                smooth_torque=int(hyper["smooth_torque"]),
                skip_num=skip_num,
            )
            for spec in specs
            if spec.split == split
        ]
        for split in ("train", "test")
    }
    train_joints = np.concatenate([episode[1] for episode in arrays["train"]], axis=0)
    preserve_stats = bool(settings.get("preserve_resume_stats", False))
    if resume and preserve_stats:
        stats_payload = json.loads(
            resolve_package_path(settings["stats"]).read_text(encoding="utf-8")
        )
        low = np.asarray(stats_payload["joint_min"], dtype=np.float32)
        high = np.asarray(stats_payload["joint_max"], dtype=np.float32)
    else:
        low = train_joints.min(axis=0)
        high = train_joints.max(axis=0)
    if np.any(high <= low):
        raise ValueError("normalization contains a zero-width channel")

    def normalized(episodes):
        return [
            (rgb, (joints - low) / (high - low), sock, leg)
            for rgb, joints, sock, leg in episodes
        ]

    batch_size = min(int(hyper["batch_size"]), len(arrays["train"]))
    train_loader = DataLoader(
        _SequenceDataset(normalized(arrays["train"]), stdev=float(hyper["stdev"]), training=True),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        _SequenceDataset(normalized(arrays["test"]), stdev=0.0, training=False),
        batch_size=min(batch_size, len(arrays["test"])),
        shuffle=False,
    )
    shareset_src = resolve_package_path(settings["shareset_src"])
    Model, Trainer = _import_shareset(shareset_src)
    model = Model(
        rec_dim=int(model_config["rec_dim"]),
        k_dim=int(model_config["k_dim"]),
        joint_dim=36,
        temperature=float(model_config["temperature"]),
        heatmap_size=float(model_config["heatmap_size"]),
        im_size=[image_size, image_size],
    )
    selected_device = device or settings.get("device", "cuda")
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu explicitly")
    selected_epochs = int(epochs if epochs is not None else hyper["num_epochs"])
    if selected_epochs < 1:
        raise ValueError("epochs must be positive")
    if config["rcareworld"].get("profile") == "dressing_player":
        if selected_epochs != 10000 or selected_device != "cuda":
            raise ValueError(
                "DressingPlayer production training requires --epochs 10000 --device cuda"
            )
    output = resolve_package_path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hyper["learning_rate"]),
        weight_decay=float(hyper["weight_decay"]),
    )
    start_epoch = 0
    if resume:
        payload = torch.load(str(resume), map_location=selected_device)
        model.load_state_dict(payload.get("model_state_dict", payload))
        start_epoch = int(payload.get("epoch", -1)) + 1
        if "optimizer_state_dict" in payload and settings.get(
            "resume_optimizer", True
        ):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
    trainer = Trainer(
        model,
        optimizer,
        loss_weights=[
            float(hyper["img_loss"]),
            float(hyper["angle_loss"]),
            float(hyper["torque_loss"]),
            float(hyper["pt_loss"]),
        ],
        device=selected_device,
    )
    history = []
    latest = output / "SARNN_latest.pth"
    for epoch in range(start_epoch, selected_epochs):
        train_loss, _ = trainer.process_epoch(train_loader, training=True)
        test_loss, _ = trainer.process_epoch(test_loader, training=False)
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        torch.save(payload, latest)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
    stats = {"joint_min": low.tolist(), "joint_max": high.tolist()}
    (output / "data.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    parameters = {
        "model": model_config,
        "hyperparameters": hyper,
        "epochs_requested": selected_epochs,
        "training_manifest": str(resolve_package_path(settings["training_manifest"])),
    }
    (output / "parameter.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )
    (output / "loss.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(latest.read_bytes()).hexdigest() if latest.is_file() else None
    return {
        "ok": latest.is_file(),
        "checkpoint": str(latest),
        "stats": str(output / "data.json"),
        "checkpoint_sha256": digest,
        "epochs_completed": len(history),
        "data_audit": audit,
    }
