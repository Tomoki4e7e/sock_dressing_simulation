from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "default.yaml"
RCAREWORLD_COMMIT = "ae0900be3e450ae08d6137468970d0ac473a001b"


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    commit = config["rcareworld"]["commit"]
    if commit != RCAREWORLD_COMMIT:
        raise ValueError(
            f"RCareWorld commit must be {RCAREWORLD_COMMIT}, got {commit}"
        )
    if "scenario" in config:
        from .scenario import scenario_from_config

        scenario_from_config(config)
    return config


def resolve_package_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PACKAGE_ROOT / path
