from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "default.yaml"
RCAREWORLD_COMMIT = "ae0900be3e450ae08d6137468970d0ac473a001b"
DRESSING_PLAYER_COMMIT = "3ee988f5d3535a6e87eadf71480db1d7a0b760fc"
CUSTOM_PLAYER_VERSION = "greenfield-sock-cloth-v1"
SUPPORTED_RCAREWORLD_PROFILES = {
    "canonical": RCAREWORLD_COMMIT,
    "dressing_player": DRESSING_PLAYER_COMMIT,
    "custom_player": CUSTOM_PLAYER_VERSION,
}


def _load_config_payload(config_path: Path, seen: set[Path]) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    if config_path in seen:
        chain = " -> ".join(str(path) for path in (*seen, config_path))
        raise ValueError(f"config extends cycle: {chain}")
    seen = {*seen, config_path}
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    extends = config.pop("extends", None)
    if extends:
        base_path = Path(extends).expanduser()
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base = _load_config_payload(base_path, seen)
        config = _deep_merge(base, config)
    return config


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    config = _load_config_payload(Path(path), set())
    profile = str(config["rcareworld"].get("profile", "canonical"))
    if profile not in SUPPORTED_RCAREWORLD_PROFILES:
        raise ValueError(f"unsupported RCareWorld profile: {profile}")
    commit = config["rcareworld"]["commit"]
    expected_commit = SUPPORTED_RCAREWORLD_PROFILES[profile]
    if commit != expected_commit:
        raise ValueError(
            f"RCareWorld profile {profile} requires commit {expected_commit}, got {commit}"
        )
    if "scenario" in config and profile in {"canonical", "custom_player"}:
        from .scenario import scenario_from_config

        scenario_from_config(config)
    return config


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_package_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PACKAGE_ROOT / path
