#!/usr/bin/env python3
"""Build episode-level manifests for the autonomous three-way comparison."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
REAL_ROOT = ROOT.parent / "ShareSet" / "share_folder" / "data"
REAL_MANIFEST = REAL_ROOT / "dataset_sock.yaml"
EXPERT_ROOT = ROOT / "artifacts" / "autonomous-comparison" / "expert-data"
OUTPUT_ROOT = ROOT / "artifacts" / "autonomous-comparison" / "manifests"


def _real_episodes() -> dict[str, dict]:
    source = yaml.safe_load(REAL_MANIFEST.read_text(encoding="utf-8"))
    source = source["data_center_general_depth_n5_sock"]
    result: dict[str, dict] = {"train": {}, "test": {}}
    for split in result:
        for name, window in source[split].items():
            result[split][f"real_{name}"] = {
                "path": str(
                    (REAL_ROOT / "data_center_general_depth_n5_sock" / split / name)
                    .resolve()
                ),
                "start": int(window["start"]),
                "end": int(window["end"]),
            }
    return result


def _simulation_episodes() -> tuple[dict[str, dict], list[dict]]:
    accepted = []
    for report_path in sorted(EXPERT_ROOT.glob("dressing_trial_*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        episode = report.get("training_episode")
        if (
            episode
            and int(report.get("training_frames", 0)) >= 50
        ):
            accepted.append(
                {
                    "report": str(report_path.resolve()),
                    "episode": str(Path(episode).resolve()),
                    "seed": report.get("parameters", {}).get("seed"),
                    "physical_sock_dressing_success": report.get(
                        "physical_sock_dressing_success"
                    ),
                    "coverage_gain": report.get("coverage_gain"),
                    "maximum_stretch": report.get("maximum_stretch"),
                }
            )
    if len(accepted) < 11:
        raise RuntimeError(
            f"need 11 successful 50-frame expert episodes, found {len(accepted)}"
        )
    accepted = accepted[:11]
    result: dict[str, dict] = {"train": {}, "test": {}}
    for index, item in enumerate(accepted):
        split = "train" if index < 9 else "test"
        result[split][f"sim_expert_{index:02d}"] = {
            "path": item["episode"],
            "start": 0,
            "end": 50,
        }
        item["split"] = split
    return result, accepted


def _write(name: str, train: dict, test: dict) -> None:
    short_name = name[len("autonomous_") :] if name.startswith("autonomous_") else name
    path = OUTPUT_ROOT / f"{short_name}.yaml"
    path.write_text(
        yaml.safe_dump({name: {"train": train, "test": test}}, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    real = _real_episodes()
    simulation, inventory = _simulation_episodes()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write("autonomous_real_only", real["train"], real["test"])
    _write("autonomous_sim_only", simulation["train"], simulation["test"])
    _write(
        "autonomous_real_sim",
        {**real["train"], **simulation["train"]},
        {**real["test"], **simulation["test"]},
    )
    (OUTPUT_ROOT / "expert_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "accepted_experts": len(inventory)}, indent=2))


if __name__ == "__main__":
    main()
