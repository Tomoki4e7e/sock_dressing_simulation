from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class JointMap:
    names: tuple
    lower: np.ndarray
    upper: np.ndarray
    max_delta: np.ndarray
    mimic: Mapping[str, Mapping[str, float]]
    simulator_indices: np.ndarray
    revolute_action_indices: tuple

    @classmethod
    def from_config(cls, config: Mapping) -> "JointMap":
        joint = config["joints"]
        result = cls(
            tuple(joint["names"]),
            np.asarray(joint["lower"], dtype=float),
            np.asarray(joint["upper"], dtype=float),
            np.asarray(joint["max_delta"], dtype=float),
            joint.get("mimic", {}),
            np.asarray(joint["simulator_indices"], dtype=int),
            tuple(int(index) for index in joint["revolute_action_indices"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if len(self.names) != 18 or len(set(self.names)) != 18:
            raise ValueError("ShareSet joint mapping must contain 18 unique names")
        for values in (self.lower, self.upper, self.max_delta):
            if values.shape != (18,):
                raise ValueError("joint bounds must each have shape (18,)")
        if self.simulator_indices.shape != (18,):
            raise ValueError("simulator_indices must have shape (18,)")
        if any(index < 0 or index >= 18 for index in self.revolute_action_indices):
            raise ValueError("revolute_action_indices must refer to action indices")
        if np.any(self.lower > self.upper) or np.any(self.max_delta <= 0):
            raise ValueError("invalid joint bounds")
        for target, rule in self.mimic.items():
            if target not in self.names or rule["source"] not in self.names:
                raise ValueError(f"unknown mimic joint in rule for {target}")
            target_index = self.names.index(target)
            source_index = self.names.index(rule["source"])
            if self.simulator_indices[target_index] != self.simulator_indices[source_index]:
                raise ValueError(
                    f"simulator mimic index for {target} must match its source"
                )
        independent = [
            int(self.simulator_indices[index])
            for index, name in enumerate(self.names)
            if name not in self.mimic
        ]
        if len(set(independent)) != len(independent):
            raise ValueError("independent simulator joint indices must be unique")

    def index(self) -> Dict[str, int]:
        return {name: index for index, name in enumerate(self.names)}

    def ordered(self, values: Mapping[str, float]) -> np.ndarray:
        missing = set(self.names) - set(values)
        if missing:
            raise KeyError(f"missing joints: {sorted(missing)}")
        return np.asarray([values[name] for name in self.names], dtype=float)

    def named(self, values: Sequence[float]) -> Dict[str, float]:
        array = self._array(values)
        return dict(zip(self.names, array.tolist()))

    def bound(self, command: Sequence[float], previous: Sequence[float] = None) -> np.ndarray:
        result = np.clip(self._array(command), self.lower, self.upper)
        if previous is not None:
            prior = self._array(previous)
            result = np.clip(result, prior - self.max_delta, prior + self.max_delta)
            result = np.clip(result, self.lower, self.upper)
        indices = self.index()
        for target, rule in self.mimic.items():
            value = (
                result[indices[rule["source"]]] * float(rule["multiplier"])
                + float(rule["offset"])
            )
            result[indices[target]] = np.clip(
                value, self.lower[indices[target]], self.upper[indices[target]]
            )
        return result

    def from_simulator(self, full_values: Sequence[float]) -> np.ndarray:
        """Select the 18 controlled joints and convert Unity degrees to radians."""
        full = np.asarray(full_values, dtype=float)
        if full.ndim != 1 or full.size <= int(self.simulator_indices.max()):
            raise ValueError("simulator joint vector is shorter than configured indices")
        result = full[self.simulator_indices].copy()
        result[list(self.revolute_action_indices)] = np.deg2rad(
            result[list(self.revolute_action_indices)]
        )
        action_indices = self.index()
        for target, rule in self.mimic.items():
            result[action_indices[target]] = (
                result[action_indices[rule["source"]]] * float(rule["multiplier"])
                + float(rule["offset"])
            )
        return result

    def merge_for_simulator(
        self, action: Sequence[float], full_previous: Sequence[float]
    ) -> np.ndarray:
        """Merge an 18-D rad/metre action into Unity's degree/metre vector."""
        result = np.asarray(full_previous, dtype=float).copy()
        if result.ndim != 1 or result.size <= int(self.simulator_indices.max()):
            raise ValueError("simulator joint vector is shorter than configured indices")
        converted = self._array(action)
        converted[list(self.revolute_action_indices)] = np.rad2deg(
            converted[list(self.revolute_action_indices)]
        )
        independent_actions = [
            index for index, name in enumerate(self.names) if name not in self.mimic
        ]
        result[self.simulator_indices[independent_actions]] = converted[
            independent_actions
        ]
        return result

    @staticmethod
    def _array(values: Sequence[float]) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape != (18,) or not np.all(np.isfinite(result)):
            raise ValueError("joint command must be 18 finite values")
        return result.copy()
