#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "fx",
    "fy",
    "fz",
    "tx",
    "ty",
    "tz",
    "abs_fz",
    "force_xy_l1",
    "torque_l1",
)
TARGET_NAMES = ("angle_x_deg", "angle_y_deg", "angle_z_deg")


def build_wrench_features(force_xyz: np.ndarray, torque_xyz: np.ndarray) -> np.ndarray:
    force = np.asarray(force_xyz, dtype=np.float64).reshape(3)
    torque = np.asarray(torque_xyz, dtype=np.float64).reshape(3)
    return np.array(
        [
            force[0],
            force[1],
            force[2],
            torque[0],
            torque[1],
            torque[2],
            abs(force[2]),
            abs(force[0]) + abs(force[1]),
            np.linalg.norm(torque, ord=1),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class AngleWrenchModel:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def predict_angle_deg(self, force_xyz: np.ndarray, torque_xyz: np.ndarray) -> np.ndarray:
        x = build_wrench_features(force_xyz, torque_xyz)
        x_norm = (x - self.x_mean) / np.maximum(self.x_std, 1e-6)
        hidden = np.tanh(x_norm @ self.w1 + self.b1)
        y_norm = hidden @ self.w2 + self.b2
        return y_norm * np.maximum(self.y_std, 1e-6) + self.y_mean

    @classmethod
    def load(cls, path: str | Path) -> "AngleWrenchModel":
        data = np.load(Path(path), allow_pickle=False)
        feature_names = tuple(str(v) for v in data["feature_names"])
        target_names = tuple(str(v) for v in data["target_names"])
        if feature_names != FEATURE_NAMES:
            raise ValueError(
                f"Feature mismatch: model has {feature_names}, expected {FEATURE_NAMES}"
            )
        if target_names != TARGET_NAMES:
            raise ValueError(
                f"Target mismatch: model has {target_names}, expected {TARGET_NAMES}"
            )
        return cls(
            x_mean=data["x_mean"].astype(np.float64),
            x_std=data["x_std"].astype(np.float64),
            y_mean=data["y_mean"].astype(np.float64),
            y_std=data["y_std"].astype(np.float64),
            w1=data["w1"].astype(np.float64),
            b1=data["b1"].astype(np.float64),
            w2=data["w2"].astype(np.float64),
            b2=data["b2"].astype(np.float64),
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            feature_names=np.asarray(FEATURE_NAMES),
            target_names=np.asarray(TARGET_NAMES),
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
        )
