#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from aic.toolkit.angle_wrench.angle_wrench_model import AngleWrenchModel, FEATURE_NAMES, TARGET_NAMES
    from aic.toolkit.angle_wrench.angle_wrench_model import build_wrench_features
except ImportError:
    from .angle_wrench_model import AngleWrenchModel, FEATURE_NAMES, TARGET_NAMES
    from .angle_wrench_model import build_wrench_features


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(("fx", "fy", "fz", "tx", "ty", "tz", *TARGET_NAMES)) - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            force = np.array([float(row["fx"]), float(row["fy"]), float(row["fz"])])
            torque = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            target = np.array([float(row[name]) for name in TARGET_NAMES])
            if not np.all(np.isfinite(force)) or not np.all(np.isfinite(torque)):
                continue
            if not np.all(np.isfinite(target)):
                continue
            xs.append(build_wrench_features(force, torque))
            ys.append(target)

    if not xs:
        raise ValueError(f"{path} did not contain any valid training rows.")
    return np.vstack(xs), np.vstack(ys)


def _train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[AngleWrenchModel, list[float]]:
    rng = np.random.default_rng(seed)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)

    x_norm = (x - x_mean) / np.maximum(x_std, 1e-6)
    y_norm = (y - y_mean) / np.maximum(y_std, 1e-6)

    w1 = rng.normal(0.0, np.sqrt(2.0 / x_norm.shape[1]), size=(x_norm.shape[1], hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, y_norm.shape[1]))
    b2 = np.zeros(y_norm.shape[1], dtype=np.float64)

    losses = []
    n = x_norm.shape[0]
    for epoch in range(epochs):
        hidden = np.tanh(x_norm @ w1 + b1)
        pred = hidden @ w2 + b2
        err = pred - y_norm
        loss = float(np.mean(err * err))
        losses.append(loss)

        grad_pred = 2.0 * err / n
        grad_w2 = hidden.T @ grad_pred + weight_decay * w2
        grad_b2 = np.sum(grad_pred, axis=0)
        grad_hidden = (grad_pred @ w2.T) * (1.0 - hidden * hidden)
        grad_w1 = x_norm.T @ grad_hidden + weight_decay * w1
        grad_b1 = np.sum(grad_hidden, axis=0)

        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2

        if epoch > 50 and abs(losses[-2] - losses[-1]) < 1e-9:
            break

    model = AngleWrenchModel(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
    )
    return model, losses


def _evaluate(model: AngleWrenchModel, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    preds = []
    for row in x:
        force = row[:3]
        torque = row[3:6]
        preds.append(model.predict_angle_deg(force, torque))
    pred = np.vstack(preds)
    abs_err = np.abs(pred - y)
    return float(np.mean(abs_err)), np.mean(abs_err, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a small wrench-to-angle model for contact_search."
    )
    parser.add_argument("csv", type=Path, help="CSV recorded by angle_wrench_recorder.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("angle_wrench_model.npz"),
        help="Output .npz model path.",
    )
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    x, y = _load_csv(args.csv)
    model, losses = _train_mlp(
        x=x,
        y=y,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    mean_abs_err, axis_abs_err = _evaluate(model, x, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)

    print(f"Saved model: {args.output}")
    print(f"Rows: {x.shape[0]}")
    print(f"Features: {', '.join(FEATURE_NAMES)}")
    print(f"Final normalized MSE: {losses[-1]:.6f}")
    print(
        "Mean absolute angle error deg: "
        f"{mean_abs_err:.3f} "
        f"[x={axis_abs_err[0]:.3f}, y={axis_abs_err[1]:.3f}, z={axis_abs_err[2]:.3f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
