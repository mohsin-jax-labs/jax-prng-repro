"""Deterministic synthetic dataset using NumPy RNG seeded input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class DataConfig:
    num_samples: int = 8_192
    input_dim: int = 32
    num_classes: int = 10
    seed: int = 0


def make_dataset(cfg: DataConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create a fixed dataset from a seeded NumPy RNG (deterministic)."""
    rng = np.random.default_rng(cfg.seed)
    centroids = rng.normal(0.0, 3.0, size=(cfg.num_classes, cfg.input_dim))
    labels = rng.integers(0, cfg.num_classes, size=(cfg.num_samples,))
    X = centroids[labels] + rng.normal(0.0, 1.0, size=(cfg.num_samples, cfg.input_dim))
    y = labels.astype(np.int32)
    return jnp.asarray(X, jnp.float32), jnp.asarray(y, jnp.int32)


def loader(X: jnp.ndarray, y: jnp.ndarray, batch_size: int):
    """Simple deterministic sequential mini-batch iterator (no shuffling)."""
    n = X.shape[0]
    pos = 0
    while True:
        end = min(pos + batch_size, n)
        xb, yb = X[pos:end], y[pos:end]
        if end == n:
            pos = 0
        else:
            pos = end
        yield xb, yb
