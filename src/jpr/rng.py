"""RNG utilities: seeding, splitting, and fold_in patterns."""

from __future__ import annotations

from dataclasses import dataclass
import jax
from jax import random


@dataclass(frozen=True, slots=True)
class SeedConfig:
    seed: int
    process_index: int = 0  # single-host default; replace with jax.process_index() in multi-host


def make_key(cfg: SeedConfig) -> jax.Array:
    """Create a base key and fold in process index for multi-host safety."""
    key = random.key(cfg.seed)
    if cfg.process_index != 0:
        key = random.fold_in(key, cfg.process_index)
    return key


def next_key(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split and return (new_key, subkey)."""
    new_key, sub = random.split(key)
    return new_key, sub
