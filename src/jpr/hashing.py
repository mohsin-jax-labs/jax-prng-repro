"""Deterministic hashing of parameter PyTrees to compare runs."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import jax
import jax.numpy as jnp


def _iter_leaves(params: Any) -> Iterable[jnp.ndarray]:
    for leaf in jax.tree_util.tree_leaves(params):
        if isinstance(leaf, jnp.ndarray):
            yield leaf
        else:
            yield jnp.asarray(leaf)


def params_hash(params: Any) -> str:
    """Compute a stable SHA256 over all leaves (dtype/shape/data)."""
    h = hashlib.sha256()
    for leaf in _iter_leaves(params):
        h.update(str(leaf.dtype).encode())
        h.update(str(leaf.shape).encode())
        h.update(leaf.tobytes())
    return h.hexdigest()
