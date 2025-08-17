"""Orbax wrappers for saving/restoring TrainState + RNG key."""

from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

import orbax.checkpoint as ocp


logger = logging.getLogger(__name__)


def save(ckpt_dir: str | Path, step: int, *, state: Any, rng_key: Any) -> None:
    ckpt_dir = Path(ckpt_dir).resolve()  # Convert to absolute path
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    payload = {"state": state, "rng": rng_key}
    logger.info("Saving checkpoint step=%d -> %s", step, ckpt_dir)
    checkpointer.save(ckpt_dir / str(step), args=ocp.args.PyTreeSave(payload))


def restore(ckpt_dir: str | Path, step: int, target: Any | None = None) -> dict[str, Any]:
    ckpt_dir = Path(ckpt_dir).resolve()  # Convert to absolute path
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    logger.info("Restoring checkpoint step=%d <- %s", step, ckpt_dir)
    if target is not None:
        payload = checkpointer.restore(ckpt_dir / str(step), args=ocp.args.PyTreeRestore(target))
    else:
        payload = checkpointer.restore(ckpt_dir / str(step))
    if not isinstance(payload, dict) or "state" not in payload or "rng" not in payload:
        raise ValueError("Invalid checkpoint payload.")
    return payload
