"""End-to-end reproducibility runner.

- Seeds with jax.random.key(seed)
- Deterministic dataset (no shuffling)
- Saves RNG + TrainState in Orbax checkpoint (optional)
- Emits a JSON with param hash and final metrics for comparison
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from flax.training import train_state as ts

from .rng import SeedConfig, make_key, next_key
from .data import DataConfig, make_dataset, loader
from .model import create_state, compute_metrics, cross_entropy_loss
from .checkpoint import save as save_ckpt, restore as restore_ckpt
from .hashing import params_hash


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunCfg:
    steps: int
    batch_size: int
    input_dim: int
    num_classes: int
    lr: float
    seed: int
    out: Path
    checkpoint_dir: Path | None
    restore_from: int | None
    save_at: int | None


def parse_args() -> RunCfg:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--input-dim", type=int, default=32)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--restore-from", type=int, default=None, help="Restore step to resume from (if provided).")
    p.add_argument("--save-at", type=int, default=None, help="Optional save step (e.g., 40).")
    a = p.parse_args()
    return RunCfg(a.steps, a.batch_size, a.input_dim, a.num_classes, a.lr, a.seed, a.out, a.checkpoint_dir, a.restore_from, a.save_at)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@jit
def train_step(state: ts.TrainState, xb: jnp.ndarray, y: jnp.ndarray) -> tuple[ts.TrainState, dict[str, jnp.ndarray]]:
    def loss_fn(p):
        logits = state.apply_fn({"params": p}, xb)
        return cross_entropy_loss(logits, y)

    loss, grads = value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    logits = state.apply_fn({"params": new_state.params}, xb)
    metrics = compute_metrics(logits, y)
    return new_state, metrics


def run(cfg: RunCfg):
    setup_logging()
    logger.info("Backend: %s Devices: %s", jax.default_backend(), jax.devices())

    X, y = make_dataset(DataConfig(num_samples=8_192, input_dim=cfg.input_dim, num_classes=cfg.num_classes, seed=cfg.seed))
    it = loader(X, y, cfg.batch_size)

    key = make_key(SeedConfig(seed=cfg.seed, process_index=0))
    state = create_state(key, cfg.input_dim, cfg.num_classes, lr=cfg.lr)

    start_step = 0
    if cfg.checkpoint_dir is not None and cfg.restore_from is not None:
        # Create target structure for proper deserialization
        target = {"state": state, "rng": key}
        payload = restore_ckpt(cfg.checkpoint_dir, cfg.restore_from, target=target)
        state = payload["state"]
        key = payload["rng"]
        start_step = cfg.restore_from
        logger.info("Resumed from step=%d", cfg.restore_from)
        # Advance the data loader to the correct position
        for _ in range(start_step):
            next(it)

    for step in range(start_step + 1, cfg.steps + 1):
        xb, yb = next(it)
        state, metrics = train_step(state, xb, yb)
        logger.info("step=%d loss=%.4f acc=%.3f", step, float(metrics["loss"]), float(metrics["acc"]))

        if cfg.checkpoint_dir is not None and cfg.save_at is not None and step == cfg.save_at:
            save_ckpt(cfg.checkpoint_dir, step, state=state, rng_key=key)
            logger.info("Saved checkpoint at step=%d", step)

        key, _ = next_key(key)

    h = params_hash(state.params)
    result = {
        "hash": h,
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "lr": cfg.lr,
        "backend": jax.default_backend(),
        "final": {"loss": float(metrics["loss"]), "acc": float(metrics["acc"])},
    }
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", cfg.out)
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
