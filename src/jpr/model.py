"""Tiny Flax MLP classifier and metrics."""

from __future__ import annotations

import jax
import optax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass
from flax.training import train_state as ts


@dataclass(frozen=True, slots=True)
class MLPConfig:
    hidden: tuple[int, ...] = (64, 32)
    num_classes: int = 10


class MLP(nn.Module):
    cfg: MLPConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for h in self.cfg.hidden:
            x = nn.Dense(h, kernel_init=nn.initializers.lecun_normal())(x)
            x = nn.relu(x)
        return nn.Dense(self.cfg.num_classes)(x)


def create_state(rng: jax.Array, input_dim: int, num_classes: int, lr: float = 1e-3) -> ts.TrainState:
    model = MLP(MLPConfig(num_classes=num_classes))
    params = model.init(rng, jnp.ones((1, input_dim), jnp.float32))["params"]
    tx = optax.adam(lr)
    return ts.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def cross_entropy_loss(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    onehot = jax.nn.one_hot(y, logits.shape[-1])
    return -jnp.mean(jnp.sum(onehot * jax.nn.log_softmax(logits), axis=-1))


def compute_metrics(logits: jnp.ndarray, y: jnp.ndarray) -> dict[str, jnp.ndarray]:
    loss = cross_entropy_loss(logits, y)
    acc = jnp.mean((jnp.argmax(logits, axis=-1) == y).astype(jnp.float32))
    return {"loss": loss, "acc": acc}
