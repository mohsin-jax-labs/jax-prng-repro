# jax-prng-repro

A **reproducibility** playbook for JAX demonstrating:

- `jax.random.key(seed)` for seeding (2025-style API).
- Folding in **process indices** for multi-host-safe seeds.
- Saving RNG alongside model state so **resume** yields the same result as uninterrupted training.
- Producing **deterministic hashes** of model params to compare runs.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e ".[dev]"
```

### 1) Run once (no checkpoint)

```bash
python -m jpr.run_repro --steps 50 --batch-size 64 --seed 0 --out runs/one.json
cat runs/one.json
```

### 2) Run again with the same seed/config — hash should match

```bash
python -m jpr.run_repro --steps 50 --batch-size 64 --seed 0 --out runs/two.json
python -m jpr.compare runs/one.json runs/two.json
```

### 3) Resume equivalence: run with a mid-checkpoint and verify equality with uninterrupted run

- Full baseline:
```bash
python -m jpr.run_repro --steps 60 --batch-size 64 --seed 123 --out runs/full.json
```

- Two-phase with checkpoint at 40:
```bash
python -m jpr.run_repro --steps 40 --batch-size 64 --seed 123 --out runs/phase1.json --checkpoint-dir checkpoints/demo --save-at 40
python -m jpr.run_repro --steps 60 --batch-size 64 --seed 123 --out runs/resumed.json --checkpoint-dir checkpoints/demo --restore-from 40
```

- Compare:
```bash
python -m jpr.compare runs/full.json runs/resumed.json
```

If your backend is deterministic (CPU/Metal generally is for this example), the hashes should match exactly.

## Notes

- This repo uses a tiny Flax MLP + Optax Adam and a synthetic dataset created with a seeded NumPy RNG.
- For multi-host scenarios, fold in `jax.process_index()` to derive distinct, reproducible per-host keys.
