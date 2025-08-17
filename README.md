# JAX PRNG Reproducibility Playbook

A comprehensive **reproducibility framework** for JAX demonstrating industry best practices:

🔑 **Core Features**:
- `jax.random.key(seed)` for explicit PRNG management (2025-style API)
- Process index folding for **multi-host safety** in distributed training
- Checkpoint/resume with **exact reproducibility** (save RNG state + model state)
- **Cryptographic hashing** of model parameters for verification
- Deterministic synthetic datasets with no external dependencies
- Complete tutorial with examples from beginner to expert level

🎯 **Perfect for**: Learning JAX reproducibility, debugging training runs, research reproducibility, production ML systems

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mohsin-jax-labs/jax-prng-repro.git
cd jax-prng-repro

# Setup virtual environment  
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Verify setup works
python verify_setup.py
```

### Basic Usage

#### 1) Test Reproducibility - Same seed = Same results
```bash
python -m jpr.run_repro --steps 50 --batch-size 64 --seed 0 --out runs/test1.json
python -m jpr.run_repro --steps 50 --batch-size 64 --seed 0 --out runs/test2.json
python -m jpr.compare runs/test1.json runs/test2.json
# Output: ✅ MATCH: hashes (and metrics if requested) are equal.
```

#### 2) Test Checkpoint/Resume - Interrupted = Uninterrupted  
```bash
# Full uninterrupted run
python -m jpr.run_repro --steps 60 --batch-size 64 --seed 123 --out runs/full.json

# Two-phase run with checkpoint  
python -m jpr.run_repro --steps 40 --batch-size 64 --seed 123 \
  --checkpoint-dir checkpoints/demo --save-at 40 --out runs/phase1.json

python -m jpr.run_repro --steps 60 --batch-size 64 --seed 123 \
  --checkpoint-dir checkpoints/demo --restore-from 40 --out runs/resumed.json

# Should be identical
python -m jpr.compare runs/full.json runs/resumed.json
# Output: ✅ MATCH: hashes (and metrics if requested) are equal.
```

## Learning Path

📚 **New to JAX reproducibility?** Follow our comprehensive tutorial:

1. **Start here**: [`TUTORIAL.md`](TUTORIAL.md) - Complete learning path
2. **Hands-on examples**: [`examples/`](examples/) directory  
3. **Usage guide**: [`examples/using_the_project.md`](examples/using_the_project.md)

## Architecture

This project demonstrates the **gold standard** for ML reproducibility:

```
🏗️ Core Framework:
├── 🎲 PRNG Management (rng.py) - Explicit key handling, multi-host safety
├── 🏠 Model (model.py) - Simple MLP with Flax + Optax  
├── 📊 Data (data.py) - Deterministic synthetic dataset
├── 💾 Checkpointing (checkpoint.py) - Save model + RNG state
├── 🔍 Verification (hashing.py) - Parameter fingerprinting
└── 🔧 CLI Tools (run_repro.py, compare.py) - Training & comparison

📚 Complete Tutorial:
├── examples/reproducibility_basics.py - Why reproducibility matters
├── examples/jax_prng_tutorial.py - JAX vs NumPy randomness  
├── examples/complete_flow.py - End-to-end demonstration
├── examples/pitfalls_and_best_practices.py - Common mistakes
└── examples/using_the_project.md - Usage guide
```

## Key Benefits

- ✅ **Perfect Reproducibility**: Same config → Identical results  
- ✅ **Checkpoint Safety**: Resume = Continue uninterrupted
- ✅ **Multi-host Ready**: Distributed training support
- ✅ **Verification Built-in**: Cryptographic proof of equivalence
- ✅ **Production Ready**: Patterns used in real ML systems
- ✅ **Educational**: Learn from basics to expert level

## Technical Notes

- **Model**: Tiny Flax MLP (64→32→10) with Optax Adam optimizer
- **Data**: Synthetic classification dataset (deterministic, seed-controlled)  
- **Backends**: Tested on CPU/GPU (some numerical differences possible on edge cases)
- **Multi-host**: Uses `jax.process_index()` folding for distributed safety
- **Dependencies**: JAX ≥0.4.26, Flax ≥0.7, Optax ≥0.2.2, Orbax ≥0.6.1
