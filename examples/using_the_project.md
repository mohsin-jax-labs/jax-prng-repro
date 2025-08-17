# How to Use This JAX Reproducibility Project

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Basic reproducibility test
python -m jpr.run_repro --steps 50 --batch-size 64 --seed 0 --out runs/test1.json
python -m jpr.run_repro --steps 50 --batch-size 64 --seed 0 --out runs/test2.json
python -m jpr.compare runs/test1.json runs/test2.json

# Should output: ✅ MATCH: hashes (and metrics if requested) are equal.
```

## Understanding the Commands

### 1. `python -m jpr.run_repro` - Main Training Script

**Purpose**: Runs reproducible training with checkpointing support

**Key Arguments**:
- `--steps N`: Number of training steps
- `--batch-size N`: Batch size for training
- `--seed N`: Random seed for reproducibility
- `--out FILE.json`: Output file with results and hash
- `--checkpoint-dir DIR`: Directory for saving checkpoints
- `--save-at N`: Save checkpoint at step N
- `--restore-from N`: Resume from checkpoint at step N

**Examples**:
```bash
# Basic training
python -m jpr.run_repro --steps 100 --seed 42 --out results.json

# Training with checkpoint at step 50
python -m jpr.run_repro --steps 50 --seed 42 \
    --checkpoint-dir checkpoints/exp1 --save-at 50 \
    --out phase1.json

# Resume from checkpoint
python -m jpr.run_repro --steps 100 --seed 42 \
    --checkpoint-dir checkpoints/exp1 --restore-from 50 \
    --out resumed.json
```

### 2. `python -m jpr.compare` - Compare Results

**Purpose**: Compares parameter hashes and metrics between runs

**Usage**:
```bash
python -m jpr.compare file1.json file2.json [file3.json ...]
```

**Output**:
- ✅ MATCH: Runs are identical
- ❌ MISMATCH: Runs differ (shows differences)

## Complete Reproducibility Workflow

### Step 1: Verify Basic Reproducibility
```bash
# Same configuration should give same results
python -m jpr.run_repro --steps 30 --seed 123 --out run1.json
python -m jpr.run_repro --steps 30 --seed 123 --out run2.json
python -m jpr.compare run1.json run2.json
```

### Step 2: Test Checkpoint/Resume Equivalence
```bash
# Full uninterrupted run
python -m jpr.run_repro --steps 60 --seed 456 --out full.json

# Two-phase run with checkpoint
python -m jpr.run_repro --steps 40 --seed 456 \
    --checkpoint-dir ckpt --save-at 40 --out phase1.json

python -m jpr.run_repro --steps 60 --seed 456 \
    --checkpoint-dir ckpt --restore-from 40 --out resumed.json

# Should be identical
python -m jpr.compare full.json resumed.json
```

### Step 3: Run Test Suite
```bash
# Automated tests
python -m pytest tests/ -v
```

## Understanding the Output JSON

Each run produces a JSON file with:

```json
{
  "hash": "9f8ec42c26e22096...",  // SHA-256 hash of final parameters
  "steps": 60,                    // Number of training steps
  "batch_size": 64,               // Batch size used
  "seed": 123,                    // Random seed
  "lr": 0.001,                    // Learning rate
  "backend": "cpu",               // JAX backend (cpu/gpu/tpu)
  "final": {
    "loss": 0.094,                // Final loss value
    "acc": 1.0                    // Final accuracy
  }
}
```

**Key field**: `hash` - This is the "fingerprint" of your trained model. Identical hashes = identical models.

## Project Architecture

```
src/jpr/
├── run_repro.py    # Main training script
├── model.py        # Simple MLP neural network
├── data.py         # Synthetic dataset generation
├── rng.py          # PRNG management utilities
├── checkpoint.py   # Orbax checkpoint wrapper
└── hashing.py      # Parameter hashing for verification
```

## What Makes This Reproducible?

1. **Deterministic Data**: Generated from fixed seeds, no external randomness
2. **Explicit PRNG**: All randomness controlled via JAX keys
3. **Checkpoint RNG**: Random state saved/restored with model
4. **Sequential Batching**: No random shuffling during training
5. **Parameter Hashing**: Cryptographic verification of model equivalence

## Common Use Cases

### Debugging
```bash
# Run same config, should get identical results
python -m jpr.run_repro --steps 10 --seed 42 --out debug1.json
# ... make code change ...
python -m jpr.run_repro --steps 10 --seed 42 --out debug2.json
python -m jpr.compare debug1.json debug2.json
```

### Experiment Tracking
```bash
# Different hyperparameters, same seed
python -m jpr.run_repro --steps 100 --seed 42 --lr 1e-3 --out exp_lr1e3.json
python -m jpr.run_repro --steps 100 --seed 42 --lr 1e-4 --out exp_lr1e4.json
```

### CI/CD Testing
```bash
# In your CI pipeline
python -m jpr.run_repro --steps 20 --seed 42 --out ci_test.json
# Compare against known good hash
expected_hash="known_good_hash_here"
actual_hash=$(python -c "import json; print(json.load(open('ci_test.json'))['hash'])")
if [ "$expected_hash" != "$actual_hash" ]; then
    echo "Reproducibility test failed!"
    exit 1
fi
```

## Tips for Real Projects

1. **Pin Dependencies**: Use exact versions in requirements.txt
2. **Document Hardware**: Different GPUs may give different results
3. **Test Regularly**: Include reproducibility tests in CI
4. **Save Everything**: Include RNG state in all checkpoints
5. **Version Control**: Track model hashes in git

## Extending This Project

To adapt this pattern to your project:

1. Replace `model.py` with your architecture
2. Replace `data.py` with your data pipeline (keep deterministic!)
3. Keep `rng.py`, `checkpoint.py`, `hashing.py` patterns
4. Add your specific metrics to the output JSON
5. Write tests using the same patterns in `tests/`

This project demonstrates the "gold standard" for reproducible ML in JAX!