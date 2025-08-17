# JAX Reproducibility Tutorial

This directory contains a comprehensive tutorial covering JAX reproducibility from beginner to expert level.

## Learning Path

Follow these examples in order:

### 1. Fundamentals
- [`examples/reproducibility_basics.py`](examples/reproducibility_basics.py) - What is reproducibility and why it matters
- [`examples/jax_prng_tutorial.py`](examples/jax_prng_tutorial.py) - JAX PRNG concepts vs NumPy
- [`examples/prng_pattern.py`](examples/prng_pattern.py) - The specific patterns used in this project

### 2. Component Deep Dives
- [`examples/model_explained.py`](examples/model_explained.py) - Understanding the neural network model
- [`examples/data_explained.py`](examples/data_explained.py) - Deterministic synthetic data generation
- [`examples/checkpoint_explained.py`](examples/checkpoint_explained.py) - Saving/restoring training state
- [`examples/hashing_explained.py`](examples/hashing_explained.py) - Parameter hashing for verification

### 3. Complete Workflow
- [`examples/complete_flow.py`](examples/complete_flow.py) - End-to-end demonstration
- [`examples/pitfalls_and_best_practices.py`](examples/pitfalls_and_best_practices.py) - Common mistakes and how to avoid them

### 4. Usage Guide
- [`examples/using_the_project.md`](examples/using_the_project.md) - How to use the actual project commands

## Running the Examples

Each Python file can be run independently:

```bash
python examples/reproducibility_basics.py
python examples/jax_prng_tutorial.py
# ... etc
```

## Key Concepts Covered

- **PRNG Management**: JAX's explicit random keys vs global state
- **Deterministic Training**: Controlling all sources of randomness
- **Checkpoint Strategy**: Saving both model and RNG state
- **Verification**: Using cryptographic hashes to prove reproducibility
- **Multi-host Safety**: Process index folding for distributed training
- **Best Practices**: Industry-standard patterns for ML reproducibility

## Prerequisites

- Basic Python knowledge
- Understanding of neural networks (helpful but not required)
- JAX installation (see main README.md)

## After This Tutorial

You'll understand:
- How to make any JAX training completely reproducible
- Common pitfalls that break reproducibility
- How to test and verify reproducible behavior
- Patterns for checkpointing and resuming training
- Best practices used in production ML systems