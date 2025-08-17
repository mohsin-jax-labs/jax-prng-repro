# Contributing to JAX PRNG Reproducibility Playbook

Thank you for your interest in contributing! This project demonstrates best practices for reproducible machine learning with JAX.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohsin-jax-labs/jax-prng-repro.git
   cd jax-prng-repro
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

## Testing

Always test that reproducibility is maintained:

```bash
# Run the test suite
python -m pytest tests/ -v

# Test basic reproducibility
python -m jpr.run_repro --steps 20 --seed 42 --out test1.json
python -m jpr.run_repro --steps 20 --seed 42 --out test2.json
python -m jpr.compare test1.json test2.json

# Test checkpoint/resume equivalence
python -m jpr.run_repro --steps 30 --seed 123 --out full.json
python -m jpr.run_repro --steps 20 --seed 123 --checkpoint-dir ckpt --save-at 20 --out phase1.json
python -m jpr.run_repro --steps 30 --seed 123 --checkpoint-dir ckpt --restore-from 20 --out resumed.json
python -m jpr.compare full.json resumed.json
```

## Code Quality

We maintain high code quality standards:

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Type checking
mypy src/
```

## Contribution Guidelines

### What We Accept

- **Bug fixes**: Reproducibility is critical - any bugs should be fixed
- **Documentation improvements**: Better explanations, examples, or tutorials
- **Performance optimizations**: That maintain exact reproducibility
- **New examples**: Demonstrating additional reproducibility patterns
- **Test improvements**: Better coverage of edge cases

### What We Don't Accept

- **Changes that break reproducibility**: This is the core principle
- **Non-deterministic features**: Random data augmentation, etc.
- **Complex dependencies**: Keep the project lightweight and focused
- **Platform-specific code**: Should work on CPU/GPU/TPU across platforms

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-improvement
   ```

2. **Make your changes** following existing patterns

3. **Test thoroughly**:
   ```bash
   # Ensure tests pass
   python -m pytest tests/
   
   # Verify reproducibility still works
   ./test_reproducibility.sh  # If we add this script
   ```

4. **Update documentation** if needed

5. **Submit a pull request** with:
   - Clear description of changes
   - Test results showing reproducibility is maintained
   - Any new examples or documentation

### Coding Standards

- **Follow existing patterns**: Look at how PRNG keys are managed
- **Explicit is better than implicit**: No hidden random state
- **Document random operations**: Explain why randomness is needed
- **Test determinism**: Every random operation should be testable
- **Use type hints**: Help others understand the interfaces

### Example Contribution

If adding a new model architecture:

```python
# Good: Explicit PRNG management
def create_transformer(key: jax.Array, config: TransformerConfig) -> TrainState:
    key, init_key = jax.random.split(key)
    # ... initialization ...
    return state

# Bad: Hidden randomness
def create_transformer(config: TransformerConfig) -> TrainState:
    # Uses some hidden random state
    # ... initialization ...
    return state
```

## Questions?

- **Check existing examples**: Most patterns are already demonstrated
- **Read the tutorial**: `TUTORIAL.md` covers common scenarios
- **Open an issue**: For bugs or feature requests
- **Start a discussion**: For questions about best practices

## Philosophy

This project demonstrates the "gold standard" for ML reproducibility. Every contribution should maintain this standard and help others learn these critical skills.

Remember: **Reproducibility is not optional in production ML systems!**