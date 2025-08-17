"""The PRNG Pattern Used in This Project

This shows the exact pattern from src/jpr/rng.py
"""

import jax
from dataclasses import dataclass

# This is how the project manages seeds
@dataclass
class SeedConfig:
    seed: int
    process_index: int = 0  # For multi-device training

def make_key(cfg: SeedConfig):
    """Create a base key and fold in process index for multi-host safety."""
    key = jax.random.key(cfg.seed)
    
    # fold_in creates a new key by mixing in an integer
    # This ensures different devices get different random streams
    if cfg.process_index != 0:
        key = jax.random.fold_in(key, cfg.process_index)
    return key

def next_key(key):
    """Split and return (new_key, subkey)."""
    new_key, sub = jax.random.split(key)
    return new_key, sub

# Example usage
print("=== Single Device Training ===")
config = SeedConfig(seed=42, process_index=0)
key = make_key(config)
print(f"Initial key: {key}")

# Get keys for different purposes
key, init_key = next_key(key)
key, data_key = next_key(key)
key, dropout_key = next_key(key)

print(f"After getting keys:")
print(f"  Current key: {key}")
print(f"  Init key gave: {jax.random.normal(init_key, shape=(2,))}")
print(f"  Data key gave: {jax.random.uniform(data_key)}")
print(f"  Dropout key gave: {jax.random.bernoulli(dropout_key, p=0.5)}")

print("\n=== Multi-Device Training ===")
# Different devices need different random streams
for device_id in range(3):
    config = SeedConfig(seed=42, process_index=device_id)
    key = make_key(config)
    value = jax.random.uniform(key)
    print(f"Device {device_id}: key={key}, random value={value}")

print("\n=== Why This Pattern? ===")
print("1. make_key: Consistent initialization from seed")
print("2. fold_in: Different randomness per device/process")
print("3. next_key: Advance the random state predictably")
print("4. Saving key in checkpoint: Can resume with exact same randomness!")