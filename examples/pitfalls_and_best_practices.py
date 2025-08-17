"""Common Pitfalls and Best Practices for JAX Reproducibility

This covers what can go wrong and how to avoid it.
"""

import jax
import jax.numpy as jnp
import numpy as np

print("=== COMMON PITFALLS IN JAX REPRODUCIBILITY ===\n")

# Pitfall 1: Using NumPy random instead of JAX random
print("PITFALL 1: Mixing NumPy and JAX randomness")
print("❌ BAD: Using NumPy random in JAX code")

# This creates non-reproducible behavior
def bad_dropout(x, keep_prob=0.8):
    # This uses NumPy's global random state!
    mask = np.random.rand(*x.shape) < keep_prob
    return x * mask

print("NumPy random in different calls:")
x = jnp.ones((3, 3))
print(f"Call 1: {bad_dropout(x)}")
print(f"Call 2: {bad_dropout(x)}")  # Different!

print("\n✅ GOOD: Using JAX random with explicit keys")
def good_dropout(x, key, keep_prob=0.8):
    # This uses JAX's explicit randomness
    mask = jax.random.bernoulli(key, keep_prob, x.shape)
    return x * mask

key = jax.random.key(42)
print("JAX random with same key:")
print(f"Call 1: {good_dropout(x, key)}")
print(f"Call 2: {good_dropout(x, key)}")  # Same!

# Pitfall 2: Not advancing the RNG key
print("\n" + "="*50)
print("PITFALL 2: Not advancing RNG keys")
print("❌ BAD: Reusing the same key")

key = jax.random.key(42)
print("Reusing same key (always same randomness):")
for i in range(3):
    value = jax.random.uniform(key)  # Same key every time!
    print(f"  Step {i+1}: {value}")

print("\n✅ GOOD: Properly splitting/advancing keys")
key = jax.random.key(42)
print("Properly advancing keys:")
for i in range(3):
    key, subkey = jax.random.split(key)  # Advance the key!
    value = jax.random.uniform(subkey)
    print(f"  Step {i+1}: {value}")

# Pitfall 3: Not saving RNG state in checkpoints
print("\n" + "="*50)
print("PITFALL 3: Not saving RNG state in checkpoints")
print("❌ BAD: Only saving model parameters")

# Simulate training with checkpointing
def simulate_training(initial_key, steps):
    key = initial_key
    for step in range(steps):
        key, subkey = jax.random.split(key)
        random_noise = jax.random.uniform(subkey)
        print(f"  Step {step+1}: noise = {random_noise:.4f}")
    return key

print("Training 3 steps, then 'checkpoint' (without RNG):")
key = jax.random.key(42)
final_key = simulate_training(key, 3)

print("\nResuming with fresh key (WRONG):")
fresh_key = jax.random.key(42)  # Started fresh, not continuing!
simulate_training(fresh_key, 2)

print("\nWhat should happen - continuing with saved key:")
simulate_training(final_key, 2)  # Continue from where we left off

# Pitfall 4: Non-deterministic data loading
print("\n" + "="*50)
print("PITFALL 4: Non-deterministic data loading")
print("❌ BAD: Random shuffling without fixed seed")

data = jnp.arange(10)
print(f"Original data: {data}")

# Bad: Different order each time
print("Random shuffling (different each time):")
for i in range(2):
    shuffled = np.random.permutation(data)  # Uses global random state
    print(f"  Shuffle {i+1}: {shuffled}")

print("\n✅ GOOD: Deterministic shuffling with seed")
print("Seeded shuffling (same each time):")
for i in range(2):
    rng = np.random.default_rng(42)  # Fixed seed
    shuffled = rng.permutation(data)
    print(f"  Shuffle {i+1}: {shuffled}")

# Pitfall 5: Hardware/platform differences
print("\n" + "="*50)
print("PITFALL 5: Hardware/platform differences")
print("⚠️  WARNING: Some operations may differ across platforms")

print("Current backend:", jax.default_backend())
print("Available devices:", jax.devices())

# Some operations might have slight numerical differences
x = jnp.array([1e-7, 1e-8, 1e-9])
result = jnp.sum(x)
print(f"Sum of small numbers: {result}")
print("Note: GPU vs CPU might give slightly different results for edge cases")

# Best Practices Summary
print("\n" + "="*50)
print("BEST PRACTICES SUMMARY")
print("="*50)

best_practices = """
1. PRNG Management:
   ✓ Always use jax.random with explicit keys
   ✓ Split keys before using: key, subkey = jax.random.split(key)
   ✓ Never reuse the same key for different random operations
   ✓ Save and restore RNG keys in checkpoints

2. Data Handling:
   ✓ Use fixed seeds for data generation
   ✓ Avoid random shuffling or use seeded shuffling
   ✓ Process data deterministically (same order every time)

3. Model Initialization:
   ✓ Initialize with seeded PRNG keys
   ✓ Use same initialization across runs
   ✓ Document your initialization strategy

4. Checkpointing:
   ✓ Save both model state AND RNG state
   ✓ Test resume equivalence (resumed = uninterrupted)
   ✓ Use absolute paths for checkpoint directories

5. Environment:
   ✓ Pin software versions (JAX, Flax, etc.)
   ✓ Document hardware used
   ✓ Test on target deployment platform

6. Testing:
   ✓ Write tests that verify reproducibility
   ✓ Compare parameter hashes across runs
   ✓ Test checkpoint/resume functionality
"""

print(best_practices)

# Testing reproducibility pattern
print("\n" + "="*50)
print("TESTING REPRODUCIBILITY PATTERN")
print("="*50)

def test_reproducibility():
    """Pattern for testing reproducibility"""
    
    def run_experiment(seed):
        # All randomness comes from this seed
        key = jax.random.key(seed)
        
        # Generate data
        key, data_key = jax.random.split(key)
        data = jax.random.normal(data_key, (100, 10))
        
        # Initialize model  
        key, init_key = jax.random.split(key)
        params = jax.random.normal(init_key, (10, 1))
        
        # Simple computation
        result = jnp.sum(data @ params)
        return float(result)
    
    # Test: Same seed = Same result
    result1 = run_experiment(42)
    result2 = run_experiment(42)
    result3 = run_experiment(123)  # Different seed
    
    print(f"Seed 42, run 1: {result1}")
    print(f"Seed 42, run 2: {result2}")
    print(f"Seed 123:       {result3}")
    
    print(f"\nSame seed reproducible: {abs(result1 - result2) < 1e-6}")
    print(f"Different seed different: {abs(result1 - result3) > 1e-6}")

test_reproducibility()

print("\n🎉 You now understand JAX reproducibility!")
print("Key takeaway: Control ALL sources of randomness explicitly.")