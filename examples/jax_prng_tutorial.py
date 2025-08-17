"""Understanding JAX PRNG (Pseudo-Random Number Generator)

Key Concepts:
1. JAX uses explicit random keys instead of global state
2. Keys must be split to get new randomness
3. This makes randomness reproducible and parallelizable
"""

import jax
import numpy as np

print("=== Problem with NumPy's Global Random State ===")
# NumPy has hidden global state that changes
np.random.seed(42)
print(f"First call:  {np.random.rand()}")
print(f"Second call: {np.random.rand()}")  # Different!
print(f"Third call:  {np.random.rand()}")  # Different again!

print("\n=== JAX's Solution: Explicit Random Keys ===")
# JAX makes randomness explicit - no hidden state!
key = jax.random.key(42)  # Create a random key from seed
print(f"Original key: {key}")

# Using the same key always gives the same result
print(f"\nUsing same key multiple times:")
print(f"First call:  {jax.random.uniform(key)}")
print(f"Second call: {jax.random.uniform(key)}")  # Same!
print(f"Third call:  {jax.random.uniform(key)}")  # Same!

print("\n=== Getting New Random Values: Key Splitting ===")
# To get different random values, we must split the key
key, subkey1 = jax.random.split(key)
print(f"After split - new key: {key}")
print(f"             subkey1: {subkey1}")

key, subkey2 = jax.random.split(key)
print(f"After split - new key: {key}")
print(f"             subkey2: {subkey2}")

print(f"\nRandom values from different subkeys:")
print(f"From subkey1: {jax.random.uniform(subkey1)}")
print(f"From subkey2: {jax.random.uniform(subkey2)}")  # Different!

print("\n=== Splitting Multiple Keys at Once ===")
# You can split into multiple keys at once
key = jax.random.key(42)
keys = jax.random.split(key, num=4)  # Split into 4 keys
print(f"Split into {len(keys)} keys:")
for i, k in enumerate(keys):
    print(f"  Key {i}: random value = {jax.random.uniform(k)}")

print("\n=== Why This Matters for Neural Networks ===")
# In neural networks, we need randomness for:
# 1. Weight initialization
# 2. Dropout
# 3. Data augmentation
# 4. Stochastic optimization

key = jax.random.key(42)
key, init_key, dropout_key = jax.random.split(key, 3)

# Initialize weights
weights = jax.random.normal(init_key, shape=(3, 2))
print(f"Initialized weights:\n{weights}")

# Apply dropout (randomly zero out elements)
dropout_mask = jax.random.bernoulli(dropout_key, p=0.8, shape=(3, 2))
print(f"\nDropout mask (1=keep, 0=drop):\n{dropout_mask}")