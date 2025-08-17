"""Understanding Parameter Hashing

Hashing creates a unique "fingerprint" of model parameters.
Same parameters = Same hash
Different parameters = Different hash (with very high probability)
"""

import jax
import hashlib
import jax.numpy as jnp

def simple_hash(data):
    """Create a hash of any data"""
    # Convert to string representation
    data_str = str(data)
    # Create SHA-256 hash
    return hashlib.sha256(data_str.encode()).hexdigest()

print("=== Basic Hashing Concept ===")
# Same data -> Same hash
data1 = [1, 2, 3]
data2 = [1, 2, 3]
data3 = [1, 2, 4]  # Different!

print(f"data1 = {data1}, hash = {simple_hash(data1)}")
print(f"data2 = {data2}, hash = {simple_hash(data2)}")
print(f"data3 = {data3}, hash = {simple_hash(data3)}")
print(f"\ndata1 == data2? {simple_hash(data1) == simple_hash(data2)}")
print(f"data1 == data3? {simple_hash(data1) == simple_hash(data3)}")

# The actual hashing approach used in the project
def params_hash(params):
    """Hash model parameters (the real implementation)"""
    # Flatten nested dict/list structure to a single array
    flat_params, tree_def = jax.tree.flatten(params)
    
    # Concatenate all arrays
    all_values = jnp.concatenate([p.reshape(-1) for p in flat_params])
    
    # Convert to bytes for hashing
    bytes_data = all_values.tobytes()
    
    # Create hash
    return hashlib.sha256(bytes_data).hexdigest()

print("\n=== Hashing Model Parameters ===")
# Create two identical models
params1 = {
    'layer1': {'w': jnp.ones((3, 2)), 'b': jnp.zeros(2)},
    'layer2': {'w': jnp.ones((2, 1)), 'b': jnp.zeros(1)}
}

params2 = {
    'layer1': {'w': jnp.ones((3, 2)), 'b': jnp.zeros(2)},
    'layer2': {'w': jnp.ones((2, 1)), 'b': jnp.zeros(1)}
}

# Slightly different model
params3 = {
    'layer1': {'w': jnp.ones((3, 2)), 'b': jnp.zeros(2)},
    'layer2': {'w': jnp.ones((2, 1)), 'b': jnp.ones(1)}  # Different bias!
}

hash1 = params_hash(params1)
hash2 = params_hash(params2)
hash3 = params_hash(params3)

print(f"Model 1 hash: {hash1[:16]}...")
print(f"Model 2 hash: {hash2[:16]}...")
print(f"Model 3 hash: {hash3[:16]}...")
print(f"\nModel 1 == Model 2? {hash1 == hash2}")
print(f"Model 1 == Model 3? {hash1 == hash3}")

print("\n=== Why Hashing is Useful ===")
print("1. Quick comparison: Just compare hash strings instead of all parameters")
print("2. Reproducibility check: Same training = Same final hash")
print("3. Version tracking: Track which model version you're using")
print("4. Debugging: Detect when models diverge unexpectedly")

# Demonstrate tree flattening
print("\n=== Understanding Tree Flattening ===")
nested_data = {
    'a': [1, 2, 3],
    'b': {'x': 4, 'y': 5},
    'c': 6
}

flat, tree_def = jax.tree.flatten(nested_data)
print(f"Original nested structure: {nested_data}")
print(f"Flattened: {flat}")
print(f"Can reconstruct: {jax.tree.unflatten(tree_def, flat)}")

# Show how small changes affect hash
print("\n=== Hash Sensitivity ===")
import numpy as np

# Create parameters with tiny differences
key = jax.random.key(42)
base_params = jax.random.normal(key, (100,))

for epsilon in [0, 1e-10, 1e-8, 1e-6]:
    modified = base_params + epsilon
    hash_val = hashlib.sha256(modified.tobytes()).hexdigest()
    print(f"Epsilon = {epsilon:1.0e}, hash = {hash_val[:16]}...")
    
print("\nEven tiny changes completely change the hash!")