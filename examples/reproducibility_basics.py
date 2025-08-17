"""Example showing non-reproducible vs reproducible code"""

import jax
import numpy as np

print("=== Non-Reproducible Example (Bad) ===")
# This uses system randomness - different every time!
for i in range(3):
    random_value = np.random.rand()
    print(f"Run {i+1}: {random_value}")

print("\n=== Reproducible Example (Good) ===")
# This uses a fixed seed - same every time!
for i in range(3):
    np.random.seed(42)  # Reset seed each time
    random_value = np.random.rand()
    print(f"Run {i+1}: {random_value}")

print("\n=== JAX Reproducible Example (Best) ===")
# JAX uses explicit random keys - even better!
key = jax.random.key(42)
for i in range(3):
    # We use the same key, so we get the same value
    random_value = jax.random.uniform(key)
    print(f"Run {i+1}: {random_value}")