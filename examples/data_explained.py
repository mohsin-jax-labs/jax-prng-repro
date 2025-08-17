"""Understanding the Data Component

The data.py creates synthetic (fake) data that's:
1. Deterministic (same seed = same data)
2. Simple classification task
3. No external dependencies
"""

import numpy as np
import jax.numpy as jnp

def make_synthetic_dataset(num_samples=100, input_dim=2, num_classes=3, seed=42):
    """Create a synthetic classification dataset
    
    The idea: 
    1. Create 'centroids' for each class (cluster centers)
    2. Generate points around each centroid with some noise
    3. This creates naturally separable classes
    """
    rng = np.random.default_rng(seed)
    
    # Step 1: Create centroids for each class
    centroids = rng.normal(0.0, 3.0, size=(num_classes, input_dim))
    print(f"Class centroids (cluster centers):")
    for i, centroid in enumerate(centroids):
        print(f"  Class {i}: {centroid}")
    
    # Step 2: Assign labels randomly
    labels = rng.integers(0, num_classes, size=(num_samples,))
    
    # Step 3: Generate points around centroids
    X = centroids[labels]  # Get the centroid for each point's class
    X += rng.normal(0.0, 1.0, size=(num_samples, input_dim))  # Add noise
    
    return jnp.asarray(X, jnp.float32), jnp.asarray(labels, jnp.int32)

# Create and visualize the dataset
print("=== Creating Synthetic Dataset ===")
X, y = make_synthetic_dataset(num_samples=300, input_dim=2, num_classes=3, seed=42)

print(f"\nDataset shape: X={X.shape}, y={y.shape}")
print(f"Features per sample: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")

# The data loader
def simple_batch_loader(X, y, batch_size=32):
    """Create batches of data sequentially (no shuffling!)
    
    Why no shuffling? For perfect reproducibility!
    The order of data affects training, so we keep it fixed.
    """
    n = X.shape[0]
    pos = 0
    
    while True:
        # Get next batch
        end = min(pos + batch_size, n)
        batch_X = X[pos:end]
        batch_y = y[pos:end]
        
        # Wrap around to beginning
        if end == n:
            pos = 0
        else:
            pos = end
            
        yield batch_X, batch_y

# Demonstrate the loader
print("\n=== Understanding the Data Loader ===")
loader = simple_batch_loader(X, y, batch_size=32)

print("First 3 batches:")
for i in range(3):
    batch_X, batch_y = next(loader)
    print(f"  Batch {i+1}: X shape={batch_X.shape}, y shape={batch_y.shape}")
    print(f"           Labels: {batch_y[:10]}...")  # First 10 labels

# Show determinism
print("\n=== Demonstrating Determinism ===")
X1, y1 = make_synthetic_dataset(num_samples=100, seed=123)
X2, y2 = make_synthetic_dataset(num_samples=100, seed=123)
X3, y3 = make_synthetic_dataset(num_samples=100, seed=456)

print(f"Same seed (123) twice:")
print(f"  Arrays equal? X: {jnp.allclose(X1, X2)}, y: {jnp.array_equal(y1, y2)}")
print(f"Different seed (456):")
print(f"  Arrays equal? X: {jnp.allclose(X1, X3)}, y: {jnp.array_equal(y1, y3)}")

# Note: Would visualize here if matplotlib was available
print("\nNote: This synthetic data creates separable clusters for each class")