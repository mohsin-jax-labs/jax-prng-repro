"""Complete Flow: How All Components Work Together

This demonstrates the entire reproducible training pipeline.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from pathlib import Path
import json
import hashlib

print("=== COMPLETE REPRODUCIBLE TRAINING FLOW ===\n")

# Step 1: Initialize reproducible randomness
print("STEP 1: Initialize PRNG")
SEED = 42
key = jax.random.key(SEED)
print(f"Created key from seed {SEED}: {key}")

# Step 2: Create synthetic data (deterministic)
print("\nSTEP 2: Create Deterministic Data")
def make_data(seed, n_samples=100):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)  # Simple rule
    return jnp.array(X), jnp.array(y)

X_train, y_train = make_data(SEED)
print(f"Created {len(X_train)} samples with {X_train.shape[1]} features")

# Step 3: Define model
print("\nSTEP 3: Define Model")
class TinyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(8)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)
        return x

model = TinyModel()

# Step 4: Initialize model with PRNG
print("\nSTEP 4: Initialize Model")
key, init_key = jax.random.split(key)
params = model.init(init_key, jnp.ones((1, 4)))
print(f"Model initialized with key: {init_key}")

# Step 5: Create training state
print("\nSTEP 5: Create Training State")
tx = optax.adam(1e-3)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params['params'],  # Extract params from the init dict
    tx=tx
)

# Step 6: Define training step
print("\nSTEP 6: Define Training Step")
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['X'])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['y'])
        return loss.mean()
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Step 7: Training loop with checkpointing
print("\nSTEP 7: Training Loop")
def train(state, key, X, y, steps=10, checkpoint_at=None, start_step=0):
    losses = []
    
    # Simple batch creation
    batch_size = 32
    n_batches = len(X) // batch_size
    
    for step in range(steps):
        # Get batch deterministically - account for start_step
        global_step = start_step + step
        batch_idx = global_step % n_batches
        start = batch_idx * batch_size
        end = start + batch_size
        batch = {'X': X[start:end], 'y': y[start:end]}
        
        # Train step
        state, loss = train_step(state, batch)
        losses.append(float(loss))
        
        # Advance RNG (even though not used here, for demonstration)
        key, _ = jax.random.split(key)
        
        # Checkpoint if requested
        if checkpoint_at and step + 1 == checkpoint_at:
            print(f"  Step {step+1}: loss={loss:.4f} [CHECKPOINTING]")
            return state, key, losses, True  # True = checkpointed
        else:
            print(f"  Step {step+1}: loss={loss:.4f}")
    
    return state, key, losses, False

# Scenario 1: Full training
print("\n--- Scenario 1: Full Training (10 steps) ---")
state1 = state  # Fresh state
key1 = key     # Fresh key
final_state1, final_key1, losses1, _ = train(state1, key1, X_train, y_train, steps=10)

# Hash final parameters
def get_param_hash(params):
    flat, _ = jax.tree.flatten(params)
    concat = jnp.concatenate([p.reshape(-1) for p in flat])
    return hashlib.sha256(concat.tobytes()).hexdigest()[:16]

hash1 = get_param_hash(final_state1.params)
print(f"Final hash: {hash1}")

# Scenario 2: Interrupted + Resumed
print("\n--- Scenario 2: Interrupted + Resumed Training ---")
print("Phase 1: Train 5 steps and checkpoint")
state2 = state  # Fresh state
key2 = key     # Fresh key
checkpoint_state, checkpoint_key, losses2a, _ = train(state2, key2, X_train, y_train, steps=5, checkpoint_at=5)

print("\nSimulating checkpoint save/load...")
saved_checkpoint = {
    'state': checkpoint_state,
    'key': checkpoint_key,
    'step': 5
}

print("Phase 2: Resume from checkpoint and train 5 more steps")
resumed_state = saved_checkpoint['state']
resumed_key = saved_checkpoint['key']
resumed_step = saved_checkpoint['step']
final_state2, final_key2, losses2b, _ = train(resumed_state, resumed_key, X_train, y_train, steps=5, start_step=resumed_step)

losses2 = losses2a + losses2b
hash2 = get_param_hash(final_state2.params)
print(f"Final hash: {hash2}")

# Compare results
print("\n=== REPRODUCIBILITY CHECK ===")
print(f"Scenario 1 (full) hash:     {hash1}")
print(f"Scenario 2 (resumed) hash:  {hash2}")
print(f"Hashes match: {hash1 == hash2} ✓" if hash1 == hash2 else f"Hashes match: {hash1 == hash2} ✗")

print(f"\nLoss curves match: {np.allclose(losses1, losses2)}")
if len(losses1) == len(losses2):
    max_diff = np.max(np.abs(np.array(losses1) - np.array(losses2)))
    print(f"Max loss difference: {max_diff:.2e}")

# Demonstrate importance of saving RNG state
print("\n=== Why Saving RNG State Matters ===")
print("If we DON'T save the RNG state:")

# Wrong way - fresh key instead of saved one
wrong_key = jax.random.key(SEED)  # Fresh key, not the advanced one!
for i in range(5):  # Advance it 5 times like before
    wrong_key, _ = jax.random.split(wrong_key)

# The keys will be different!
print(f"Correct resumed key: {resumed_key}")
print(f"Wrong fresh key:     {wrong_key}")
print(f"Keys match: {np.array_equal(resumed_key, wrong_key)}")

print("\n=== KEY INSIGHTS ===")
print("1. Seed → Deterministic initialization")
print("2. Data → Generated from seed (no external randomness)")
print("3. Training → Deterministic operations")
print("4. Checkpointing → Save BOTH model state AND RNG state")
print("5. Result → Perfect reproducibility!")

# Save results like the actual project
print("\n=== Output Format (like the project) ===")
result = {
    "hash": hash1,
    "steps": 10,
    "batch_size": 32,
    "seed": SEED,
    "lr": 0.001,
    "backend": jax.default_backend(),
    "final": {
        "loss": float(losses1[-1]),
        "step": 10
    }
}
print(json.dumps(result, indent=2))