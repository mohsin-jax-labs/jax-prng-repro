"""Understanding Checkpointing

Checkpointing allows us to:
1. Save training progress
2. Resume from exact same state
3. Recover from crashes
4. Share trained models
"""

import jax
import optax
import shutil
import jax.numpy as jnp
from pathlib import Path
import orbax.checkpoint as ocp
from flax.training import train_state

# Create a simple state to demonstrate
def create_dummy_state():
    """Create a simple train state for demonstration"""
    # Dummy model parameters
    params = {
        'layer1': {'w': jnp.ones((3, 2)), 'b': jnp.zeros(2)},
        'layer2': {'w': jnp.ones((2, 1)), 'b': jnp.zeros(1)}
    }
    
    # Create optimizer
    tx = optax.adam(1e-3)
    
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=lambda x: x,  # Dummy function
        params=params,
        tx=tx
    )
    
    return state

print("=== Understanding What Gets Saved ===")

# Create state and RNG key
state = create_dummy_state()
rng_key = jax.random.key(42)

print("What we want to save:")
print(f"1. Model parameters (weights & biases)")
print(f"2. Optimizer state (momentum, history, etc)")
print(f"3. Training step count: {state.step}")
print(f"4. RNG key: {rng_key}")
print("\nWhy save RNG key? So randomness continues exactly where it left off!")

# Save checkpoint
print("\n=== Saving Checkpoint ===")
ckpt_dir = Path("examples/demo_checkpoint").resolve()  # Make absolute
if ckpt_dir.exists():
    shutil.rmtree(ckpt_dir)
ckpt_dir.mkdir(parents=True)

# This is what the project does
payload = {"state": state, "rng": rng_key}
checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
checkpointer.save(ckpt_dir / "step_100", args=ocp.args.PyTreeSave(payload))

print(f"Saved checkpoint to {ckpt_dir}/step_100/")
print("Files created:")
for file in sorted((ckpt_dir / "step_100").glob("*")):
    print(f"  {file.name}")

# Demonstrate the importance of saving RNG
print("\n=== Why Saving RNG Matters ===")

# Scenario 1: Not saving RNG (BAD)
print("Scenario 1: Not saving RNG (different randomness after resume)")
key1 = jax.random.key(42)
key1, subkey = jax.random.split(key1)
print(f"  Before checkpoint: random value = {jax.random.uniform(subkey)}")
# ... imagine we checkpoint here WITHOUT saving key1 ...
# ... then on resume, we start with a fresh key ...
key2 = jax.random.key(42)  # Fresh key, not the advanced one!
key2, subkey = jax.random.split(key2)
print(f"  After resume (wrong): random value = {jax.random.uniform(subkey)}")

# Scenario 2: Saving RNG (GOOD)
print("\nScenario 2: Saving RNG (same randomness after resume)")
key1 = jax.random.key(42)
key1, subkey = jax.random.split(key1)
print(f"  Before checkpoint: random value = {jax.random.uniform(subkey)}")
saved_key = key1  # We save this!
# ... imagine we checkpoint here WITH key1 ...
# ... then on resume, we load the saved key ...
key2 = saved_key  # Restored from checkpoint
key2, subkey = jax.random.split(key2)
print(f"  After resume (correct): random value = {jax.random.uniform(subkey)}")

# The complete checkpoint/resume pattern
print("\n=== The Complete Pattern ===")
print("""
1. During training:
   state, loss = train_step(state, batch)
   key, subkey = jax.random.split(key)  # Advance RNG
   
2. When checkpointing:
   save_checkpoint({"state": state, "rng": key})
   
3. When resuming:
   checkpoint = load_checkpoint()
   state = checkpoint["state"]
   key = checkpoint["rng"]  # Continue from exact same randomness!
   
4. Result: Resumed training = Uninterrupted training
""")

# Clean up
shutil.rmtree(ckpt_dir)