"""Understanding the Model Component

The model.py file contains:
1. A simple Multi-Layer Perceptron (MLP)
2. Training state management
3. Loss and metric functions
"""

import jax
import optax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

# Let's recreate the model to understand it
class SimpleMLP(nn.Module):
    """A Multi-Layer Perceptron (fully connected neural network)
    
    Think of it like this:
    Input (32 features) → Hidden Layer 1 (64 neurons) → Hidden Layer 2 (32 neurons) → Output (10 classes)
    """
    hidden_sizes: tuple[int, ...] = (64, 32)
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch_size, input_features)
        
        # Pass through hidden layers
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)  # Linear transformation
            x = nn.relu(x)  # Activation function (makes it non-linear)
        
        # Final output layer (no activation - raw logits)
        x = nn.Dense(self.num_classes)(x)
        return x

# Create and inspect the model
print("=== Creating the Model ===")
model = SimpleMLP()

# Initialize with dummy input to see the structure
key = jax.random.key(42)
dummy_input = jnp.ones((1, 32))  # 1 sample, 32 features
params = model.init(key, dummy_input)

print(f"Model initialized!")
print(f"Parameter structure:")
for layer_name, layer_params in params['params'].items():
    print(f"  {layer_name}:")
    for param_name, param_array in layer_params.items():
        print(f"    {param_name}: shape {param_array.shape}")

# Understanding TrainState
print("\n=== Understanding TrainState ===")
print("TrainState bundles together:")
print("1. Model parameters (weights & biases)")
print("2. Model apply function (forward pass)")
print("3. Optimizer state (momentum, etc)")
print("4. Step counter")

# Create optimizer (Adam - adaptive learning rate)
tx = optax.adam(learning_rate=1e-3)

# Create train state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params['params'],
    tx=tx
)

print(f"\nTrainState created with step={state.step}")

# Understanding the loss function
print("\n=== Understanding Cross-Entropy Loss ===")
# Simulate some predictions and labels
batch_size = 4
logits = jnp.array([
    [2.0, -1.0, 0.5],  # Predicts class 0 (highest value)
    [-1.0, 3.0, 0.0],  # Predicts class 1
    [0.0, 0.0, 2.0],   # Predicts class 2
    [1.0, 1.0, 1.0],   # Uncertain (all equal)
])
labels = jnp.array([0, 1, 2, 0])  # True classes

# Convert to probabilities
probs = jax.nn.softmax(logits)
print("Logits (raw outputs):")
print(logits)
print("\nProbabilities (after softmax):")
print(probs)
print(f"\nTrue labels: {labels}")

# Calculate cross-entropy loss
def cross_entropy_loss(logits, labels):
    # One-hot encode labels
    num_classes = logits.shape[-1]
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    
    # Cross-entropy = -sum(true_label * log(predicted_prob))
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(one_hot_labels * log_probs, axis=-1)
    return jnp.mean(loss)

loss = cross_entropy_loss(logits, labels)
print(f"\nCross-entropy loss: {loss:.4f}")
print("(Lower is better, 0 = perfect predictions)")

# Calculate accuracy
predictions = jnp.argmax(logits, axis=-1)
accuracy = jnp.mean(predictions == labels)
print(f"\nPredictions: {predictions}")
print(f"Accuracy: {accuracy:.2%}")