import torch
from chemprop import nn
# Define dummy input: e.g., batch size 8, input_dim = 100.
dummy_input = torch.randn(8, 100)

# Instantiate your multi-head predictor.
# Here, we use n_tasks=5 (one per property) and n_targets=1.
predictor = nn.MultiHeadRegressionFFN(
    input_dim=100,
    n_tasks=5,       # 5 tasks (properties)
    n_targets=1,     # each task outputs one value
    trunk_hidden_dim=64,
    trunk_layers=2,
    head_layers=2,
    head_hidden_dim=32,
    head_dropout=0.1,
    head_activation="relu",
    dropout=0.0,
    activation="relu",
)

# Forward pass: the output should have shape [8, 5]
output = predictor(dummy_input)
print("Output shape:", output.shape)  # Expected: torch.Size([8, 5])

# Create a dummy target and compute loss.
dummy_target = torch.randn(8, 5)
loss = torch.nn.functional.mse_loss(output, dummy_target)
loss.backward()

# Check that gradients have been computed for some parameters.
print("Gradient norm for shared trunk's first layer weight:", predictor.shared_trunk[0][0].weight.grad.norm())

import torch
import torch.nn as nn_torch
import torch.optim as optim

# Generate synthetic data:
# Let's say the true function is a simple linear mapping plus noise.
batch_size = 64
input_dim = 100
n_samples = 1024
X = torch.randn(n_samples, input_dim)
true_W = torch.randn(input_dim, 5)
true_b = torch.randn(5)
Y = X @ true_W + true_b + 0.1 * torch.randn(n_samples, 5)

# Create DataLoader:
dataset = torch.utils.data.TensorDataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate your multi-head predictor:
multihead_model = nn.MultiHeadRegressionFFN(
    input_dim=input_dim,
    n_tasks=5,
    n_targets=1,
    trunk_hidden_dim=64,
    trunk_layers=2,
    head_layers=2,
    head_hidden_dim=32,
    head_dropout=0.1,
    head_activation="relu",
    dropout=0.0,
    activation="relu",
)

# For comparison, define a baseline model that outputs 5 values directly.
baseline_model = nn_torch.Sequential(
    nn_torch.Linear(input_dim, 64),
    nn_torch.ReLU(),
    nn_torch.Linear(64, 5)
)

# Define optimizers:
opt_multi = optim.Adam(multihead_model.parameters(), lr=1e-3)
opt_base = optim.Adam(baseline_model.parameters(), lr=1e-3)

# Simple training loop for a few epochs:
def train_model(model, optimizer, loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            X_batch, Y_batch = batch
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = nn_torch.MSELoss()(preds, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

print("Training Multi-Head Model:")
train_model(multihead_model, opt_multi, loader)

print("Training Baseline Model:")
train_model(baseline_model, opt_base, loader)

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def test_multihead_predictor():
    # Define dummy input: batch size 8, input_dim = 100.
    dummy_input = torch.randn(8, 100)
    
    # Instantiate the multi-head predictor:
    # For 5 properties, we use n_tasks=5 and n_targets=1.
    predictor =nn. MultiHeadRegressionFFN(
        input_dim=100,
        n_tasks=5,       # 5 tasks (properties)
        n_targets=1,     # each task outputs one value
        trunk_hidden_dim=64,
        trunk_layers=2,
        head_layers=2,
        head_hidden_dim=32,
        head_dropout=[0.0, 0.1, 0.2, 0.3, 0.4],
        head_activation="relu",
        dropout=0.0,
        activation="relu",
    )
    
    # Forward pass: output should have shape [8, 5]
    output = predictor(dummy_input)
    print("Output shape:", output.shape)  # Expected: torch.Size([8, 5])
    assert output.shape == (8, 5), "Output shape is incorrect"
    
    # Compute a dummy loss and run backward pass.
    dummy_target = torch.randn(8, 5)
    loss = nn_torch.MSELoss()(output, dummy_target)
    loss.backward()
    
    # Check gradients on the shared trunk's first layer.
    # Assuming self.shared_trunk[0] is a Sequential, access its first module.
    first_linear = predictor.shared_trunk[0][0] if isinstance(predictor.shared_trunk[0], nn_torch.Sequential) else predictor.shared_trunk[0]
    print("Gradient norm for shared trunk's first layer weight:", first_linear.weight.grad.norm())

def train_and_compare():
    # Generate synthetic data:
    batch_size = 64
    input_dim = 100
    n_samples = 1024
    X = torch.randn(n_samples, input_dim)
    true_W = torch.randn(input_dim, 5)
    true_b = torch.randn(5)
    Y = X @ true_W + true_b + 0.1 * torch.randn(n_samples, 5)

    # Create DataLoader:
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the multi-head predictor:
    multihead_model = nn.MultiHeadRegressionFFN(
        input_dim=input_dim,
        n_tasks=5,
        n_targets=1,
        trunk_hidden_dim=64,
        trunk_layers=2,
        head_layers=2,
        head_hidden_dim=32,
        head_dropout=[0.0, 0.1, 0.2, 0.3, 0.4],
        head_activation="relu",
        dropout=0.0,
        activation="relu",
    )
    # For comparison, define a baseline model that outputs 5 values directly.
    baseline_model = nn_torch.Sequential(
        nn_torch.Linear(input_dim, 64),
        nn_torch.ReLU(),
        nn_torch.Linear(64, 5)
    )

    opt_multi = optim.Adam(multihead_model.parameters(), lr=1e-3)
    opt_base = optim.Adam(baseline_model.parameters(), lr=1e-3)

    def train_model(model, optimizer, loader, epochs=5):
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch, Y_batch in loader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = nn_torch.MSELoss()(preds, Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    print("Training Multi-Head Model:")
    train_model(multihead_model, opt_multi, loader)

    print("Training Baseline Model:")
    train_model(baseline_model, opt_base, loader)

if __name__ == "__main__":
    print("Testing Multi-Head Predictor:")
    test_multihead_predictor()
    print("\nTraining and Comparing Models:")
    train_and_compare()