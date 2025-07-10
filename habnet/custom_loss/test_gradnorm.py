"""
1. test_train_step_requires_grad
    Verifies that in training mode, with GradNorm enabled, the loss computed by your model requires gradients, and that the gradnorm weights (learnable parameters) receive non-zero gradients during backpropagation.

    - Model is created with GradNorm enabled.
    - A batch of inputs, Z, is generated with requires_grad=True (so the gradients flow back) along with corresponding target values
    - The train step is called, which uses the GradNorm logic to compute the loss (return a scalar loss)
    - After calling loss.backward(), the test asserts that the entire gradnorm_weights parameter has a gradient (and that its shape is as expected and nonzero)

2. test_eval_mode_uses_forward
    Test ensures that when model is in eval mode, the the gradnorm logic is bypassed. In eval mode, the train_step should simply run the forward pass and apply the criterion, resulting in a scalar loss

    - Model is created with GradNorm enabled and set to eval mode
    - The model's criterion is forced to be a simple lambda that calls F.mse_loss, with reduction = 'mean' - ensuring it produces a scalar loss
    - The test computes:
        loss_from_train_step by calling train_step in eval mode
        loss_from_forward by calling forward and then applying the criterion
    - Asserts that both losses are scalars and uses torch.testing.assert_close to verify that they are equal (within a tiny tolerance)

3. test_gradnorm_loss_component_nonzero
    Test checks that when the per-task losses differ (i.e. the targets are on different scales), the additional loss component introduced by your GradNorm logic is nonzero. In other words, the total loss (which includes the GradNorm term)
    should be greater than just the weight sum of per-task losses

    - Model is created with GradNorm enabled and set to training mode
    - Targets are created on different scales (one tasks targets are multiplied by 1, another by 2 and another by 3) so that the losses per task will differ
    - The test computes the total loss using train_step (which included the GradNorm component) and also computes the "base weighted loss" without the gradnorm term
    - The test asserts that the total loss is greater than the base weight loss by a noticeable amount (greater than 1e-6), confirming that the GradNorm component is contributing

"""



import pytest
import torch
import torch.nn.functional as F
from chemprop.nn.predictors import MultiHeadRegressionFFN

# Use a simple lambda for the criterion to avoid internal shape issues.
def create_model(grad_norm: bool = False):
    return MultiHeadRegressionFFN(
        n_tasks=3,
        input_dim=8,
        trunk_hidden_dim=4,
        grad_norm=grad_norm,
        criterion=lambda preds, targets, *args: F.mse_loss(preds, targets, reduction='mean'),
        output_transform=None  # Identity transform
    )

def test_train_step_requires_grad():
    """
    Test that in training mode with gradnorm enabled, the computed loss requires grad
    and that the gradnorm weights (as a whole parameter) receive a gradient.
    """
    model = create_model(grad_norm=True)
    model.train()
    batch_size = 5
    input_dim = 8
    n_tasks = model.n_tasks
    # Create input that requires grad.
    Z = torch.randn(batch_size, input_dim, requires_grad=True)
    targets = torch.randn(batch_size, n_tasks)
    shared_params = list(model.shared_trunk.parameters())
    loss = model.train_step(Z, targets, shared_params)
    assert loss.requires_grad, "Loss should require gradients in training mode with gradnorm enabled."
    loss.backward()
    # Check the gradnorm weights parameter (not individual indexed elements)
    assert model.gradnorm_weights.grad is not None, "Gradnorm weights should have a gradient."
    # Optionally, check that the gradient tensor has the expected shape and nonzero values.
    assert model.gradnorm_weights.grad.shape[0] == n_tasks, "Unexpected gradnorm weights gradient shape."
    assert torch.any(model.gradnorm_weights.grad != 0), "Gradnorm weights gradient should be nonzero."

def test_eval_mode_uses_forward():
    """
    Test that in evaluation mode, the gradnorm logic is bypassed and
    train_step returns the result of forward followed by the criterion.
    """
    model = create_model(grad_norm=True)
    model.eval()  # Set model to evaluation mode.
    # Explicitly ensure that model.training is False.
    assert model.training is False, "Model should be in eval mode (training flag False)."
    # Force the criterion to be a simple lambda so that it returns a scalar.
    model.criterion = lambda preds, targets, *args: F.mse_loss(preds, targets, reduction='mean')
    
    batch_size = 5
    input_dim = 8
    n_tasks = model.n_tasks
    Z = torch.randn(batch_size, input_dim)
    targets = torch.randn(batch_size, n_tasks)
    
    # In eval mode, train_step should simply call forward and then apply the criterion.
    loss_from_train_step = model.train_step(Z, targets, shared_params=None)
    preds = model.forward(Z)
    loss_from_forward = model.criterion(preds, targets)
    
    # Check that both returned losses are scalars.
    assert loss_from_train_step.dim() == 0, f"Expected scalar loss from train_step, got shape {loss_from_train_step.shape}"
    assert loss_from_forward.dim() == 0, f"Expected scalar loss from criterion, got shape {loss_from_forward.shape}"
    
    torch.testing.assert_close(loss_from_train_step, loss_from_forward, rtol=0, atol=1e-6)
 

def test_gradnorm_loss_component_nonzero():
    """
    Test that when per-task losses differ (using targets on different scales),
    the additional gradnorm loss component is nonzero.
    """
    model = create_model(grad_norm=True)
    model.train()
    batch_size = 5
    input_dim = 8
    n_tasks = model.n_tasks
    # Create targets with different scales.
    scales = [1.0, 2.0, 3.0]
    targets = torch.cat([torch.randn(batch_size, 1) * s for s in scales], dim=1)
    Z = torch.randn(batch_size, input_dim, requires_grad=True)
    shared_params = list(model.shared_trunk.parameters())
    loss = model.train_step(Z, targets, shared_params)
    preds = model.forward(Z)
    base_losses = [F.mse_loss(preds[:, i], targets[:, i]) for i in range(n_tasks)]
    weighted_loss = sum([model.gradnorm_weights[i] * base_losses[i] for i in range(n_tasks)])
    # If the gradnorm loss were zero, total loss would equal weighted_loss.
    # We expect the total loss to be noticeably larger.
    assert loss.item() > weighted_loss.item() + 1e-6, "Total loss should include a nonzero gradnorm component."

def test_gradnorm_weights_update():
    """
    Test that after a training step and a backward pass, the gradnorm weights receive gradients.
    (We check gradients on the parameter before the optimizer step.)
    """
    model = create_model(grad_norm=True)
    model.train()
    batch_size = 5
    input_dim = 8
    n_tasks = model.n_tasks
    Z = torch.randn(batch_size, input_dim, requires_grad=True)
    targets = torch.randn(batch_size, n_tasks)
    shared_params = list(model.shared_trunk.parameters())
    loss = model.train_step(Z, targets, shared_params)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    # Check gradients on the whole gradnorm weights parameter.
    assert model.gradnorm_weights.grad is not None, "Gradnorm weights should have a gradient after backward."
    assert torch.any(model.gradnorm_weights.grad != 0), "Gradnorm weights gradient should be nonzero after backward."
    optimizer.step()
    # Check that the weights have been updated (i.e. not still equal to 1.0).
    assert torch.any(model.gradnorm_weights.detach() != 1.0), "Gradnorm weights should have been updated from their initial value."

if __name__ == "__main__":
    pytest.main([__file__])
