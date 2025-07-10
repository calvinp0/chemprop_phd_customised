import torch
from numpy.typing import ArrayLike
from chemprop.nn.metrics import ChempropMetric

class MaskedMSELoss(ChempropMetric):

    def __init__(self, task_weights: ArrayLike = 1.0):
        """
        Custom masked Mean Squared Error Loss.
        
        Args:
            task_weights: A scalar or tensor to weight each task (target) in the loss.
        """
        super().__init__()
        self.task_weights = torch.as_tensor(task_weights, dtype=torch.float).view(1, -1)

    def _calc_unreduced_loss(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                             weights: torch.Tensor, lt_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the element-wise squared error only where mask is True.
        
        Args:
            preds: Tensor of model predictions with shape [batch, n_tasks, ...]
            targets: Tensor of target values with shape [batch, n_tasks]
            mask: Boolean tensor (same shape as targets) with True where the target is present and finite.
            weights: Tensor of weights for each data point (or None).
            lt_mask: Tensor indicating "less than" targets (unused in this MSE version).
            gt_mask: Tensor indicating "greater than" targets (unused in this MSE version).
        
        Returns:
            A tensor of unreduced losses (one per data point per task).
        """

        error  = (preds - targets) ** 2
        # Zero out error where mask is False (i.e. where target is missing)
        error = error * mask.float()

        # Incorporate per-data point weights if provided
        if weights is not None:
            error = error * weights
        
        # Apply task-specific weights (broadcasted along the batch dimension)
        error = error * self.task_weights

        # Return unreduced loss (e.g. shape [batch, n_tasks])
        return error
