import torch
from torch import nn

class GradNormMixin(nn.Module):
    def __init__(self, n_tasks: int | None = None):
        super().__init__()
        # initialise weights
        if n_tasks is not None:
            self.gradnorm_weights = nn.Parameter(torch.ones(n_tasks), requires_grad=True)
