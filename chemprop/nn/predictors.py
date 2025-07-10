from abc import abstractmethod

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from chemprop.conf import DEFAULT_HIDDEN_DIM
from chemprop.nn.ffn import MLP
from chemprop.nn.hparams import HasHParams
from chemprop.nn.metrics import (
    MSE,
    SID,
    BCELoss,
    BinaryAUROC,
    ChempropMetric,
    CrossEntropyLoss,
    DirichletLoss,
    EvidentialLoss,
    MulticlassMCCMetric,
    MVELoss,
    QuantileLoss,
)
from chemprop.nn.transforms import UnscaleTransform
from chemprop.utils import ClassRegistry, Factory
from chemprop.optimisers.optimisers import GradNormMixin


__all__ = [
    "Predictor",
    "PredictorRegistry",
    "RegressionFFN",
    "MveFFN",
    "EvidentialFFN",
    "BinaryClassificationFFNBase",
    "BinaryClassificationFFN",
    "BinaryDirichletFFN",
    "MulticlassClassificationFFN",
    "MulticlassDirichletFFN",
    "SpectralFFN",
]


class Predictor(nn.Module, HasHParams):
    r"""A :class:`Predictor` is a protocol that defines a differentiable function
    :math:`f` : \mathbb R^d \mapsto \mathbb R^o"""

    input_dim: int
    """the input dimension"""
    output_dim: int
    """the output dimension"""
    n_tasks: int
    """the number of tasks `t` to predict for each input"""
    n_targets: int
    """the number of targets `s` to predict for each task `t`"""
    criterion: ChempropMetric
    """the loss function to use for training"""
    task_weights: Tensor
    """the weights to apply to each task when calculating the loss"""
    output_transform: UnscaleTransform
    """the transform to apply to the output of the predictor"""

    @abstractmethod
    def forward(self, Z: Tensor) -> Tensor:
        pass

    @abstractmethod
    def train_step(self, Z: Tensor) -> Tensor:
        pass

    @abstractmethod
    def encode(self, Z: Tensor, i: int) -> Tensor:
        """Calculate the :attr:`i`-th hidden representation

        Parameters
        ----------
        Z : Tensor
            a tensor of shape ``n x d`` containing the input data to encode, where ``d`` is the
            input dimensionality.
        i : int
            The stop index of slice of the MLP used to encode the input. That is, use all
            layers in the MLP *up to* :attr:`i` (i.e., ``MLP[:i]``). This can be any integer
            value, and the behavior of this function is dependent on the underlying list
            slicing behavior. For example:

            * ``i=0``: use a 0-layer MLP (i.e., a no-op)
            * ``i=1``: use only the first block
            * ``i=-1``: use *up to* the final block

        Returns
        -------
        Tensor
            a tensor of shape ``n x h`` containing the :attr:`i`-th hidden representation, where
            ``h`` is the number of neurons in the :attr:`i`-th hidden layer.
        """
        pass


PredictorRegistry = ClassRegistry[Predictor]()


class _FFNPredictorBase(Predictor, HyperparametersMixin):
    """A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
    underlying :class:`SimpleFFN` to map the learned fingerprint to the desired output.
    """

    _T_default_criterion: ChempropMetric
    _T_default_metric: ChempropMetric

    def __init__(
        self,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        criterion: ChempropMetric | None = None,
        task_weights: Tensor | None = None,
        threshold: float | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        super().__init__()
        # manually add criterion and output_transform to hparams to suppress lightning's warning
        # about double saving their state_dict values.
        self.save_hyperparameters(ignore=["criterion", "output_transform"])
        self.hparams["criterion"] = criterion
        self.hparams["output_transform"] = output_transform
        self.hparams["cls"] = self.__class__

        self.ffn = MLP.build(
            input_dim, n_tasks * self.n_targets, hidden_dim, n_layers, dropout, activation
        )
        task_weights = torch.ones(n_tasks) if task_weights is None else task_weights
        self.criterion = criterion or Factory.build(
            self._T_default_criterion, task_weights=task_weights, threshold=threshold
        )
        self.output_transform = output_transform if output_transform is not None else nn.Identity()

    @property
    def input_dim(self) -> int:
        return self.ffn.input_dim

    @property
    def output_dim(self) -> int:
        return self.ffn.output_dim

    @property
    def n_tasks(self) -> int:
        return self.output_dim // self.n_targets

    def forward(self, Z: Tensor) -> Tensor:
        return self.ffn(Z)

    def encode(self, Z: Tensor, i: int) -> Tensor:
        return self.ffn[:i](Z)


@PredictorRegistry.register("regression")
class RegressionFFN(_FFNPredictorBase):
    n_targets = 1
    _T_default_criterion = MSE
    _T_default_metric = MSE

    def forward(self, Z: Tensor) -> Tensor:
        return self.output_transform(self.ffn(Z))

    train_step = forward


@PredictorRegistry.register("regression-mve")
class MveFFN(RegressionFFN):
    n_targets = 2
    _T_default_criterion = MVELoss

    def forward(self, Z: Tensor) -> Tensor:
        Y = self.ffn(Z)
        mean, var = torch.chunk(Y, self.n_targets, 1)
        var = F.softplus(var)

        mean = self.output_transform(mean)
        if not isinstance(self.output_transform, nn.Identity):
            var = self.output_transform.transform_variance(var)

        return torch.stack((mean, var), dim=2)

    train_step = forward


@PredictorRegistry.register("regression-evidential")
class EvidentialFFN(RegressionFFN):
    n_targets = 4
    _T_default_criterion = EvidentialLoss

    def forward(self, Z: Tensor) -> Tensor:
        Y = self.ffn(Z)
        mean, v, alpha, beta = torch.chunk(Y, self.n_targets, 1)
        v = F.softplus(v)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta)

        mean = self.output_transform(mean)
        if not isinstance(self.output_transform, nn.Identity):
            beta = self.output_transform.transform_variance(beta)

        return torch.stack((mean, v, alpha, beta), dim=2)

    train_step = forward


@PredictorRegistry.register("regression-quantile")
class QuantileFFN(RegressionFFN):
    n_targets = 2
    _T_default_criterion = QuantileLoss

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        lower_bound, upper_bound = torch.chunk(Y, self.n_targets, 1)

        lower_bound = self.output_transform(lower_bound)
        upper_bound = self.output_transform(upper_bound)

        mean = (lower_bound + upper_bound) / 2
        interval = upper_bound - lower_bound

        return torch.stack((mean, interval), dim=2)

    train_step = forward


class BinaryClassificationFFNBase(_FFNPredictorBase):
    pass


@PredictorRegistry.register("classification")
class BinaryClassificationFFN(BinaryClassificationFFNBase):
    n_targets = 1
    _T_default_criterion = BCELoss
    _T_default_metric = BinaryAUROC

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)

        return Y.sigmoid()

    def train_step(self, Z: Tensor) -> Tensor:
        return super().forward(Z)


@PredictorRegistry.register("classification-dirichlet")
class BinaryDirichletFFN(BinaryClassificationFFNBase):
    n_targets = 2
    _T_default_criterion = DirichletLoss
    _T_default_metric = BinaryAUROC

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z).reshape(len(Z), -1, 2)

        alpha = F.softplus(Y) + 1

        u = 2 / alpha.sum(-1)
        Y = alpha / alpha.sum(-1, keepdim=True)

        return torch.stack((Y[..., 1], u), dim=2)

    def train_step(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z).reshape(len(Z), -1, 2)

        return F.softplus(Y) + 1


@PredictorRegistry.register("multiclass")
class MulticlassClassificationFFN(_FFNPredictorBase):
    n_targets = 1
    _T_default_criterion = CrossEntropyLoss
    _T_default_metric = MulticlassMCCMetric

    def __init__(
        self,
        n_classes: int,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        criterion: ChempropMetric | None = None,
        task_weights: Tensor | None = None,
        threshold: float | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        task_weights = torch.ones(n_tasks) if task_weights is None else task_weights
        super().__init__(
            n_tasks * n_classes,
            input_dim,
            hidden_dim,
            n_layers,
            dropout,
            activation,
            criterion,
            task_weights,
            threshold,
            output_transform,
        )

        self.n_classes = n_classes

    @property
    def n_tasks(self) -> int:
        return self.output_dim // (self.n_targets * self.n_classes)

    def forward(self, Z: Tensor) -> Tensor:
        return self.train_step(Z).softmax(-1)

    def train_step(self, Z: Tensor) -> Tensor:
        return super().forward(Z).reshape(Z.shape[0], -1, self.n_classes)


@PredictorRegistry.register("multiclass-dirichlet")
class MulticlassDirichletFFN(MulticlassClassificationFFN):
    _T_default_criterion = DirichletLoss
    _T_default_metric = MulticlassMCCMetric

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().train_step(Z)

        alpha = F.softplus(Y) + 1

        Y = alpha / alpha.sum(-1, keepdim=True)

        return Y

    def train_step(self, Z: Tensor) -> Tensor:
        Y = super().train_step(Z)

        return F.softplus(Y) + 1


class _Exp(nn.Module):
    def forward(self, X: Tensor):
        return X.exp()


@PredictorRegistry.register("spectral")
class SpectralFFN(_FFNPredictorBase):
    n_targets = 1
    _T_default_criterion = SID
    _T_default_metric = SID

    def __init__(self, *args, spectral_activation: str | None = "softplus", **kwargs):
        super().__init__(*args, **kwargs)

        match spectral_activation:
            case "exp":
                spectral_activation = _Exp()
            case "softplus" | None:
                spectral_activation = nn.Softplus()
            case _:
                raise ValueError(
                    f"Unknown spectral activation: {spectral_activation}. "
                    "Expected one of 'exp', 'softplus' or None."
                )

        self.ffn.add_module("spectral_activation", spectral_activation)

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        Y = self.ffn.spectral_activation(Y)
        return Y / Y.sum(1, keepdim=True)

    train_step = forward


@PredictorRegistry.register("multihead-regression")
class MultiHeadRegressionFFN(GradNormMixin, nn.Module, HasHParams, HyperparametersMixin):
    """
    A multi-head regression predictor with a configurable number of targets.
    
    This predictor uses a shared trunk to process the input fingerprint, then
    has separate output heads for each property. In Chemprop's conventions,
    each property is treated as a separate task with one target (n_tasks=number of properties, n_targets=1).
    """
    _T_default_criterion = MSE
    _T_default_metric = MSE

    def __init__(
        self,
        n_tasks: int = 5,           # 5 tasks, one per property
        n_targets: int = 1,         # 1 output per task
        input_dim: int = DEFAULT_HIDDEN_DIM,
        trunk_hidden_dim: int = 300,
        trunk_layers: int = 1,
        # Parameters for head configuration. These can be scalars or lists.
        head_layers: int | list[int] = 1,
        head_hidden_dim: int | list[int] = 128,
        head_dropout: float | list[float] = 0.0,
        head_activation: str | list[str] = "relu",
        dropout: float = 0.0,
        activation: str = "relu",
        criterion: nn.Module | None = None,
        task_weights: torch.Tensor | None = None,
        threshold: float | None = None,
        output_transform: UnscaleTransform | None = None,
        grad_norm: bool = False,
    ):

        super().__init__()
        self.grad_norm = grad_norm
        if grad_norm:
            GradNormMixin.__init__(self, n_tasks)

        self.save_hyperparameters(ignore=["criterion", "output_transform"])
        self.n_tasks = n_tasks  # number of tasks, which equals number of heads
        self.n_targets = n_targets  # always 1 for each task in this design

        # Build shared trunk from input_dim to trunk_hidden_dim.
        self.shared_trunk = MLP.build(
            input_dim=input_dim,
            output_dim=trunk_hidden_dim,
            hidden_dim=trunk_hidden_dim,
            n_layers=trunk_layers,
            dropout=dropout,
            activation=activation,
        )

        # If scalar parameters are provided, convert them into lists of length n_tasks.
        if isinstance(head_layers, int):
            head_layers = [head_layers] * self.n_tasks
        if isinstance(head_hidden_dim, int):
            head_hidden_dim = [head_hidden_dim] * self.n_tasks
        if isinstance(head_dropout, (int, float)):
            head_dropout = [head_dropout] * self.n_tasks
        if isinstance(head_activation, str):
            head_activation = [head_activation] * self.n_tasks

        # Build one head per task using the MLP builder.
        self.heads = nn.ModuleList([
            MLP.build(
                input_dim=trunk_hidden_dim,
                output_dim=1,
                hidden_dim=head_hidden_dim[i],
                n_layers=head_layers[i],
                dropout=head_dropout[i],
                activation=head_activation[i],
            )
            for i in range(self.n_tasks)
        ])

        if task_weights is None:
            task_weights = torch.ones(n_tasks)
        self.criterion = criterion or Factory.build(self._T_default_criterion, task_weights=task_weights, threshold=threshold)
        self.output_transform = output_transform if output_transform is not None else nn.Identity()

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Compute shared representation.
        shared_repr = self.shared_trunk(Z)
        # Process with each head. Each head returns a tensor of shape [batch, 1].
        outputs = [head(shared_repr) for head in self.heads]
        # Concatenate outputs along dimension 1 to get [batch, n_tasks] i.e. [batch, 5].
        out = torch.cat(outputs, dim=1)
        # Do not unsqueeze here. Return shape [batch, 5] so that it matches other predictors.
        return self.output_transform(out)

    def train_step(self, Z: torch.Tensor, targets: torch.Tensor | None = None, shared_params: list | None = None) -> torch.Tensor:
        if self.grad_norm and self.training:
            return self.gradnorm_train_step(Z, targets, shared_params)
        else:
            return self.forward(Z)


    def encode(self, Z: torch.Tensor, i: int) -> torch.Tensor:
        return self.shared_trunk(Z)


    def gradnorm_train_step(self, Z: torch.Tensor, targets: torch.Tensor, shared_params: list) -> torch.Tensor:
        # 1. Forward pass.
        preds = self.forward(Z)

        # 2. Compute per-task losses.
        task_losses = []
        for i in range(self.n_tasks):
            loss_i = F.mse_loss(preds[:, i], targets[:, i])
            task_losses.append(loss_i)

        # 3. Compute weighted task losses.
        weighted_losses = [self.gradnorm_weights[i] * task_losses[i] for i in range(self.n_tasks)]
        total_task_loss = sum(weighted_losses)

        # 4. Compute gradient norms for each task using autograd.grad.
        grad_norms = []
        for i in range(self.n_tasks):
            # Compute gradients of the weighted loss with respect to shared parameters.
            grads = torch.autograd.grad(weighted_losses[i], shared_params, retain_graph=True, create_graph=True)
            norm = sum([g.norm(2) for g in grads if g is not None])
            grad_norms.append(norm)

        # 5. Record initial task losses if not set.
        if not hasattr(self, 'initial_task_losses'):
            self.initial_task_losses = [loss.item() for loss in task_losses]

        avg_grad_norm = sum(grad_norms) / self.n_tasks
        gamma = 0.5  # Hyperparameter to tune.
        r = torch.tensor([task_losses[i].item() / self.initial_task_losses[i] for i in range(self.n_tasks)], device=avg_grad_norm.device)
        target_norms = [avg_grad_norm * (r_i ** gamma) for r_i in r]

        # 6. Compute GradNorm loss.
        gradnorm_loss = sum([abs(grad_norms[i] - target_norms[i]) for i in range(self.n_tasks)])

        # 7. Total loss.
        total_loss = total_task_loss + gradnorm_loss
        return total_loss