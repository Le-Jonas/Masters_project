import torch
import torch.nn.functional as F


class _GaussianNLLBase(torch.nn.Module):
    def __init__(
        self,
        reduction: str,
        min_log_variance: float,
        max_log_variance: float,
    ):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "reduction must be 'none', 'mean', or 'sum'."
            )
        if min_log_variance >= max_log_variance:
            raise ValueError(
                "min_log_variance must be smaller than max_log_variance."
            )
        self.reduction = reduction
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance

    def _split_outputs(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if outputs.ndim == 0 or outputs.shape[-1] == 0:
            raise ValueError(f"{loss_name} expects a non-empty output tensor.")

        target_values = (
            targets.reshape(targets.shape[0], -1)
            if targets.ndim > 1
            else targets
        )

        if outputs.shape[-1] == 2:
            mean = outputs[..., 0]
            raw_log_variance = outputs[..., 1]
        elif (
            outputs.shape[-1] % 2 == 0
            and target_values.ndim == 2
            and target_values.shape[-1] == outputs.shape[-1] // 2
        ):
            mean = outputs[..., 0::2]
            raw_log_variance = outputs[..., 1::2]
        else:
            raise ValueError(
                f"{loss_name} expects output_size=2 for scalar targets or "
                "an even output_size with one mean/log-variance pair per "
                "target value."
            )

        try:
            target_values = target_values.reshape_as(mean)
        except RuntimeError as error:
            raise ValueError(
                f"{loss_name} received targets with shape {tuple(targets.shape)} "
                f"that is incompatible with outputs of shape {tuple(outputs.shape)}."
            ) from error

        return mean, raw_log_variance, target_values

    def _bounded_log_variance(self, raw_log_variance: torch.Tensor) -> torch.Tensor:
        # Softplus keeps the lower variance bound differentiable while the
        # upper bound prevents overflow in exp(-log_variance).
        log_variance = self.min_log_variance + F.softplus(
            raw_log_variance - self.min_log_variance
        )
        return log_variance.clamp(max=self.max_log_variance)

    def _reduce(self, losses: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses


class AngularMSELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "reduction must be 'none', 'mean', or 'sum'."
            )
        self.reduction = reduction

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes mean squared error for periodic angular values.

        Angles are expected in radians. Differences are wrapped to the
        interval [-pi, pi], making -pi and +pi equivalent.
        """
        residual = torch.atan2(
            torch.sin(outputs - targets),
            torch.cos(outputs - targets),
        )
        return F.mse_loss(
            residual,
            torch.zeros_like(residual),
            reduction=self.reduction,
        )


class GaussianNLLLoss(_GaussianNLLBase):
    def __init__(
        self,
        reduction: str = "mean",
        min_log_variance: float = -10.0,
        max_log_variance: float = 7.0,
    ):
        super().__init__(reduction, min_log_variance, max_log_variance)

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the Gaussian negative log-likelihood loss.

        Arguments:
            outputs: A tensor whose final dimension contains alternating
                mean and variance-parameter values. For pair targets, the
                expected layout is ``[mean_1, logvar_1, mean_2, logvar_2]``.
                The variance parameter is smoothly bounded internally.
            targets: A tensor with one target value for each predicted mean.

        Returns:
            A scalar or per-sample tensor, depending on ``reduction``.
        """
        mean, raw_log_variance, targets = self._split_outputs(
            outputs, targets, "GaussianNLLLoss"
        )
        log_variance = self._bounded_log_variance(raw_log_variance)
        squared_error = (targets - mean).square()

        losses = 0.5 * (
            torch.exp(-log_variance) * squared_error
            + log_variance
        )
        return self._reduce(losses)


class AngularGaussianNLLLoss(_GaussianNLLBase):
    def __init__(
        self,
        reduction: str = "mean",
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0,
    ):
        super().__init__(reduction, min_log_variance, max_log_variance)

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian NLL using wrapped angular residuals."""
        mean, raw_log_variance, targets = self._split_outputs(
            outputs, targets, "AngularGaussianNLLLoss"
        )
        log_variance = self._bounded_log_variance(raw_log_variance)
        residual = torch.atan2(
            torch.sin(mean - targets),
            torch.cos(mean - targets),
        )
        losses = 0.5 * (
            torch.exp(-log_variance) * residual.square()
            + log_variance
        )
        return self._reduce(losses)
