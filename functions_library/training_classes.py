import torch


class GaussianNLLLoss(torch.nn.Module):
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
        Computes the Gaussian negative log-likelihood loss.

        Arguments:
            outputs: A tensor whose final dimension contains alternating
                mean and log-variance values. For pair targets, the expected
                layout is ``[mean_1, logvar_1, mean_2, logvar_2]``.
            targets: A tensor with one target value for each predicted mean.
        
        Returns:
            A scalar or per-sample tensor, depending on ``reduction``.
        """
        target_values = (
            targets.reshape(targets.shape[0], -1)
            if targets.ndim > 1
            else targets
        )

        if outputs.shape[-1] == 2:
            mean = outputs[..., 0]
            log_variance = outputs[..., 1]
        elif (
            outputs.shape[-1] % 2 == 0
            and target_values.ndim == 2
            and target_values.shape[-1] == outputs.shape[-1] // 2
        ):
            mean = outputs[..., 0::2]
            log_variance = outputs[..., 1::2]
        else:
            raise ValueError(
                "GaussianNLLLoss expects output_size=2 for scalar targets or "
                "an even output_size with one mean/log-variance pair per "
                "target value."
            )

        log_variance = log_variance.clamp(-10.0, 10.0)
        targets = target_values.reshape_as(mean)
        squared_error = (targets - mean).square()

        losses = 0.5 * (
            torch.exp(-log_variance) * squared_error
            + log_variance
        )
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses


class AngularGaussianNLLLoss(torch.nn.Module):
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
        """Compute Gaussian NLL using wrapped angular residuals."""
        target_values = (
            targets.reshape(targets.shape[0], -1)
            if targets.ndim > 1
            else targets
        )
        if outputs.shape[-1] == 2:
            mean = outputs[..., 0]
            log_variance = outputs[..., 1]
        elif (
            outputs.shape[-1] % 2 == 0
            and target_values.ndim == 2
            and target_values.shape[-1] == outputs.shape[-1] // 2
        ):
            mean = outputs[..., 0::2]
            log_variance = outputs[..., 1::2]
        else:
            raise ValueError(
                "AngularGaussianNLLLoss expects output_size=2 for scalar "
                "targets or one mean/log-variance pair per target value."
            )

        log_variance = log_variance.clamp(-10.0, 10.0)
        targets = target_values.reshape_as(mean)
        residual = torch.atan2(
            torch.sin(mean - targets),
            torch.cos(mean - targets),
        )
        losses = 0.5 * (
            torch.exp(-log_variance) * residual.square()
            + log_variance
        )
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses