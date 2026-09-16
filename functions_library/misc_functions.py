import numpy as np
import torch

def compute_mean_std(dataset, sample_size=100_000, batch_size=256):
    sample_size = min(sample_size, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)

    feature_sum = None
    feature_squared_sum = None
    count = 0

    for start in range(0, sample_size, batch_size):
        batch_indices = indices[start:start + batch_size]
        sampled_data = dataset.__getitems__(batch_indices)

        features = torch.stack([
            feature_row for feature_row, _ in sampled_data
        ]).double()

        finite = torch.isfinite(features)
        safe_features = torch.where(finite, features, torch.zeros_like(features))

        if feature_sum is None:
            feature_sum = torch.zeros(features.shape[1], dtype=torch.float64)
            feature_squared_sum = torch.zeros(features.shape[1], dtype=torch.float64)

        feature_sum += safe_features.sum(dim=0)
        feature_squared_sum += (safe_features ** 2).sum(dim=0)
        count += finite.sum(dim=0)

    means = feature_sum / count
    variances = feature_squared_sum / count - means ** 2
    stds = torch.sqrt(torch.clamp(variances, min=0))

    stds = torch.where(
        torch.isfinite(stds) & (stds > 0),
        stds,
        torch.ones_like(stds),
    )

    return means.float().numpy(), stds.float().numpy()