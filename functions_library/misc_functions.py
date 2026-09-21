import numpy as np
import torch

def compute_mean_std(dataset, sample_size=100_000, batch_size=256):
    num_batches = sample_size // batch_size
    num_avaliable_batches = len(dataset) // batch_size
    if num_batches > num_avaliable_batches:
        raise ValueError(f"Sample size {sample_size} is too large for the dataset size {len(dataset)}. "
                         f"Maximum sample size is {num_avaliable_batches * batch_size}.")
    
    batch_indices = np.random.choice(num_avaliable_batches, num_batches, replace=False)
    indices = np.concatenate([np.arange(batch * batch_size, (batch + 1) * batch_size) for batch in batch_indices])
    indices.sort()

    feature_sum = None
    feature_squared_sum = None
    count = 0

    for start in range(0, len(indices), batch_size):
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

        print(f"Calculating mean and std for normalization using sample size {sample_size}, {start + batch_size} samples processed", end='\r')

    means = feature_sum / count
    variances = feature_squared_sum / count - means ** 2
    stds = torch.sqrt(torch.clamp(variances, min=0))

    stds = torch.where(
        torch.isfinite(stds) & (stds > 0),
        stds,
        torch.ones_like(stds),
    )

    return means.float(), stds.float()