import h5py
import numpy as np
import torch
from pathlib import Path

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

def find_Z_peak(h5_files_path):
    print(f"Finding Z peak in H5 files at {h5_files_path}")
    def h5_files_from_path(path):
        path = Path(path)
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(file for file in path.iterdir() if file.is_file() and file.suffix == ".h5")
        raise FileNotFoundError(f"H5 path does not exist: {path}")

    h5_files = h5_files_from_path(h5_files_path)
    print(f"Found {len(h5_files)} H5 files for Z peak calculation", end='\r')


    z_masses = []
    for file in h5_files:
        with h5py.File(file, 'r') as f:
            n_egammas = f["eventwise"]["nEgammas"]
            indexes = f["eventwise"]["firstEgammaIndex"]

            mask = np.array([0] * (indexes[-1] + n_egammas[-1]), dtype=bool)
            for i in range(len(n_egammas)):
                if n_egammas[i] == 2:
                    mask[indexes[i]:indexes[i] + 2] = True

            pt = f["egammas"]["pt"][mask]
            eta = f["egammas"]["eta"][mask]
            phi = f["egammas"]["phi"][mask]
            e = f["egammas"]["e"][mask]

            pt1, eta1, phi1, e1 = pt[::2], eta[::2], phi[::2], e[::2]
            pt2, eta2, phi2, e2 = pt[1::2], eta[1::2], phi[1::2], e[1::2]
            
            z_mass = compute_Z_mass(pt1, eta1, phi1, e1, pt2, eta2, phi2, e2)
            z_masses.extend(z_mass)

    np.savetxt("z_masses.csv", z_masses)
    return 1

def compute_Z_mass(pt1, eta1, phi1, e1, pt2, eta2, phi2, e2):
    x1 = pt1 * np.cos(phi1)
    y1 = pt1 * np.sin(phi1)
    z1 = pt1 * np.sinh(eta1)
    x2 = pt2 * np.cos(phi2)
    y2 = pt2 * np.sin(phi2)
    z2 = pt2 * np.sinh(eta2)
    z_mass = np.sqrt((e1 + e2)**2 - (x1 + x2)**2 - (y1 + y2)**2 - (z1 + z2)**2)
    return z_mass