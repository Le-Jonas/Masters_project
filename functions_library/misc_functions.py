import h5py
import numpy as np
import torch
from pathlib import Path

def _h5_files_from_path(path):
    """
    Given a path, return a list of H5 files. If the path is a file, return a list containing that file.
    If the path is a directory, return a sorted list of all H5 files in that directory.

    Arguments:
    - path (str or Path): The path to a file or directory.

    Returns:
    - list: A list of Path objects representing the H5 files.
    """
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(file for file in path.iterdir() if file.is_file() and file.suffix == ".h5")
    raise FileNotFoundError(f"H5 path does not exist: {path}")

def _compute_Z_mass(pt1, eta1, phi1, e1, pt2, eta2, phi2, e2):
    """
    Given the transverse momentum (pt), pseudorapidity (eta), azimuthal angle (phi), and energy (e) of two particles, compute the invariant mass of the Z boson formed by these two particles.

    Arguments:
    - pt1, eta1, phi1, e1: Values or arrays of values of transverse momentum, pseudorapidity, azimuthal angle, and energy for the first particle.
    - pt2, eta2, phi2, e2: Values or arrays of values of transverse momentum, pseudorapidity, azimuthal angle, and energy for the second particle.

    Returns:
    - z_mass: Value or array of invariant mass(es) of the Z boson(s) formed by the particle pair(s).
    """
    x1 = pt1 * np.cos(phi1)
    y1 = pt1 * np.sin(phi1)
    z1 = pt1 * np.sinh(eta1)
    x2 = pt2 * np.cos(phi2)
    y2 = pt2 * np.sin(phi2)
    z2 = pt2 * np.sinh(eta2)
    z_mass = np.sqrt(np.abs((e1 + e2)**2 - (x1 + x2)**2 - (y1 + y2)**2 - (z1 + z2)**2))
    return z_mass

def compute_mean_std(dataset, sample_size=100_000, batch_size=256):
    """
    Given a dataset with two feature outputs and a target output and , compute the mean and standard deviation of its features for normalization. The computation is done using a random sample of the dataset to improve efficiency.

    Arguments:
    - dataset: A dataset object that implements the __getitems__ method to return a tuple of (features, event_features, target) for a given index.

    Keywords:
    - sample_size (int): The number of samples to use for computing the mean and standard deviation. Default is 100,000.
    - batch_size (int): The number of samples to process in each batch. Default is 256.

    Returns:
    - means: A tuple containing the mean of the features and the mean of the event features.
    - stds: A tuple containing the standard deviation of the features and the standard deviation of the event features.
    """
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

    feature_sum_event = None
    feature_squared_sum_event = None


    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        sampled_data = dataset.__getitems__(batch_indices, include_target=False)

        features = torch.stack([
            feature_row for feature_row, _, _ in sampled_data
        ]).double()
        event_features = torch.stack([
            event_row for _, event_row, _ in sampled_data
        ]).double()

        finite = torch.isfinite(features)
        safe_features = torch.where(finite, features, torch.zeros_like(features))

        if feature_sum is None:
            feature_sum = torch.zeros(features.shape[1], dtype=torch.float64)
            feature_squared_sum = torch.zeros(features.shape[1], dtype=torch.float64)
            count = torch.zeros(features.shape[1], dtype=torch.float64)

        feature_sum += safe_features.sum(dim=0)
        feature_squared_sum += (safe_features ** 2).sum(dim=0)
        count += finite.sum(dim=0)

        finite_event = torch.isfinite(event_features)
        safe_event_features = torch.where(finite_event, event_features, torch.zeros_like(event_features))


        if feature_sum_event is None:
            feature_sum_event = torch.zeros(event_features.shape[1], dtype=torch.float64)
            feature_squared_sum_event = torch.zeros(event_features.shape[1], dtype=torch.float64)
            count_event = torch.zeros(event_features.shape[1], dtype=torch.float64)

        feature_sum_event += safe_event_features.sum(dim=0)
        feature_squared_sum_event += (safe_event_features ** 2).sum(dim=0)
        count_event += finite_event.sum(dim=0)

        print(f"Calculating mean and std for normalization using sample size {sample_size}, {start + batch_size} samples processed", end='\r')

    means = feature_sum / count
    variances = feature_squared_sum / count - means ** 2
    stds = torch.sqrt(torch.clamp(variances, min=0))

    stds = torch.where(
        torch.isfinite(stds) & (stds > 0),
        stds,
        torch.ones_like(stds),
    )

    means_event = feature_sum_event / count_event
    variances_event = feature_squared_sum_event / count_event - means_event ** 2
    stds_event = torch.sqrt(torch.clamp(variances_event, min=0))

    stds_event = torch.where(
        torch.isfinite(stds_event) & (stds_event > 0),
        stds_event,
        torch.ones_like(stds_event),
    )

    return (means.float(), means_event.float()), (stds.float(), stds_event.float())



def find_Z_peak(h5_files_path, csv_output_path="z_masses.csv", namespace="electron", global_mask = False):
    """
    Given a path to H5 files, compute the invariant mass of Z bosons formed by pairs of particles (electrons, muons, or taus) and save the results to a CSV file. Optionally, apply a global mask to filter the particles.

    Arguments:
    - h5_files_path (str or Path): The path to a file or directory containing H5 files.

    Keywords:
    - csv_output_path (str): The path to the output CSV file where the Z masses will be saved. Default is "z_masses.csv".
    - namespace (str): The type of particles to consider for Z boson formation. Must be one of "electron", "muon", or "tau". Default is "electron".
    - global_mask (array-like or bool): An optional mask to filter the particles. If False, no mask is applied. If provided, it should be a boolean array of the same length as the total number of particles across all H5 files.

    Returns:
    - int: Returns 1 upon successful completion.
    """
    print(f"Finding Z peak in H5 files at {h5_files_path}")
    h5_files = _h5_files_from_path(h5_files_path)
    print(f"Found {len(h5_files)} H5 files for Z peak calculation", end='\r')

    if namespace == "electron":
        name_feature = "egammas"
        name_n = "nEgammas"
        name_index = "firstEgammaIndex"
    elif namespace == "muon":
        name_feature = "muons"
        name_n = "nMuons"
        name_index = "firstMuonIndex"
    elif namespace == "tau":
        name_feature = "taus"
        name_n = "nTaus"
        name_index = "firstTauIndex"
    else:
        raise ValueError(f"Invalid namespace: {namespace}. Must be one of 'electron', 'muon', or 'tau'.")

    
    z_masses = []
    global_mask_index = 0
    for j, file in enumerate(h5_files):
        with h5py.File(file, 'r') as f:
            n_egammas = f["eventwise"][name_n]
            indexes = np.asarray(f["eventwise"][name_index], dtype=np.uint64)

            index_s = np.where(indexes[1:] < indexes[:-1])[0]
            if len(index_s) > 0:
                print(f"Warning: Indexes are not strictly increasing in file {file}. Found {len(index_s)} decreasing indexes at positions {index_s}.")
                for i in index_s:
                    indexes[i + 1:] += (indexes[i] - indexes[i + 1] + n_egammas[i])

            mask = np.zeros(indexes[-1] + n_egammas[-1], dtype=bool)
            local_mask = None
            if global_mask is not False:
                local_mask = np.asarray(
                    global_mask[global_mask_index:global_mask_index + len(mask)],
                    dtype=bool,
                )
                if len(local_mask) != len(mask):
                    raise ValueError(
                        f"Global mask length {len(global_mask)} does not match "
                        f"the expected length {len(mask)} for file {file}."
                    )
                global_mask_index += len(mask)

            for start, count in zip(indexes, n_egammas):
                if count != 2:
                    continue
                start = int(start)
                if local_mask is None or local_mask[start] and local_mask[start + 1]:
                    mask[start:start + 2] = True

            pt = f[name_feature]["pt"][mask]
            eta = f[name_feature]["eta"][mask]
            phi = f[name_feature]["phi"][mask]
            e = f[name_feature]["e"][mask]

            pt1, eta1, phi1, e1 = pt[::2], eta[::2], phi[::2], e[::2]
            pt2, eta2, phi2, e2 = pt[1::2], eta[1::2], phi[1::2], e[1::2]

            z_mass = _compute_Z_mass(pt1, eta1, phi1, e1, pt2, eta2, phi2, e2)
            z_masses.extend(z_mass)
        print(f"Processed file {j+1}/{len(h5_files)}: Found {len(z_mass)} Z masses", end='\r')

    print(f"Saving Z masses to {csv_output_path}")
    np.savetxt(csv_output_path, z_masses)
    return 1

def find_Z_peak_pairs(h5_files_path, csv_output_path="z_masses.csv", namespace="electron", global_mask = False):
    """
    Given a path to H5 files, compute the invariant mass of Z bosons formed by pairs of particles (electrons, muons, or taus) and save the results to a CSV file. Optionally, apply a global mask to filter the particles.

    Arguments:
    - h5_files_path (str or Path): The path to a file or directory containing H5 files.

    Keywords:
    - csv_output_path (str): The path to the output CSV file where the Z masses will be saved. Default is "z_masses.csv".
    - namespace (str): The type of particles to consider for Z boson formation. Must be one of "electron", "muon", or "tau". Default is "electron".
    - global_mask (array-like or bool): An optional mask to filter the pairs. If False, no mask is applied. If provided, it should be a boolean array of the same length as the total number of particle pairs across all H5 files.
    
    Returns:
    - int: Returns 1 upon successful completion.
    """
    print(f"Finding Z peak in H5 files at {h5_files_path}")
    h5_files = _h5_files_from_path(h5_files_path)
    print(f"Found {len(h5_files)} H5 files for Z peak calculation", end='\r')

    if namespace == "electron":
        name_feature = "egammas"
        name_n = "nEgammas"
        name_index = "firstEgammaIndex"
    elif namespace == "muon":
        name_feature = "muons"
        name_n = "nMuons"
        name_index = "firstMuonIndex"
    elif namespace == "tau":
        name_feature = "taus"
        name_n = "nTaus"
        name_index = "firstTauIndex"
    else:
        raise ValueError(f"Invalid namespace: {namespace}. Must be one of 'electron', 'muon', or 'tau'.")

    
    z_masses = []
    global_mask_index = 0
    ###TEMP WILL BE CONTINUED ONCE MORALE IMPROVES!


def get_data_length(h5_files_path):
    """
    Given a path to H5 files, compute the total number of events across all files. If the path is a file, return the number of events in that file. If the path is a directory, return the sum of events across all H5 files in that directory.

    Arguments:
    - h5_files_path (str or Path): The path to a file or directory containing H5 files.
    
    Returns:
    - int: The total number of events across all H5 files.
    """
    h5_files = _h5_files_from_path(h5_files_path)
    total_length = 0
    for i, file in enumerate(h5_files):
        with h5py.File(file, 'r') as f:
            total_length += len(f["eventwise"]["eventNumber"])
        print(f"Processed file {i+1}/{len(h5_files)}: Current total length is {total_length}", end='\r')
    return total_length