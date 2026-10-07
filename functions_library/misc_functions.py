import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from .dataset_classes import H5EgammaDataset


class _FeatureOnlyDataset(Dataset):
    """Expose feature-only batched reads to DataLoader workers."""

    def __init__(self, dataset : Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[index]

    def __getitems__(self, indices : list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.dataset.__getitems__(indices, include_target=False)

def _h5_files_from_path(path : str | Path) -> list[Path]:
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

def _compute_Z_mass(pt1 : float, eta1 : float, phi1 : float, e1 : float, pt2 : float, eta2 : float, phi2 : float, e2 : float) -> float:
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

def _selected_rows_for_files(h5_files : list[Path], mix_files : str | Path | list[Path] | None, mixture_ratio : float, mixture_seed : int) -> tuple[list[Path], list[np.ndarray]]:
    """Return the row selections used by H5EgammaDataset for each file."""
    if mix_files is None:
        dataset_mix_files = None
    elif isinstance(mix_files, (str, Path)):
        mix_path = Path(mix_files)
        dataset_mix_files = (
            sorted(
                file for file in mix_path.iterdir()
                if file.is_file() and file.suffix == ".h5"
            )
            if mix_path.is_dir()
            else [mix_path]
        )
    else:
        dataset_mix_files = list(mix_files)

    dataset = H5EgammaDataset(
        files=h5_files,
        mix_files=dataset_mix_files,
        mixture_ratio=mixture_ratio,
        mixture_seed=mixture_seed,
        y_field=None,
    )
    files = dataset.files
    valid_rows = dataset.valid_rows
    dataset.close()
    return files, valid_rows

def compute_mean_std(dataset : Dataset, sample_size : int = 100_000, batch_size : int = 256, num_workers : int = 0) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Given a dataset with two feature outputs and a target output and , compute the mean and standard deviation of its features for normalization. The computation is done using a random sample of the dataset to improve efficiency.

    Arguments:
    - dataset: A dataset object that implements the __getitems__ method to return a tuple of (features, event_features, target) for a given index.

    Keywords:
    - sample_size (int): The number of samples to use for computing the mean and standard deviation. Default is 100,000.
    - batch_size (int): The number of samples to process in each batch. Default is 256.
    - num_workers (int): Number of DataLoader worker processes used for reads. Default is 0.

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

    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")

    loader = None
    if num_workers:
        loader = DataLoader(
            _FeatureOnlyDataset(dataset),
            batch_size=batch_size,
            sampler=indices.tolist(),
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    feature_sum = None
    feature_squared_sum = None

    feature_sum_event = None
    feature_squared_sum_event = None


    batches = loader if loader is not None else (
        dataset.__getitems__(indices[start:start + batch_size], include_target=False)
        for start in range(0, len(indices), batch_size)
    )
    for batch_number, sampled_data in enumerate(batches):
        if loader is None:
            features = torch.stack([
                feature_row for feature_row, _, _ in sampled_data
            ]).double()
            event_features = torch.stack([
                event_row for _, event_row, _ in sampled_data
            ]).double()
        else:
            features, event_features, _ = sampled_data
            features = features.double()
            event_features = event_features.double()

        start = batch_number * batch_size

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



def find_Z_peak(
    h5_files_path : str | Path,
    csv_output_path : str | Path = "z_masses.csv",
    namespace : str = "electron",
    global_mask : bool | np.ndarray = False,
    mix_files_path : str | Path | list[Path] | None =None,
    mixture_ratio : float | None = None,
    mixture_seed : int = 0,
) -> int:
    """
    Given a path to H5 files, compute the invariant mass of Z bosons formed by pairs of particles (electrons, muons, or taus) and save the results to a CSV file. Optionally, apply a global mask to filter the particles.

    Arguments:
    - h5_files_path (str or Path): The path to a file or directory containing H5 files.

    Keywords:
    - csv_output_path (str): The path to the output CSV file where the Z masses will be saved. Default is "z_masses.csv".
    - namespace (str): The type of particles to consider for Z boson formation. Must be one of "electron", "muon", or "tau". Default is "electron".
    - global_mask (array-like or bool): An optional mask to filter the particles. If False, no mask is applied. If provided, it should be a boolean array of the same length as the total number of particles across all H5 files.
    - mix_files (str, Path, or list): Optional mixed H5 file(s), following H5EgammaDataset semantics.
    - mixture_ratio (float): Fraction of rows selected from the main files when mixing.
    - mixture_seed (int): Seed used for deterministic mixture row selection.

    Returns:
    - int: Returns 1 upon successful completion.
    """
    print(f"Finding Z peak in H5 files at {h5_files_path}")
    h5_files = _h5_files_from_path(h5_files_path)
    mix_files = _h5_files_from_path(mix_files_path) if mix_files_path is not None else None
    h5_files, selected_rows = _selected_rows_for_files(
        h5_files, mix_files, mixture_ratio, mixture_seed
    )
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

            file_rows = selected_rows[j]
            if global_mask is not False:
                local_mask = np.asarray(
                    global_mask[global_mask_index:global_mask_index + len(file_rows)],
                    dtype=bool,
                )
                if len(local_mask) != len(file_rows):
                    raise ValueError(
                        f"Global mask length {len(global_mask)} does not match "
                        f"the expected selected-row length for file {file}."
                    )
                global_mask_index += len(file_rows)
            else:
                local_mask = None

            pt = f[name_feature]["pt"]
            eta = f[name_feature]["eta"]
            phi = f[name_feature]["phi"]
            e = f[name_feature]["e"]

            data = {"pt": pt, "eta": eta, "phi": phi, "e": e}

            two_particle_events = np.asarray(n_egammas) == 2
            two_particle_starts = indexes[two_particle_events].astype(np.int64)
            if mix_files_path is None:
                first_rows = two_particle_starts
                second_rows = first_rows + 1
                selected_positions = first_rows
            else:
                event_positions = np.searchsorted(
                    file_rows, two_particle_starts, side="left"
                )
                event_ends = np.searchsorted(
                    file_rows, two_particle_starts + 2, side="left"
                )
                complete_events = (event_ends - event_positions) == 2
                event_positions = event_positions[complete_events]
                first_rows = file_rows[event_positions]
                second_rows = file_rows[event_positions + 1]
                selected_positions = event_positions

            if local_mask is not None:
                keep = (
                    local_mask[selected_positions]
                    & local_mask[selected_positions + 1]
                )
                first_rows = first_rows[keep]
                second_rows = second_rows[keep]

            values = [
                data[field][rows]
                for rows in (first_rows, second_rows)
                for field in ("pt", "eta", "phi", "e")
            ]
            z_masses.extend(_compute_Z_mass(
                values[0], values[1], values[2], values[3],
                values[4], values[5], values[6], values[7],
            ))
            z_count = len(first_rows)
        print(f"Processed file {j+1}/{len(h5_files)}: Found {z_count} Z masses", end='\r')

    if global_mask is not False and global_mask_index != len(global_mask):
        raise ValueError(
            f"Global mask length {len(global_mask)} does not match "
            f"the expected selected-row length {global_mask_index}."
        )

    print(f"Saving Z masses to {csv_output_path}")
    np.savetxt(csv_output_path, z_masses)
    return 1

def find_Z_peak_pairs(
    h5_files_path : str | Path,
    csv_output_path : str | Path = "z_masses.csv",
    namespace : str = "electron",
    global_mask : bool | np.ndarray = False,
    mix_files_path : str | Path | list[Path] | None = None,
    mixture_ratio : float | None = None,
    mixture_seed : int = 0,
) -> int:
    """
    Given a path to H5 files, compute the invariant mass of Z bosons formed by pairs of particles (electrons, muons, or taus) and save the results to a CSV file. Optionally, apply a global mask to filter the particles.

    Arguments:
    - h5_files_path (str or Path): The path to a file or directory containing H5 files.

    Keywords:
    - csv_output_path (str): The path to the output CSV file where the Z masses will be saved. Default is "z_masses.csv".
    - namespace (str): The type of particles to consider for Z boson formation. Must be one of "electron", "muon", or "tau". Default is "electron".
    - global_mask (array-like or bool): An optional mask to filter the pairs. If False, no mask is applied. If provided, it should be a boolean array of the same length as the total number of particle pairs across all H5 files.
    - mix_files_path (str or Path): Optional path to mixed H5 file(s), following ZPairDataset semantics.
    - mixture_ratio (float): Fraction of rows selected from the main files when mixing.
    - mixture_seed (int): Seed used for deterministic mixture row selection.
    
    Returns:
    - int: Returns 1 upon successful completion.
    """
    print(f"Finding Z peak in H5 files at {h5_files_path}")
    h5_files = _h5_files_from_path(h5_files_path)
    mix_files = _h5_files_from_path(mix_files_path) if mix_files_path is not None else None

    h5_files, selected_rows = _selected_rows_for_files(
        h5_files, mix_files, mixture_ratio, mixture_seed
    )
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
    n_pairs = 0
    mask_array = None if global_mask is False else np.asarray(global_mask, dtype=bool).reshape(-1)

    for file_index, file in enumerate(h5_files):
        with h5py.File(file, 'r') as f:
            n_egammas = f["eventwise"][name_n]
            indexes = np.asarray(f["eventwise"][name_index], dtype=np.uint64)

            index_s = np.where(indexes[1:] < indexes[:-1])[0]
            if len(index_s) > 0:
                print(f"Warning: Indexes are not strictly increasing in file {file}. Found {len(index_s)} decreasing indexes at positions {index_s}.")
                for i in index_s:
                    indexes[i + 1:] += (indexes[i] - indexes[i + 1] + n_egammas[i])

            pt = f[name_feature]["pt"]
            eta = f[name_feature]["eta"]
            phi = f[name_feature]["phi"]
            e = f[name_feature]["e"]

            data = {"pt": pt, "eta": eta, "phi": phi, "e": e}

            file_rows = selected_rows[file_index]
            event_starts = np.asarray(indexes, dtype=np.int64)
            event_counts = np.asarray(n_egammas, dtype=np.int64)
            event_positions = np.searchsorted(
                file_rows, event_starts, side="left"
            )
            event_ends = np.searchsorted(
                file_rows, event_starts + event_counts, side="left"
            )

            selected_counts = event_ends - event_positions
            pairable_events = np.flatnonzero(selected_counts >= 2)
            pair_counts = selected_counts[pairable_events] * (
                selected_counts[pairable_events] - 1
            ) // 2
            pair_count = int(pair_counts.sum())

            first_row_chunks = []
            second_row_chunks = []
            pair_event_chunks = []

            two_particle_events = pairable_events[
                selected_counts[pairable_events] == 2
            ]
            if len(two_particle_events):
                first_row_chunks.append(file_rows[event_positions[two_particle_events]])
                second_row_chunks.append(
                    file_rows[event_positions[two_particle_events] + 1]
                )
                pair_event_chunks.append(two_particle_events)

            for event_index in pairable_events[
                selected_counts[pairable_events] > 2
            ]:
                event_rows = file_rows[
                    event_positions[event_index]:event_ends[event_index]
                ]
                first_positions, second_positions = np.triu_indices(
                    len(event_rows), k=1
                )
                first_row_chunks.append(event_rows[first_positions])
                second_row_chunks.append(event_rows[second_positions])
                pair_event_chunks.append(
                    np.full(len(first_positions), event_index, dtype=np.int64)
                )

            if first_row_chunks:
                first_rows = np.concatenate(first_row_chunks)
                second_rows = np.concatenate(second_row_chunks)
                pair_event_ids = np.concatenate(pair_event_chunks)
                order = np.argsort(pair_event_ids, kind="stable")
                first_rows = first_rows[order]
                second_rows = second_rows[order]

                pair_slice = slice(n_pairs, n_pairs + pair_count)
                if mask_array is not None:
                    if n_pairs + pair_count > len(mask_array):
                        raise ValueError(
                            "Global mask is shorter than the expected pair sequence."
                        )
                    selected_pairs = mask_array[pair_slice]
                    first_rows = first_rows[selected_pairs]
                    second_rows = second_rows[selected_pairs]
                z_count = len(first_rows)
            else:
                first_rows = second_rows = np.empty(0, dtype=np.int64)
                z_count = 0
            n_pairs += pair_count

            if len(first_rows):
                values = [
                    data[field][rows]
                    for rows in (first_rows, second_rows)
                    for field in ("pt", "eta", "phi", "e")
                ]
                z_masses.extend(_compute_Z_mass(
                    values[0], values[1], values[2], values[3],
                    values[4], values[5], values[6], values[7],
                ))

        print(
            f"Processed file {file_index + 1}/{len(h5_files)}: "
            f"Found {z_count} selected Z masses ({n_pairs} candidate pairs processed)",
            end='\r',
            flush=True,
        )
    if mask_array is not None and n_pairs != len(mask_array):
        raise ValueError(
            f"Global mask length {len(mask_array)} does not match "
            f"the expected pair count {n_pairs}."
        )
    print(f"Saving Z masses to {csv_output_path}")
    np.savetxt(csv_output_path, z_masses)
    return 1


def get_data_length(h5_files_path : str | Path) -> int:
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