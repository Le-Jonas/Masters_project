import numpy as np
from numpy.ma import size
import torch
from torch.utils.data import  Sampler, DataLoader
from pathlib import Path
import time
import numpy.typing as npt
import xgboost as xgb

from . import dataset_classes as _class_
from . import misc_functions as _misc_
from . import neural_models as _neural_
from . import model_training as _train_


class ShuffledContiguousBatchSampler(Sampler):
    """
    A custom batch sampler that generates batches of contiguous indices and shuffles the starting indices of the batches.
    This is useful for datasets where you want to maintain some order within each batch but still want to shuffle the batches themselves.
    If the dataset has a 'mix_files' attribute set to True, it will create mixed batches from two sources based on the specified mixture ratio.

    Arguments:
    - dataset_size (int): The total number of samples in the dataset.
    - batch_size (int): The number of samples in each batch.
    - dataset (Dataset): The dataset to sample from.

    Keywords:
    - discard_remainder (bool): Whether to discard rows that cannot be included
      in a complete ratio-balanced batch. Defaults to False.
    """
    #Initializes the sampler with the dataset size and batch size. It will create batches of contiguous indices and shuffle them for each epoch.
    def __init__(
        self,
        dataset_size : int,
        batch_size : int,
        dataset : _class_.Dataset | None = None,
        *,
        discard_remainder : bool = False,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.dataset = dataset
        if not isinstance(discard_remainder, bool):
            raise TypeError("discard_remainder must be a boolean.")
        self.discard_remainder = discard_remainder

    def _size(self) -> int:
        """
        Returns the size of the dataset. If a dataset is provided, it returns the length of the dataset; otherwise, it returns the specified dataset size.

        Returns:
        - int: The size of the dataset.
        """
        return len(self.dataset) if self.dataset is not None else self.dataset_size

    def _offsets(self) -> list:
        """
        Finds the offsets for each file in the dataset. If the dataset has a 'pair_offsets' attribute, it returns that; otherwise, it returns the 'offsets' attribute.

        Returns:
        - list: A list of offsets for each file in the dataset.
        """
        if self.dataset is not None and hasattr(self.dataset, "pair_offsets"):
            return self.dataset.pair_offsets
        return self.dataset.offsets

    #This function generates batches of contiguous indices, shuffles the starting indices of the batches, and yields the indices for each batch.
    def __iter__(self):
        """
        Generates batches of contiguous indices, shuffles the starting indices of the batches, and yields the indices for each batch.
        If the dataset has a 'mix_files' attribute set to True, it will create mixed batches from two sources based on the specified mixture ratio.

        Yields:
        - list: A list of indices for each batch.
        """
        if self.dataset is not None and self.dataset.mix_files:
            yield from self._mixed_batches()
            return

        starts = np.arange(0, self._size(), self.batch_size)
        np.random.shuffle(starts)

        for start in starts:
            stop = min(start + self.batch_size, self._size())
            yield list(range(start, stop))

    def _mixed_batches(self):
        """
        Creates mixed batches from two sources based on the specified mixture ratio.

        Yields:
        - list: A list of mixed batches.
        """
        offsets = self._offsets()
        source_indices = [
            np.concatenate([
                np.arange(offsets[file_index], offsets[file_index + 1])
                for file_index, label in enumerate(self.dataset.file_labels)
                if label == source_label
            ])
            for source_label in (0, 1)
        ]
        source_0_batch_size = int(round(self.batch_size * self.dataset.mixture_ratio))
        source_0_batch_size = min(max(source_0_batch_size, 1), self.batch_size - 1)
        source_1_batch_size = self.batch_size - source_0_batch_size

        batches = []
        source_0_position = 0
        source_1_position = 0
        while (
            source_0_position + source_0_batch_size <= len(source_indices[0])
            and source_1_position + source_1_batch_size <= len(source_indices[1])
        ):
            source_0_batch = source_indices[0][source_0_position:source_0_position + source_0_batch_size]
            source_1_batch = source_indices[1][source_1_position:source_1_position + source_1_batch_size]
            batches.append(np.concatenate((source_0_batch, source_1_batch)).tolist())
            source_0_position += len(source_0_batch)
            source_1_position += len(source_1_batch)

        if not self.discard_remainder:
            remaining_indices = [
                source_indices[0][source_0_position:],
                source_indices[1][source_1_position:],
            ]
            non_empty_remaining = [
                indices for indices in remaining_indices if len(indices)
            ]
            if non_empty_remaining:
                remaining = np.concatenate(non_empty_remaining)
                batches.extend(
                    remaining[start:start + self.batch_size].tolist()
                    for start in range(0, len(remaining), self.batch_size)
                )
        np.random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        """
        Returns the number of batches in the dataset.
        If the dataset has a 'mix_files' attribute set to True, it calculates the number of batches based on the mixture ratio and the number of samples from each source.

        Returns:
        - int: The number of batches in the dataset.
        """
        if self.dataset is not None and self.dataset.mix_files:
            offsets = self._offsets()
            source_counts = [
                sum(
                    int(offsets[file_index + 1] - offsets[file_index])
                    for file_index, label in enumerate(self.dataset.file_labels)
                    if label == source_label
                )
                for source_label in (0, 1)
            ]
            source_0_batch_size = int(round(self.batch_size * self.dataset.mixture_ratio))
            source_0_batch_size = min(max(source_0_batch_size, 1), self.batch_size - 1)
            source_1_batch_size = self.batch_size - source_0_batch_size
            balanced_batches = min(
                source_counts[0] // source_0_batch_size,
                source_counts[1] // source_1_batch_size,
            )
            if self.discard_remainder:
                return balanced_batches
            remaining_count = (
                source_counts[0] - balanced_batches * source_0_batch_size
                + source_counts[1] - balanced_batches * source_1_batch_size
            )
            return balanced_batches + (
                remaining_count + self.batch_size - 1
            ) // self.batch_size
        size = self._size()
        return (size + self.batch_size - 1) // self.batch_size


class BatchIterator(xgb.DataIter):
    def __init__(self, loader, binary_target=False):
        super().__init__()
        self.loader = loader
        self.binary_target = binary_target
        self.iterator = None
        self.batch_index = 0
        self.n_features = None

    def reset(self):
        self.iterator = iter(self.loader)
        self.batch_index = 0
        self.n_features = None

    def next(self, input_data):
        try:
            batch = next(self.iterator)
        except StopIteration:
            return False

        particle, eventwise, target, domain = batch

        X = torch.cat((particle, eventwise), dim=1)

        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if self.binary_target:
            target = _train_._prepare_binary_targets(
                target,
                domain,
                13,
                domain="bool",
            )

        X_np = np.ascontiguousarray(
            X.detach().cpu().numpy(),
            dtype=np.float32,
        )
        target_np = np.ascontiguousarray(
            target.detach().cpu().numpy(),
            dtype=np.float32,
        ).reshape(-1)

        if X_np.ndim != 2:
            raise ValueError(
                f"Batch {self.batch_index}: X has shape {X_np.shape}"
            )

        if target_np.ndim != 1:
            raise ValueError(
                f"Batch {self.batch_index}: target has shape {target_np.shape}"
            )

        if X_np.shape[0] != target_np.shape[0]:
            raise ValueError(
                f"Batch {self.batch_index}: "
                f"{X_np.shape[0]} rows but {target_np.shape[0]} targets"
            )

        if self.n_features is None:
            self.n_features = X_np.shape[1]
        elif X_np.shape[1] != self.n_features:
            raise ValueError(
                f"Batch {self.batch_index}: expected {self.n_features} "
                f"features, got {X_np.shape[1]}"
            )

        #if not np.isfinite(X_np).all():
        #    bad = np.argwhere(~np.isfinite(X_np))[0]
        #    raise ValueError(
        #        f"Batch {self.batch_index}: non-finite feature at "
        #        f"row={bad[0]}, column={bad[1]}"
        #    )

        if not np.isfinite(target_np).all():
            raise ValueError(
                f"Batch {self.batch_index}: non-finite target"
            )

        input_data(data=X_np, label=target_np)

        self.batch_index += 1
        return True

def h5_to_csv(
    h5_file_path : str | Path, 
    csv_file_path : str | Path, 
    y_source : str, 
    y_field : str, 
    exclude_features : list[str] | None = None, 
    exclude_fields : dict[str, list[str]] | None = None,
    include_features : list[str] | None = None, 
    include_fields : dict[str, list[str]] | None = None,
    batch_size : int = 256, 
    sample_size : int = 100_000, 
    mix_h5_file_path : str | Path | None = None, 
    mixture_ratio : float | None = None, 
    mixture_seed : int = 0,
) -> int:
    """
    Converts H5 files to CSV format, normalizing the features based on the mean and standard deviation calculated from a sample of the dataset.

    Arguments:
    - h5_file_path (str or Path): Path to the input H5 file or directory containing H5 files.
    - csv_file_path (str or Path): Path to the output CSV file.
    - y_source (str): The source of the target variable in the H5 files.
    - y_field (str): The field name of the target variable in the H5 files

    Keywords:
    - exclude_features (list of str): List of feature names to exclude from the output. Default is None, which means no features are excluded.
    - exclude_fields (dict): Feature names mapped to field names to exclude. Default is None, which means no fields are excluded.
    - include_features (list of str): List of feature names to include in the output. If specified, only these features will be included. Default is None, which means all features are included.
    - include_fields (dict): Feature names mapped to field names to include. Features not listed retain all fields. Default is None, which means all fields are included.
    - batch_size (int): The number of samples to process in each batch. Default is 256.
    - sample_size (int): The number of samples to use for calculating mean and standard deviation for normalization. Default is 100,000.
    - mix_h5_file_path (str or Path): Path to a second set of H5 files to mix with the primary dataset. Default is None, which means no mixing is performed.
    - mixture_ratio (float): The ratio of samples to take from the second set of H5 files when mixing. Must be between 0 and 1. Default is None, which means no mixing is performed.
    - mixture_seed (int): Random seed for shuffling the mixed dataset. Default is 0.

    Returns:
    - int: Returns 1 upon successful completion.
    """
    print(f"Converting H5 files in {h5_file_path} to CSV at {csv_file_path}")
    h5_file_path = Path(h5_file_path)
    csv_file_path = Path(csv_file_path)

    h5_files = _misc_._h5_files_from_path(h5_file_path)
    mix_h5_files = _misc_._h5_files_from_path(mix_h5_file_path) if mix_h5_file_path is not None else None
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {h5_file_path}")
    if mix_h5_file_path is not None and not mix_h5_files:
        raise FileNotFoundError(f"No H5 files found in {mix_h5_file_path}")
    print(f"Creating dataset from {len(h5_files)} H5 files", end='\r')
    if mix_h5_files is not None:
        print(f"Mixing with {len(mix_h5_files)} H5 files at ratio {mixture_ratio}", end='\r')

    dataset = _class_.H5EgammaDataset(
        files=h5_files,
        mix_files=mix_h5_files,
        mixture_ratio=mixture_ratio,
        mixture_seed=mixture_seed,
        y_source=y_source,
        y_field=y_field,
        exclude_features=exclude_features,
        exclude_fields=exclude_fields,
        include_features=include_features,
        include_fields=include_fields
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Calculating mean and std for normalization using sample size {sample_size}", end='\r')
    means, stds = _misc_.compute_mean_std(dataset, sample_size=sample_size, batch_size=batch_size)

    with open(csv_file_path, 'w', newline='') as csv_file:
        # Write header
        target_name = "source" if mix_h5_files is not None else f"{y_source}_{y_field}"
        csv_file.write(','.join(dataset.fields + [target_name]) + '\n')

        # Write data rows
        writing_time_start = time.perf_counter()
        for batch, (features, target) in enumerate(loader):
            for feature_row, target_value in zip(features.numpy(), target.numpy()):
                if np.isnan(target_value):  # Check for NaN values in target
                    continue  # Skip rows with NaN target values
                feature_row = np.nan_to_num((feature_row-means)/stds, nan=0.0)  # Replace NaN values in features with 0.0
                row = np.concatenate((feature_row, [target_value]))
                csv_file.write(','.join(map(str, row)) + '\n')
            elapsed_time = time.perf_counter() - writing_time_start
            completion_percentage = (batch + 1) / len(loader)
            estimated_total_time = elapsed_time / completion_percentage
            estimated_time_remaining = estimated_total_time - elapsed_time
            print(f'Writing batch {batch+1}/{len(loader)}, estimated time to completion: {estimated_time_remaining:.0f} seconds', end='\r')

    dataset.close()
    return 1

def h5_to_npy(h5_file_path : str | Path, 
    npy_file_path : str | Path, 
    y_source : str, 
    y_field : str, 
    exclude_features : list[str] | None = None, 
    exclude_fields : dict[str, list[str]] | None = None,
    include_features : list[str] | None = None, 
    include_fields : dict[str, list[str]] | None = None,
    batch_size : int = 256, 
    sample_size : int = 100_000, 
    mix_h5_file_path : str | Path | None = None, 
    mixture_ratio : float | None = None, 
    mixture_seed : int = 0
) -> int:
    """
    Convert H5 files to a binary NumPy file.

    Arguments:
    - h5_file_path (str or Path): Path to the directory containing the H5 files.
    - npy_file_path (str or Path): Path to the output NumPy file.
    - y_source (str): The source of the target variable.
    - y_field (str): The field of the target variable.

    Keywords:
    - exclude_features (list of str): List of feature names to exclude from the output. Default is None.
    - exclude_fields (dict): Feature names mapped to field names to exclude. Default is None.
    - include_features (list of str): List of feature names to include in the output. If specified, only these features will be included. Default is None.
    - include_fields (dict): Feature names mapped to field names to include. Features not listed retain all fields. Default is None.
    - batch_size (int): The number of samples to process in each batch. Default is 256.
    - sample_size (int): The number of samples to use for calculating mean and standard deviation for normalization. Default is 100,000.
    - mix_h5_file_path (str or Path): Path to a second set of H5 files to mix with the primary dataset. Default is None.
    - mixture_ratio (float): The ratio of samples to take from the second set of H5 files when mixing. Must be between 0 and 1. Default is None.
    - mixture_seed (int): Random seed for shuffling the mixed dataset. Default is 0.

    Returns:
    - int: Returns 1 upon successful completion.
    """
    print(f"Converting H5 files in {h5_file_path} to binary NumPy data at {npy_file_path}")
    npy_file_path = Path(npy_file_path)

    h5_files = _misc_._h5_files_from_path(h5_file_path)
    mix_h5_files = _misc_._h5_files_from_path(mix_h5_file_path) if mix_h5_file_path is not None else None
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {h5_file_path}")
    if mix_h5_file_path is not None and not mix_h5_files:
        raise FileNotFoundError(f"No H5 files found in {mix_h5_file_path}")
    print(f"Creating dataset from {len(h5_files)} H5 files", end='\r')
    if mix_h5_files is not None:
        print(f"Mixing with {len(mix_h5_files)} H5 files at ratio {mixture_ratio}", end='\r')

    dataset = _class_.H5EgammaDataset(
        files=h5_files,
        mix_files=mix_h5_files,
        mixture_ratio=mixture_ratio,
        mixture_seed=mixture_seed,
        y_source=y_source,
        y_field=y_field,
        exclude_features=exclude_features,
        exclude_fields=exclude_fields,
        include_features=include_features,
        include_fields=include_fields,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Calculating mean and std for normalization using sample size {sample_size}", end='\r')
    means, stds = _misc_.compute_mean_std(
        dataset,
        sample_size=sample_size,
        batch_size=batch_size,
    )

    output = np.lib.format.open_memmap(
        npy_file_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), len(dataset.fields) + 1),
    )

    row_start = 0
    writing_time_start = time.perf_counter()
    for batching, (features, target) in enumerate(loader):
        features = torch.nan_to_num((features - means) / stds, nan=0.0)
        batch = torch.cat((features, target.unsqueeze(1)), dim=1).numpy()
        row_end = row_start + len(batch)
        output[row_start:row_end] = batch
        row_start = row_end

        elapsed_time = time.perf_counter() - writing_time_start
        completion_percentage = (batching + 1) / len(loader)
        estimated_total_time = elapsed_time / completion_percentage
        estimated_time_remaining = estimated_total_time - elapsed_time
        print(f'Writing batch {batching+1}/{len(loader)}, estimated time to completion: {estimated_time_remaining:.0f} seconds', end='\r')

    output.flush()
    del output
    dataset.close()
    return 1

def create_path_lists(paths : list[str], outputs : list[str] | None = None, seed : int = 0, mixing_list : list[tuple[int, int]] | None = None) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(seed=seed)

    path_num = len(paths)

    path_dir = {}
    for i, path in enumerate(paths):
        if outputs is None:
            name = path
        else:
            name = outputs[i]
        path_dir[name] = _misc_._h5_files_from_path(path)

    if mixing_list:
        for i, mix_tuple in enumerate(mixing_list):
            idx1, idx2 = mix_tuple[0], mix_tuple[1]
            if outputs is None:
                name1, name2 = paths[idx1], paths[idx2]
                comb_name = f"{name1}_{name2}"
            else:
                name1, name2 = outputs[idx1], outputs[idx2]
                comb_name = outputs[path_num + i]
            temp_list = path_dir[name1] + path_dir[name2]
            rng.shuffle(temp_list)
            path_dir[comb_name] = temp_list

    return path_dir

import numpy as np
from numpy.typing import ArrayLike


def electron_predictions_to_pair_predictions(
    electron_predictions: ArrayLike,
    electron_dataset: _class_.H5EgammaDataset,
    pair_dataset: _class_.ZPairDataset,
) -> np.ndarray:
    """Map [e_num] or [e_num, 2] electron predictions to [z_pair_num, 2] or [z_pair_num, 4] pair predictions."""

    predictions = np.asarray(electron_predictions)

    if predictions.ndim != 1 and predictions.ndim != 2:
        raise ValueError(
            f"electron_predictions must have shape [e_num] or [e_num, 2], got {predictions.shape}"
        )

    if len(predictions) != len(electron_dataset):
        raise ValueError(
            f"Expected {len(electron_dataset)} electron predictions, "
            f"got {len(predictions)}"
        )

    if list(electron_dataset.files) != list(pair_dataset.files):
        raise ValueError(
            "electron_dataset and pair_dataset must use the same files in the same order"
        )

    if predictions.ndim == 1:
        pair_predictions = np.empty((len(pair_dataset), 2), dtype=predictions.dtype)
    else:
        pair_predictions = np.empty((len(pair_dataset), 4), dtype=predictions.dtype)

    output_start = 0

    for file_index, pair_rows in enumerate(pair_dataset.pair_rows):
        pair_count = len(pair_rows)
        output_end = output_start + pair_count

        if pair_count == 0:
            continue

        # Raw HDF5 row indices for the two electrons in each pair.
        first_rows = pair_rows[:, 0]
        second_rows = pair_rows[:, 1]

        # Convert raw HDF5 rows to filtered local dataset positions.
        electron_rows = electron_dataset.valid_rows[file_index]

        first_local = np.searchsorted(electron_rows, first_rows)
        second_local = np.searchsorted(electron_rows, second_rows)

        if not np.all(electron_rows[first_local] == first_rows):
            raise ValueError("A pair contains an electron absent from electron_dataset")
        if not np.all(electron_rows[second_local] == second_rows):
            raise ValueError("A pair contains an electron absent from electron_dataset")

        # Convert local positions to global H5EgammaDataset positions.
        offset = electron_dataset.offsets[file_index]
        first_global = offset + first_local
        second_global = offset + second_local
        if predictions.ndim == 1:
            pair_predictions[output_start:output_end, 0] = predictions[first_global]
            pair_predictions[output_start:output_end, 1] = predictions[second_global]
        else:
            pair_predictions[output_start:output_end, 0] = predictions[first_global, 0]
            pair_predictions[output_start:output_end, 1] = predictions[first_global, 1]
            pair_predictions[output_start:output_end, 2] = predictions[second_global, 0]
            pair_predictions[output_start:output_end, 3] = predictions[second_global, 1]

        output_start = output_end

    return pair_predictions




def models_to_csv_files(models_path : Path | str, 
                        csv_path : Path | str, 
                        h5_egam_files : list[Path] | list[str], 
                        h5_data_files : list[Path] | list[str], 
                        batch_size : int = 4096, 
                        means : tuple | None = None, 
                        stds : tuple | None = None, 
                        valid_fields : dict[str, bool] | None = None, 
                        mixture_ratio : float | None = 0.66, 
                        mix_seed : int = 42, 
                        model_type : str = "egamma", 
                        model_configs : dict[str, tuple] = {},
                        max_rows : int | None = None,
                        num_workers : int = 4,
                        pin_memory : bool = False,
                        ) -> None:
    """
    Convert model predictions to CSV files.

    Arguments:
    - models_path (Path or str): Path to the directory containing the model files.
    - csv_path (Path or str): Path to the output CSV file.
    - h5_egam_files (list of Path or str): List of paths to the H5 Egamma files.
    - h5_data_files (list of Path or str): List of paths to the H5 data files.

    Keywords:
    - model_type (str): The type of the model. Default is "egammas".
    - batch_size (int): The number of samples to process in each batch. Default is 256.
    - mixture_ratio (float): The ratio of the mixture. Default is 0.66.
    - mix_seed (int): The seed for the random number generator. Default is 42.  
    """
    models_path = Path(models_path)
    csv_path = Path(csv_path)

    if (not means or not stds):
        means_dataset = _class_.H5EgammaDataset(
                        files=h5_egam_files,
                        mix_files=h5_data_files,
                        mixture_ratio=mixture_ratio,
                        mixture_seed=mix_seed,
                        exclude_features=["truth"],
                        exclude_fields = valid_fields,
                        valid_fields = valid_fields,
                        y_source = "egammas",
                        y_field = None,
                    )
        means, stds = _misc_.compute_mean_std(means_dataset, sample_size=1_000_000, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        means_dataset.close()

    if model_type == "egamma":
        mc_dataset = _class_.H5EgammaDataset(
            files=h5_egam_files,
            exclude_features=["truth"],
            exclude_fields = valid_fields,
            valid_fields = valid_fields,
            y_source = "egammas",
            y_field = None,
        )
        data_dataset = _class_.H5EgammaDataset(
            files=h5_data_files,
            exclude_features=["truth"],
            exclude_fields = valid_fields,
            valid_fields = valid_fields,
            y_source = "egammas",
            y_field = None,
        )

        pair_mc_dataset = _class_.ZPairDataset(
            files=h5_egam_files,
            exclude_features=["truth"],
            exclude_fields = valid_fields,
            valid_fields = valid_fields,
            y_source = "egammas",
            y_field = None
        )
        pair_data_dataset = _class_.ZPairDataset(
            files=h5_data_files,
            exclude_features=["truth"],
            exclude_fields = valid_fields,
            valid_fields = valid_fields,
            y_source = "egammas",
            y_field = None
        )

        particle_size = len([field for field in mc_dataset.fields if not field.startswith("eventwise_")])
        event_size = len([field for field in mc_dataset.fields if field.startswith("eventwise_")])

    elif model_type == "pair":
        mc_dataset = _class_.ZPairDataset(
            files=h5_egam_files,
            exclude_features=["truth"],
            exclude_fields = valid_fields,
            valid_fields = valid_fields,
            y_source = "egammas",
            y_field = ["z0", "d0", "sig_z0"],
        )
        
        data_dataset = _class_.ZPairDataset(
            files=h5_data_files,
            exclude_features=["truth"],
            exclude_fields = valid_fields,
            valid_fields = valid_fields,
            y_source = "egammas",
            y_field = ["z0", "d0", "sig_z0"],
        )

        particle_size = len([field for field in mc_dataset.fields if not field.startswith("eventwise_")]) // 2
        event_size = len([field for field in mc_dataset.fields if field.startswith("eventwise_")])
        target_size = len([field for field in mc_dataset.y_fields])

    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'egamma' or 'pair'.")

    for model_str in models_path.glob("*.pth"):
        name = model_str.stem
        name_parts = name.split("_")
        output_size = 1 #if name_parts[-1] in ["origin", "pt", "z"] else 2

        if name_parts[-1] not in ["logvar", "origin"]:
            continue

        if (name_parts[1] == "pair" and model_type == "egamma"):
            print(f"Skipping model {name} because it is a pair model and model_type is 'egamma'.")
            continue
        if (name_parts[1] != "pair" and model_type == "pair"):
            print(f"Skipping model {name} because it is an egammas model and model_type is 'pair'.")
            continue
        if (name_parts[-1] == "new" or name_parts[-1] == "old"):
            print(f"Skipping model {name} because it is a non-normal model.")
            continue

        csv_mc_file = csv_path / f"{name_parts[-1]}_{model_type}_predictions_mc.csv"
        csv_data_file = csv_path / f"{name_parts[-1]}_{model_type}_predictions_data.csv"

        print(f"Processing model: {model_str.name}")
        if model_type == "egamma":
            model = _neural_.ParticleOriginNetwork(
                particle_size,
                event_size,
                particle_hidden_sizes=[1024, 256, 256, 256, 256, 256, 64, 8],
                event_hidden_sizes=[64, 64, 16, 4],
                consolidation_hidden_sizes=[32, 32, 32, 8],
                output_size=output_size,
            )
        elif model_type == "pair" and name_parts[-1] != "origin":
            model = _neural_.ParticlePairNetwork(
                particle_size,
                event_size,
                particle_hidden_sizes=[1024, 256, 256, 256, 256, 256, 64, 8],
                event_hidden_sizes=[64, 64, 16, 4],
                consolidation_hidden_sizes=[32, 32, 32, 8],
                output_size=output_size,
            )
        else:
            model = _neural_.ParticlePairNetwork(
                target_size + 2,
                event_size,
                particle_hidden_sizes=[64, 64, 16, 16, 8],
                event_hidden_sizes=[16, 8],
                consolidation_hidden_sizes=[16, 8],
                output_size=output_size,
            )
        model.load_state_dict(torch.load(model_str, map_location=torch.device("cpu")))
        model.eval()

        try:
            config_tuple = model_configs[name_parts[-1]]
            log_target, norm_target, variance_target = config_tuple
        except KeyError:
            log_target, norm_target, variance_target = False, False, False

        if name_parts[-1] == "logvar":
            variance_target = True
            csv_mc_file = csv_path / f"{name_parts[-2]}_{name_parts[-1]}_{model_type}_predictions_mc.csv"
            csv_data_file = csv_path / f"{name_parts[-2]}_{name_parts[-1]}_{model_type}_predictions_data.csv"

        if (norm_target and name_parts[-1] != "z"):
            norm_dataset = _class_.H5EgammaDataset(
                files=h5_egam_files,
                mix_files=h5_data_files,
                mixture_ratio=mixture_ratio,
                mixture_seed=mix_seed,
                exclude_features=["truth"],
                exclude_fields = valid_fields,
                valid_fields = valid_fields,
                y_source = "truth",
                fallback_source = "egammas",
                y_field = [name_parts[-1]]
            )
            norm_target = _misc_.compute_target_mean_std(norm_dataset, sample_size=1_000_000, batch_size=batch_size, log_target=log_target, num_workers=num_workers, pin_memory=pin_memory)
        elif norm_target:
            norm_dataset = _class_.ZPairDataset(
                files=h5_egam_files,
                mix_files=h5_data_files,
                mixture_ratio=mixture_ratio,
                mixture_seed=mix_seed,
                exclude_features=["truth"],
                exclude_fields = valid_fields,
                valid_fields = valid_fields,
                y_source = "truth",
                fallback_source = "egammas",
                y_field = ["pt", "eta", "phi"]
            )
            norm_target = _misc_.compute_target_mean_std(norm_dataset, sample_size=500_000, batch_size=batch_size, log_target=log_target, z_target=True, num_workers=num_workers, pin_memory=pin_memory)

        if name_parts[-1] != "origin":
            mc_loader = DataLoader(mc_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

            mc_predictions = _train_.predict(model, mc_loader, means=means, stds=stds, log_target=log_target, norm_target=norm_target, variance_target=variance_target)
            data_predictions = _train_.predict(model, data_loader, means=means, stds=stds, log_target=log_target, norm_target=norm_target, variance_target=variance_target)
            np.savetxt(csv_mc_file, mc_predictions[:max_rows], delimiter=",")
            np.savetxt(csv_data_file, data_predictions[:max_rows], delimiter=",")

        else:
            norm_dataset = _class_.H5EgammaDataset(
                files=h5_egam_files,
                mix_files=h5_data_files,
                mixture_ratio=mixture_ratio,
                mixture_seed=mix_seed,
                exclude_features=["truth"],
                exclude_fields = valid_fields,
                valid_fields = valid_fields,
                y_source = "truth",
                fallback_source = "egammas",
                y_field = ["z0", "d0", "sig_z0"]
            )
            means_t, stds_t = _misc_.compute_target_mean_std(norm_dataset, sample_size=1_000_000, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

            mc_loader = DataLoader(mc_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            mc_targets = []
            data_targets = []
            eventwise_mc = []
            eventwise_data = []

            mc_length = len(mc_loader)
            data_length = len(data_loader)
            for batch, data in enumerate(mc_loader):
                _, _, eventwise, target, _ = data
                mc_targets.append(target.numpy())
                eventwise_mc.append(eventwise.numpy())
                print(200*" ", end='\r')  # Clear the line
                print(f"MC loading batch {batch+1}/{mc_length}", end='\r')
            for batch, data in enumerate(data_loader):
                _, _, eventwise, target, _ = data
                data_targets.append(target.numpy())
                eventwise_data.append(eventwise.numpy())
                print(200*" ", end='\r')  # Clear the line
                print(f"Data loading batch {batch+1}/{data_length}", end='\r')
            print(200*" ", end='\r')

            mc_targets = (np.concatenate(mc_targets) - means_t.detach().numpy()) / stds_t.detach().numpy()
            data_targets = (np.concatenate(data_targets) - means_t.detach().numpy()) / stds_t.detach().numpy()
            eventwise_mc = (np.concatenate(eventwise_mc) - means[1].detach().numpy()) / stds[1].detach().numpy()
            eventwise_data = (np.concatenate(eventwise_data) - means[1].detach().numpy()) / stds[1].detach().numpy()

            type_paired_predictions_mc = np.loadtxt("./csv_library/type_paired_predictions_mc.csv", delimiter=",") / 8.0
            isolation_paired_predictions_mc = np.loadtxt("./csv_library/isolation_paired_predictions_mc.csv", delimiter=",") / 8.0
            type_paired_predictions_data = np.loadtxt("./csv_library/type_paired_predictions_data.csv", delimiter=",") / 8.0
            isolation_paired_predictions_data = np.loadtxt("./csv_library/isolation_paired_predictions_data.csv", delimiter=",") / 8.0

            if mc_targets.shape[0] != type_paired_predictions_mc.shape[0]:
                mc_targets = mc_targets[:type_paired_predictions_mc.shape[0]]
                eventwise_mc = eventwise_mc[:type_paired_predictions_mc.shape[0]]
            if data_targets.shape[0] != type_paired_predictions_mc.shape[0]:
                data_targets = data_targets[:type_paired_predictions_mc.shape[0]]
                eventwise_data = eventwise_data[:type_paired_predictions_mc.shape[0]]

            mc_inputs_1 = np.column_stack((mc_targets[:, 0, :], type_paired_predictions_mc[:, 0], isolation_paired_predictions_mc[:, 0]))
            mc_inputs_2 = np.column_stack((mc_targets[:, 1, :], type_paired_predictions_mc[:, 1], isolation_paired_predictions_mc[:, 1]))
            data_inputs_1 = np.column_stack((data_targets[:, 0, :], type_paired_predictions_data[:, 0], isolation_paired_predictions_data[:, 0]))
            data_inputs_2 = np.column_stack((data_targets[:, 1, :], type_paired_predictions_data[:, 1], isolation_paired_predictions_data[:, 1]))
            
            mc_inputs = (torch.from_numpy(mc_inputs_1).float(), torch.from_numpy(mc_inputs_2).float())
            data_inputs = (torch.from_numpy(data_inputs_1).float(), torch.from_numpy(data_inputs_2).float())
            eventwise_mc = torch.from_numpy(eventwise_mc).float()
            eventwise_data = torch.from_numpy(eventwise_data).float()

            num_batches = int(np.ceil(mc_inputs[0].shape[0] / batch_size))

            origin_predictions_mc = []
            origin_predictions_data = []
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, mc_inputs[0].shape[0])
                mc_batch_inputs = (mc_inputs[0][start_idx:end_idx], mc_inputs[1][start_idx:end_idx])
                data_batch_inputs = (data_inputs[0][start_idx:end_idx], data_inputs[1][start_idx:end_idx])
                eventwise_mc_batch = eventwise_mc[start_idx:end_idx]
                eventwise_data_batch = eventwise_data[start_idx:end_idx]

                origin_predictions_mc_batch = model(*mc_batch_inputs, eventwise_mc_batch)
                origin_predictions_data_batch = model(*data_batch_inputs, eventwise_data_batch)
                origin_predictions_mc.append(origin_predictions_mc_batch.detach().numpy())
                origin_predictions_data.append(origin_predictions_data_batch.detach().numpy())
                print(200*" ", end='\r')  # Clear the line
                print(f"Origin model predictions batch {batch+1}/{num_batches}", end='\r')
            print(200*" ", end='\r')

            origin_predictions_mc = np.concatenate(origin_predictions_mc, axis=0)
            origin_predictions_data = np.concatenate(origin_predictions_data, axis=0)

            np.savetxt(csv_mc_file, origin_predictions_mc[:max_rows], delimiter=",")
            np.savetxt(csv_data_file, origin_predictions_data[:max_rows], delimiter=",")



        if model_type == "egamma":
            mc_pair_predictions = electron_predictions_to_pair_predictions(mc_predictions, mc_dataset, pair_mc_dataset)
            data_pair_predictions = electron_predictions_to_pair_predictions(data_predictions, data_dataset, pair_data_dataset)

            csv_mc_pair_file = csv_path / f"{name_parts[-1]}_paired_predictions_mc.csv"
            csv_data_pair_file = csv_path / f"{name_parts[-1]}_paired_predictions_data.csv"
            if name_parts[-1] == "logvar":
                csv_mc_pair_file = csv_path / f"{name_parts[-2]}_{name_parts[-1]}_paired_predictions_mc.csv"
                csv_data_pair_file = csv_path / f"{name_parts[-2]}_{name_parts[-1]}_paired_predictions_data.csv"

            np.savetxt(csv_mc_pair_file, mc_pair_predictions[:max_rows], delimiter=",")
            np.savetxt(csv_data_pair_file, data_pair_predictions[:max_rows], delimiter=",")

            pair_mc_dataset.close()
            pair_data_dataset.close()

    
def datasets_to_csv_files(
    h5_mc_files: list[Path] | list[str],
    h5_data_files: list[Path] | list[str],
    csv_path: Path | str,
    valid_fields: dict[str, bool] | None = None,
    batch_size: int = 4096,
    max_rows: int | None = None,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> None:
    """
    Convert datasets to CSV files.

    Arguments:
    - datasets (list of Dataset): List of datasets to convert.
    - csv_path (Path or str): Path to the output CSV file.

    Keywords:
    - batch_size (int): The number of samples to process in each batch. Default is 4096.
    """
    csv_path = Path(csv_path)
    mc_dataset = _class_.H5EgammaDataset(
        files=h5_mc_files,
        exclude_features=["truth", "cells", "tracks"],
        exclude_fields=valid_fields,
        valid_fields=valid_fields,
        y_source="truth",
        y_field=["pt", "eta", "phi"],
    )
    mc_type_dataset = _class_.H5EgammaDataset(
        files=h5_mc_files,
        exclude_features=["truth", "cells", "tracks"],
        exclude_fields=valid_fields,
        valid_fields=valid_fields,
        y_source="egammas",
        y_field=["truthType", "passid_Loose", "passiso_Loose_VarRad", "pt", "eta", "phi"],
    )
    data_dataset = _class_.H5EgammaDataset(
        files=h5_data_files,
        exclude_features=["truth", "cells", "tracks"],
        exclude_fields=valid_fields,
        valid_fields=valid_fields,
        y_source="egammas",
        y_field=["pt", "eta", "phi", "passid_Loose", "passiso_Loose_VarRad"],
    )
    mc_loader = DataLoader(_misc_._TargetOnlyDataset(mc_dataset), batch_size=batch_size*4, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    type_loader = DataLoader(_misc_._TargetOnlyDataset(mc_type_dataset), batch_size=batch_size*4, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    data_loader = DataLoader(_misc_._TargetOnlyDataset(data_dataset), batch_size=batch_size*4, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    mc_length = len(mc_loader)
    type_length = len(type_loader)
    data_length = len(data_loader)

    mc_variables = [[], [], [], [], [], [], [], [], []]
    data_variables = [[], [], [], [], []]
    
    for batch, target in enumerate(mc_loader):
        for i in range(3):
            mc_variables[i].append(target[:, i].numpy())
        print(200*" ", end='\r')  # Clear the line
        print(f"MC loading batch {batch+1}/{mc_length}", end='\r')

    for batch, target in enumerate(type_loader):
        for i in range(6):
            mc_variables[i+3].append(target[:, i].numpy())
        print(200*" ", end='\r')  # Clear the line
        print(f"MC type loading batch {batch+1}/{type_length}", end='\r')

    for i in range(len(mc_variables)):
        mc_variables[i] = np.concatenate(mc_variables[i])

    for batch, target in enumerate(data_loader):
        for i in range(5):
            data_variables[i].append(target[:, i].numpy())
        print(200*" ", end='\r')  # Clear the line
        print(f"Data loading batch {batch+1}/{data_length}", end='\r')
    print(200*" ", end='\r') 

    for i in range(len(data_variables)):
        data_variables[i] = np.concatenate(data_variables[i])

    name_list = ["pt", "eta", "phi", "type", "isolation"]

    for i in range(3):
        np.savetxt(csv_path / f"{name_list[i]}_egamma_truth_mc.csv", mc_variables[i][:max_rows], delimiter=",")
        np.savetxt(csv_path / f"{name_list[i]}_egamma_atlas_mc.csv", mc_variables[i+6][:max_rows], delimiter=",")
        np.savetxt(csv_path / f"{name_list[i]}_egamma_atlas_data.csv", data_variables[i][:max_rows], delimiter=",")

    np.savetxt(csv_path / f"{name_list[3]}_egamma_truth_mc.csv", mc_variables[3][:max_rows], delimiter=",")
    np.savetxt(csv_path / f"{name_list[3]}_egamma_atlas_mc.csv", mc_variables[4][:max_rows], delimiter=",")
    np.savetxt(csv_path / f"{name_list[4]}_egamma_atlas_mc.csv", mc_variables[5][:max_rows], delimiter=",")

    for i in range(2):
        np.savetxt(csv_path / f"{name_list[i+3]}_egamma_atlas_data.csv", data_variables[i+3][:max_rows], delimiter=",")

    mc_pair_dataset = _class_.ZPairDataset(
        files=h5_mc_files,
        exclude_features=["truth", "cells", "tracks"],
        exclude_fields=valid_fields,
        valid_fields=valid_fields,
        y_source="egammas",
        y_field=["truthOrigin"],
    )
    data_pair_dataset = _class_.ZPairDataset(
        files=h5_data_files,
        exclude_features=["truth", "cells", "tracks"],
        exclude_fields=valid_fields,
        valid_fields=valid_fields,
        y_source="egammas",
        y_field=["passid_Loose"],
    )

    mc_pair_loader = DataLoader(_misc_._TargetOnlyDataset(mc_pair_dataset), batch_size=batch_size*4, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    origin_truth_mc = []
    for target in mc_pair_loader:
        e1_origin = target[:, 0] == 13
        e2_origin = target[:, 1] == 13
        origin_truth_mc.append(e1_origin * e2_origin) 
        print(200*" ", end='\r')  # Clear the line
        print(f"Pair loading batch {batch+1}/{data_length}", end='\r')
    origin_truth_mc = np.concatenate(origin_truth_mc)
    print(200*" ", end='\r')

    mc_variables_pair = [[], [], [], [], [], [], [], [], []]
    data_variables_pair = [[], [], [], [], []]

    for i in range(len(mc_variables_pair)):
        mc_variables_pair[i] = electron_predictions_to_pair_predictions(mc_variables[i], mc_dataset, mc_pair_dataset)
    for i in range(len(data_variables_pair)):
        data_variables_pair[i] = electron_predictions_to_pair_predictions(data_variables[i], data_dataset, data_pair_dataset)

    z_mass_truth_mc = _misc_._compute_z_mass(mc_variables_pair[0][:, 0], mc_variables_pair[1][:, 0], mc_variables_pair[2][:, 0], mc_variables_pair[0][:, 1], mc_variables_pair[1][:, 1], mc_variables_pair[2][:, 1])
    z_mass_atlas_mc = _misc_._compute_z_mass(mc_variables_pair[6][:, 0], mc_variables_pair[7][:, 0], mc_variables_pair[8][:, 0], mc_variables_pair[6][:, 1], mc_variables_pair[7][:, 1], mc_variables_pair[8][:, 1])
    z_mass_atlas_data = _misc_._compute_z_mass(data_variables_pair[0][:, 0], data_variables_pair[1][:, 0], data_variables_pair[2][:, 0], data_variables_pair[0][:, 1], data_variables_pair[1][:, 1], data_variables_pair[2][:, 1])

    origin_atlas_mc = (abs(z_mass_atlas_mc/1000.0 - 91.2) < 100.0) & (mc_variables_pair[4][:, 0] == 1) & (mc_variables_pair[4][:, 1] == 1) #& ((mc_variables_pair[6][:, 0]/1000.0 > 45.0) | (mc_variables_pair[6][:, 1]/1000.0 > 45.0))
    origin_atlas_data = (abs(z_mass_atlas_data/1000.0 - 91.2) < 100.0) & (data_variables_pair[3][:, 0] == 1) & (data_variables_pair[3][:, 1] == 1) #& ((data_variables_pair[0][:, 0]/1000.0 > 45.0) | (data_variables_pair[0][:, 1]/1000.0 > 45.0))

    np.savetxt(csv_path / "origin_pair_truth_mc.csv", origin_truth_mc[:max_rows], delimiter=",")
    np.savetxt(csv_path / "origin_pair_atlas_mc.csv", origin_atlas_mc[:max_rows], delimiter=",")
    np.savetxt(csv_path / "origin_pair_atlas_data.csv", origin_atlas_data[:max_rows], delimiter=",")

    np.savetxt(csv_path / "z_pair_truth_mc.csv", z_mass_truth_mc[:max_rows], delimiter=",")
    np.savetxt(csv_path / "z_pair_atlas_mc.csv", z_mass_atlas_mc[:max_rows], delimiter=",")
    np.savetxt(csv_path / "z_pair_atlas_data.csv", z_mass_atlas_data[:max_rows], delimiter=",")

    for i in range(3):
        np.savetxt(csv_path / f"{name_list[i]}_pair_truth_mc.csv", mc_variables_pair[i][:max_rows], delimiter=",")
        np.savetxt(csv_path / f"{name_list[i]}_pair_atlas_mc.csv", mc_variables_pair[i+6][:max_rows], delimiter=",")
        np.savetxt(csv_path / f"{name_list[i]}_pair_atlas_data.csv", data_variables_pair[i][:max_rows], delimiter=",")

    np.savetxt(csv_path / f"{name_list[3]}_pair_truth_mc.csv", mc_variables_pair[3][:max_rows], delimiter=",")
    np.savetxt(csv_path / f"{name_list[3]}_pair_atlas_mc.csv", mc_variables_pair[4][:max_rows], delimiter=",")
    np.savetxt(csv_path / f"{name_list[4]}_pair_atlas_mc.csv", mc_variables_pair[5][:max_rows], delimiter=",")

    for i in range(2):
        np.savetxt(csv_path / f"{name_list[i+3]}_pair_atlas_data.csv", data_variables_pair[i+3][:max_rows], delimiter=",")