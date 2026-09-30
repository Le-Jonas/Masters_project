import numpy as np
import torch
from torch.utils.data import  Sampler, DataLoader
from pathlib import Path
from . import dataset_classes as _class_
from . import misc_functions as _misc_
import time

class ShuffledContiguousBatchSampler(Sampler):
    """
    A custom batch sampler that generates batches of contiguous indices and shuffles the starting indices of the batches.
    This is useful for datasets where you want to maintain some order within each batch but still want to shuffle the batches themselves.
    If the dataset has a 'mix_files' attribute set to True, it will create mixed batches from two sources based on the specified mixture ratio.

    Arguments:
    - dataset_size (int): The total number of samples in the dataset.
    - batch_size (int): The number of samples in each batch.
    - dataset (Dataset): The dataset to sample from.
    """
    #Initializes the sampler with the dataset size and batch size. It will create batches of contiguous indices and shuffle them for each epoch.
    def __init__(self, dataset_size, batch_size, dataset=None):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.dataset = dataset

    def _size(self):
        """
        Returns the size of the dataset. If a dataset is provided, it returns the length of the dataset; otherwise, it returns the specified dataset size.

        Returns:
        - int: The size of the dataset.
        """
        return len(self.dataset) if self.dataset is not None else self.dataset_size

    def _offsets(self):
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
        while source_0_position < len(source_indices[0]) or source_1_position < len(source_indices[1]):
            source_0_batch = source_indices[0][source_0_position:source_0_position + source_0_batch_size]
            source_1_batch = source_indices[1][source_1_position:source_1_position + source_1_batch_size]
            if len(source_0_batch) == 0 and len(source_1_batch) == 0:
                break
            batches.append(np.concatenate((source_0_batch, source_1_batch)).tolist())
            source_0_position += len(source_0_batch)
            source_1_position += len(source_1_batch)
        np.random.shuffle(batches)
        yield from batches

    def __len__(self):
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
            return max(
                (source_counts[0] + source_0_batch_size - 1) // source_0_batch_size,
                (source_counts[1] + source_1_batch_size - 1) // source_1_batch_size,
            )
        size = self._size()
        return (size + self.batch_size - 1) // self.batch_size

def h5_to_csv(h5_file_path, csv_file_path, y_source, y_field, exclude_features=None, exclude_fields=None, include_features=None, include_fields=None, batch_size=256, sample_size=100_000, mix_h5_file_path=None, mixture_ratio=None, mixture_seed=0):
    """
    Converts H5 files to CSV format, normalizing the features based on the mean and standard deviation calculated from a sample of the dataset.

    Arguments:
    - h5_file_path (str or Path): Path to the input H5 file or directory containing H5 files.
    - csv_file_path (str or Path): Path to the output CSV file.
    - y_source (str): The source of the target variable in the H5 files.
    - y_field (str): The field name of the target variable in the H5 files

    Keywords:
    - exclude_features (list of str): List of feature names to exclude from the output. Default is None, which means no features are excluded.
    - exclude_fields (list of str): List of field names to exclude from the output. Default is None, which means no fields are excluded.
    - include_features (list of str): List of feature names to include in the output. If specified, only these features will be included. Default is None, which means all features are included.
    - include_fields (list of str): List of field names to include in the output. If specified, only these fields will be included. Default is None, which means all fields are included.
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

    def h5_files_from_path(path):
        path = Path(path)
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(file for file in path.iterdir() if file.is_file() and file.suffix == ".h5")
        raise FileNotFoundError(f"H5 path does not exist: {path}")

    h5_files = h5_files_from_path(h5_file_path)
    mix_h5_files = h5_files_from_path(mix_h5_file_path) if mix_h5_file_path is not None else None
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {h5_file_path}")
    if mix_h5_file_path is not None and not mix_h5_files:
        raise FileNotFoundError(f"No H5 files found in {mix_h5_file_path}")
    print(f"Creating dataset from {len(h5_files)} H5 files", end='\r')
    if mix_h5_files is not None:
        print(f"Mixing with {len(mix_h5_files)} H5 files at ratio {mixture_ratio}", end='\r')

    dataset = _class_.H5EgammaDataset_fully_batched(
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

def h5_to_npy(h5_file_path, npy_file_path, y_source, y_field, exclude_features=None, exclude_fields=None, include_features=None, include_fields=None, batch_size=256, sample_size=100_000, mix_h5_file_path=None, mixture_ratio=None, mixture_seed=0):
    """
    Convert H5 files to a binary NumPy file.

    Arguments:
    - h5_file_path (str or Path): Path to the directory containing the H5 files.
    - npy_file_path (str or Path): Path to the output NumPy file.
    - y_source (str): The source of the target variable.
    - y_field (str): The field of the target variable.

    Keywords:
    - exclude_features (list of str): List of feature names to exclude from the output. Default is None.
    - exclude_fields (list of str): List of field names to exclude from the output. Default is None.
    - include_features (list of str): List of feature names to include in the output. If specified, only these features will be included. Default is None.
    - include_fields (list of str): List of field names to include in the output. If specified, only these fields will be included. Default is None.
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

    h5_files = _misc_.h5_files_from_path(h5_file_path)
    mix_h5_files = _misc_.h5_files_from_path(mix_h5_file_path) if mix_h5_file_path is not None else None
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {h5_file_path}")
    if mix_h5_file_path is not None and not mix_h5_files:
        raise FileNotFoundError(f"No H5 files found in {mix_h5_file_path}")
    print(f"Creating dataset from {len(h5_files)} H5 files", end='\r')
    if mix_h5_files is not None:
        print(f"Mixing with {len(mix_h5_files)} H5 files at ratio {mixture_ratio}", end='\r')

    dataset = _class_.H5EgammaDataset_fully_batched(
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