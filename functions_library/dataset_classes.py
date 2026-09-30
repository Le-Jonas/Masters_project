import h5py
import numpy as np
from pathlib import Path
#from sqlalchemy import values
from torch.utils.data import Dataset 
import torch
import time

class H5EgammaDataset(Dataset):
    #Initialization of the dataset class. Needs input for the dataset file paths, the source of the target variable, 
    #the field name of the target variable, and any fields to exclude from the features.
    def __init__(
        self,
        files,
        y_source="truth",
        y_field="pt",
        exclude_fields=None,
        exclude_features=None,
        include_fields=None,
        include_features=None,
        mix_files=None,
        mixture_ratio=None,
        mixture_seed=0,
    ):        
        if include_fields is not None and exclude_fields is not None:
            raise ValueError("Cannot specify both include_fields and exclude_fields.")
        if include_features is not None and exclude_features is not None:
            raise ValueError("Cannot specify both include_features and exclude_features.")
        
        def as_paths(paths):
            if isinstance(paths, (str, Path)):
                return [Path(paths)]
            return [Path(path) for path in paths]

        self.files = as_paths(files)
        self.mix_files = as_paths(mix_files) if mix_files is not None else []
        if self.mix_files and mixture_ratio is None:
            raise ValueError("mixture_ratio is required when mix_files is provided.")
        if not self.mix_files and mixture_ratio is not None:
            raise ValueError("mix_files is required when mixture_ratio is provided.")
        if mixture_ratio is not None and not 0 < mixture_ratio < 1:
            raise ValueError("mixture_ratio must be between 0 and 1.")
        self.mixture_ratio = mixture_ratio
        self.file_labels = [0] * len(self.files) + [1] * len(self.mix_files)
        self.files.extend(self.mix_files)
        self.y_source = y_source
        self.y_fields = [] if y_field is None else [y_field] if isinstance(y_field, str) else list(y_field)
        self.exclude_fields = set(exclude_fields or [])
        self.include_fields = set(include_fields) if include_fields is not None else None
        self.valid_rows = []
        self.handles = {}
        self.lengths = []
        self.fields = []



        example_file = self.files[0]
        with h5py.File(example_file, "r") as h5_file:
            feature_names = list(h5_file.keys())
            feature_names.remove("eventwise") if "eventwise" in h5_file else None
            if self.y_fields and self.y_source not in feature_names:
                raise KeyError(f"Unknown y source: {self.y_source}")
            feature_names.remove("truth") if "truth" in h5_file else None

            if include_features is not None:
                feature_names = [name for name in feature_names if name in include_features]

            self.features = [name for name in feature_names if name not in exclude_features] if exclude_features != None else feature_names

            for feature in self.features:
                field_names = self._selected_field_names(feature, h5_file[feature].dtype.names)
                
                shape = h5_file[feature][0].shape
                if len(shape) > 1:
                    raise ValueError(f"Feature {feature} has shape {shape}, expected 1D-like structured array")
                elif len(shape) == 0:
                    self.fields.extend([f"{feature}_{name}" for name in field_names])
                else:
                    for i in range(shape[0]):
                        self.fields.extend([f"{feature}_{name}_{i}" for name in field_names])
                    
            field_names = self._selected_field_names(
                "eventwise", h5_file["eventwise"].dtype.names
            )
            self.fields.extend([f"eventwise_{name}" for name in field_names])

        #Finds the finite-target rows and creates offsets over the filtered dataset.
        for path in self.files:
            with h5py.File(path, "r") as h5_file:
                lengths = {
                    name: len(h5_file[name])
                    for name in feature_names
                }
                if len(set(lengths.values())) != 1:
                    raise ValueError(f"Event data not aligned in {path}")
                if self.y_fields:
                    target = self._read_target(h5_file, slice(None))
                    valid_rows = np.flatnonzero(np.isfinite(target).all(axis=-1))
                else:
                    valid_rows = np.arange(next(iter(lengths.values())), dtype=np.int64)
                self.lengths.append(len(valid_rows))
                self.valid_rows.append(valid_rows)

        if self.mix_files:
            self._apply_mixture(mixture_ratio, mixture_seed)

        self.offsets = np.concatenate(([0], np.cumsum(self.lengths)))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handles"] = {}
        state["eventwise_cache"] = {}
        state["eventwise_features_cache"] = {}
        state["eventwise_first_indices_cache"] = {}
        state["eventwise_metadata_cache"] = {}
        return state

    def _get_handle(self, file_index):
        if not 0 <= file_index < len(self.files):
            raise IndexError(
                f"Dataset file index {file_index} is out of range for "
                f"{len(self.files)} files; offsets={self.offsets.tolist()}"
            )
        if file_index not in self.handles:
            self.handles[file_index] = h5py.File(self.files[file_index], "r")
        return self.handles[file_index]

    def _global_file_index(self, index, offsets=None):
        offsets = self.offsets if offsets is None else offsets
        index = int(index)
        if index < 0 or index >= int(offsets[-1]):
            raise IndexError(
                f"Dataset index {index} is out of range for length {int(offsets[-1])}"
            )
        return int(np.searchsorted(offsets, index, side="right") - 1)

    def _apply_mixture(self, mixture_ratio, mixture_seed):
        source_rows = [
            np.concatenate([
                self.valid_rows[file_index]
                for file_index, label in enumerate(self.file_labels)
                if label == source_label
            ])
            for source_label in (0, 1)
        ]
        source_file_indices = [
            np.concatenate([
                np.full(len(self.valid_rows[file_index]), file_index, dtype=np.int64)
                for file_index, label in enumerate(self.file_labels)
                if label == source_label
            ])
            for source_label in (0, 1)
        ]

        source_counts = [len(rows) for rows in source_rows]
        total_count = int(min(
            source_counts[0] / mixture_ratio,
            source_counts[1] / (1 - mixture_ratio),
        ))
        if total_count == 0:
            raise ValueError("The requested mixture has no valid rows in one of the sources.")

        source_0_count = int(round(total_count * mixture_ratio))
        source_1_count = total_count - source_0_count
        rng = np.random.default_rng(mixture_seed)
        selected = []
        for rows, file_indices, count in zip(source_rows, source_file_indices, (source_0_count, source_1_count)):
            selected_indices = rng.choice(len(rows), size=count, replace=False)
            selected.extend(zip(file_indices[selected_indices], rows[selected_indices]))

        selected_by_file = {file_index: [] for file_index in range(len(self.files))}
        for file_index, local_index in selected:
            selected_by_file[int(file_index)].append(int(local_index))
        self.valid_rows = [
            np.sort(np.asarray(selected_by_file[file_index], dtype=np.int64))
            for file_index in range(len(self.files))
        ]
        self.lengths = [len(rows) for rows in self.valid_rows]

    #Finds the total length of the dataset by summing the lengths of all files. (Cumsum has already been calculated)
    def __len__(self):
        return int(self.offsets[-1])

    #Finds the file and row index corresponding to a given global index across all files.
    def _get_file_and_row(self, index):
        file_index = self._global_file_index(index)
        filtered_index = int(index - self.offsets[file_index])
        local_index = int(self.valid_rows[file_index][filtered_index])
        return file_index, self._get_handle(file_index), local_index

    def _selected_field_names(self, dataset_name, field_names):
        excluded = set(self.exclude_fields)
        if dataset_name == self.y_source:
            excluded.update(self.y_fields)

        if self.include_fields is not None:
            return [name for name in field_names if name in self.include_fields and name not in excluded]
        return [name for name in field_names if name not in excluded]

    def _read_target(self, h5_file, indices):
        values = [
            np.asarray(h5_file[self.y_source][field][indices], dtype=np.float32)
            for field in self.y_fields
        ]
        return np.stack(values, axis=-1)

    #Converts a structured row to a flat array using the same field selection as the header.
    def _structured_to_array(self, rows, dataset_name):
        names = self._selected_field_names(dataset_name, rows.dtype.names)
        values = [np.asarray(rows[name], dtype=np.float32).reshape(-1) for name in names]
        return np.concatenate(values) if values else np.empty(0, dtype=np.float32)

    def _eventwise_feature_matrix(self, eventwise):
        names = self._selected_field_names("eventwise", eventwise.dtype.names)
        if not names:
            return np.empty((len(eventwise), 0), dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(eventwise[name], dtype=np.float32).reshape(len(eventwise), -1)
                for name in names
            ],
            axis=1,
        )

    def _eventwise_first_indices(self, eventwise, row_count=None):
        counts = self._eventwise_counts(eventwise)
        if "firstEgammaIndex" in eventwise.dtype.names:
            field = "firstEgammaIndex"
        elif "firstMuonIndex" in eventwise.dtype.names:
            field = "firstMuonIndex"
        elif "firstTauIndex" in eventwise.dtype.names:
            field = "firstTauIndex"
        else:
            field = None

        if field is not None:
            starts = np.asarray(eventwise[field], dtype=np.int64)
            ends = starts + counts
            if row_count is None:
                return starts
            if starts.size and starts[0] != 0:
                shifted_starts = starts - starts[0]
                shifted_ends = shifted_starts + counts
                if shifted_ends[-1] == row_count and np.all(
                    shifted_starts[1:] == shifted_ends[:-1]
                ):
                    return shifted_starts
            elif starts.size and ends[-1] == row_count and np.all(
                starts[1:] == ends[:-1]
            ):
                return starts

        # Some files store absolute or otherwise non-local first indices. The
        # feature datasets are packed locally, so cumulative counts are the
        # correct row mapping when explicit indices fail validation.
        return np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))

    def _get_eventwise_first_indices(self, file_index, h5_file, eventwise):
        if not hasattr(self, "eventwise_first_indices_cache"):
            self.eventwise_first_indices_cache = {}
        if file_index not in self.eventwise_first_indices_cache:
            row_count = len(h5_file[self.features[0]])
            self.eventwise_first_indices_cache[file_index] = (
                self._eventwise_first_indices(eventwise, row_count)
            )
        return self.eventwise_first_indices_cache[file_index]

    def _get_eventwise_metadata(self, file_index, h5_file):
        if not hasattr(self, "eventwise_metadata_cache"):
            self.eventwise_metadata_cache = {}
        if file_index not in self.eventwise_metadata_cache:
            eventwise_dataset = h5_file["eventwise"]
            count_field = (
                "nEgammas" if "nEgammas" in eventwise_dataset.dtype.names
                else "nMuons" if "nMuons" in eventwise_dataset.dtype.names
                else "nTaus" if "nTaus" in eventwise_dataset.dtype.names
                else eventwise_dataset.dtype.names[0]
            )
            counts = np.asarray(eventwise_dataset[count_field][:], dtype=np.int64)
            if "firstEgammaIndex" in eventwise_dataset.dtype.names:
                starts = np.asarray(eventwise_dataset["firstEgammaIndex"][:], dtype=np.int64)
            elif "firstMuonIndex" in eventwise_dataset.dtype.names:
                starts = np.asarray(eventwise_dataset["firstMuonIndex"][:], dtype=np.int64)
            elif "firstTauIndex" in eventwise_dataset.dtype.names:
                starts = np.asarray(eventwise_dataset["firstTauIndex"][:], dtype=np.int64)
            else:
                starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
            row_count = len(h5_file[self.features[0]])
            if starts.size and starts[0] != 0:
                shifted = starts - starts[0]
                if shifted[-1] + counts[-1] == row_count and np.all(
                    shifted[1:] == shifted[:-1] + counts[:-1]
                ):
                    starts = shifted
            if not starts.size or starts[0] != 0 or starts[-1] + counts[-1] != row_count:
                starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
            self.eventwise_metadata_cache[file_index] = (
                starts,
                counts,
                count_field,
            )
        return self.eventwise_metadata_cache[file_index]

    def _read_eventwise_rows(self, h5_file, event_indices):
        event_indices = np.asarray(event_indices, dtype=np.int64)
        eventwise_dataset = h5_file["eventwise"]
        return eventwise_dataset[event_indices]

    def _eventwise_counts(self, eventwise):
        count_field = (
            "nEgammas" if "nEgammas" in eventwise.dtype.names
            else "nMuons" if "nMuons" in eventwise.dtype.names
            else "nTaus" if "nTaus" in eventwise.dtype.names
            else eventwise.dtype.names[0]
        )
        return np.asarray(eventwise[count_field], dtype=np.int64)

    #Tries to read aligned rows in few contiguous slices if possible, otherwise reads them individually. Returns the rows and the sort order of the indices.
    def _read_aligned_rows(
        self,
        h5_file,
        indices,
        file_index=None,
        include_target=True,
        prefer_contiguous_span=False,
    ):
        indices = np.asarray(indices, dtype=np.int64)
        sort_order = np.argsort(indices)
        sorted_indices = indices[sort_order]

        if prefer_contiguous_span and len(sorted_indices):
            first = int(sorted_indices[0])
            last = int(sorted_indices[-1]) + 1
            relative_indices = sorted_indices - first
            span_size = last - first
            if span_size <= max(len(sorted_indices) * 4, len(sorted_indices) + 32):
                rows = {
                    name: h5_file[name][first:last][relative_indices]
                    for name in self.features
                }
                if not include_target or not self.y_fields:
                    target = np.empty(
                        (len(sorted_indices), 0), dtype=np.float32
                    )
                elif self.y_source == "eventwise":
                    target = [None] * len(sorted_indices)
                else:
                    target = np.stack(
                        [
                            np.asarray(
                                h5_file[self.y_source][field][sorted_indices],
                                dtype=np.float32,
                            )
                            for field in self.y_fields
                        ],
                        axis=-1,
                    )
                return rows, target, sort_order

        runs = np.flatnonzero(np.diff(sorted_indices) > 1) + 1
        run_starts = np.r_[0, runs]
        run_ends = np.r_[runs, len(sorted_indices)]
        use_slices = len(run_starts) <= max(1, len(sorted_indices) // 2)

        if not include_target or self.mix_files:
            if not include_target or not self.y_fields:
                target = np.empty((len(sorted_indices), 0), dtype=np.float32)
            elif self.y_source == "eventwise":
                target = [None] * len(sorted_indices)
            else:
                target = np.stack(
                    [
                        np.asarray(
                            h5_file[self.y_source][field][sorted_indices],
                            dtype=np.float32,
                        )
                        for field in self.y_fields
                    ],
                    axis=-1,
                )
            return (
                {name: h5_file[name][sorted_indices] for name in self.features},
                target,
                sort_order,
            )

        #Reverts to reading individual rows if the number of contiguous slices is too high, otherwise reads them in slices.
        if not use_slices:
            if not include_target or not self.y_fields:
                target = np.empty((len(sorted_indices), 0), dtype=np.float32)
            elif self.y_source == "eventwise":
                target = [None] * len(sorted_indices)
            else:
                target = target = np.stack(
                    [
                        np.asarray(h5_file[self.y_source][field][sorted_indices], dtype=np.float32)
                        for field in self.y_fields
                    ],
                    axis=-1,
                )
            return {name: h5_file[name][sorted_indices] for name in self.features}, target, sort_order

        #This part takes care of the contigous reads. Reading a batch of 32 rows in one go reduces computation from 8s to 0.2s
        rows = {name: [] for name in self.features}
        target = []
        for start, end in zip(run_starts, run_ends):
            first = int(sorted_indices[start])
            last = int(sorted_indices[end - 1]) + 1
            relative_indices = sorted_indices[start:end] - first
            contiguous = np.array_equal(
                relative_indices,
                np.arange(len(relative_indices), dtype=np.int64),
            )
            for name in self.features:
                if contiguous:
                    rows[name].append(h5_file[name][first:last])
                else:
                    rows[name].append(h5_file[name][first:last][relative_indices])
            if include_target and self.y_fields and self.y_source != "eventwise":
                if contiguous:
                    target.append(
                        np.stack(
                            [
                                np.asarray(h5_file[self.y_source][field][first:last], dtype=np.float32)
                                for field in self.y_fields
                            ],
                            axis=-1,
                        )
                    )
                else:
                    target.append(
                        np.stack(
                            [
                                np.asarray(h5_file[self.y_source][field][first:last], dtype=np.float32)
                                for field in self.y_fields
                            ],
                            axis=-1,
                        )
                    )
        if not include_target or not self.y_fields:
            return {name: np.concatenate(rows[name]) for name in self.features}, np.empty((len(sorted_indices), 0), dtype=np.float32), sort_order
        if self.y_source == "eventwise":
            return {name: np.concatenate(rows[name]) for name in self.features}, [None]*len(sorted_indices), sort_order
        return {name: np.concatenate(rows[name]) for name in self.features}, np.concatenate(target), sort_order

    #This function groups a single data row into a torch tensor of features and a torch tensor of the target variable. It can also read the h5 file if no data is set yet.
    def _read_one(self, h5_file, local_index, eventwise=None, first_indices=None, rows=None, target=None, eventwise_features=None):
        
        if eventwise is None:
            eventwise = h5_file["eventwise"][:]
        eventwise_names = eventwise.dtype.names
        if first_indices is None:
            first_indices = self._eventwise_first_indices(
                eventwise, len(h5_file[self.features[0]])
            )

        event_index = int(np.searchsorted(first_indices, local_index, side="right") - 1)
        if event_index < 0:
            raise IndexError(f"Egamma row {local_index} is not covered by eventwise")
        event_end = first_indices[event_index] + self._eventwise_counts(eventwise)[event_index]
        if local_index >= event_end:
            raise IndexError(f"Egamma row {local_index} is not covered by eventwise")

        if rows is None:
            rows = {name: h5_file[name][local_index] for name in self.features}

        feature_arrays = [
            self._structured_to_array(rows[name], name)
            for name in self.features
        ]

        if eventwise_features is None:
            eventwise_features = self._structured_to_array(eventwise[event_index], "eventwise")
        elif isinstance(eventwise_features, dict):
            eventwise_features = eventwise_features.setdefault(
                event_index,
                self._structured_to_array(eventwise[event_index], "eventwise"),
            )
        else:
            eventwise_features = eventwise_features[event_index]

        features = np.concatenate(feature_arrays)
        
               
        if self.y_fields and self.y_source not in h5_file:
            raise KeyError(f"Unknown y source: {self.y_source}")
        target_index = event_index if self.y_source == "eventwise" else local_index
        if target is None and self.y_fields:
            target = np.asarray(
                [h5_file[self.y_source][field][target_index] for field in self.y_fields],
                dtype=np.float32,
            )
        elif target is None:
            target = np.empty(0, dtype=np.float32)
        return torch.from_numpy(features), torch.from_numpy(eventwise_features), torch.tensor(target)
    
    #Depreciated single row read function. Torch dataloader will call __getitems__ instead of this function.
    def __getitem__(self, index):
        file_index, h5_file, local_index = self._get_file_and_row(index)
        return self._read_one(h5_file, local_index)

    #Main read function for the fully batched dataset. It groups the indices by file, reads the aligned rows in batches, and returns a list of feature-target pairs.
    def __getitems__(self, indices, include_target=True):
        indices = list(indices)
        grouped = {}
        for output_position, index in enumerate(indices):
            file_index = self._global_file_index(index)
            filtered_index = int(index - self.offsets[file_index])
            local_index = int(self.valid_rows[file_index][filtered_index])
            grouped.setdefault(file_index, []).append((output_position, local_index))

        if not hasattr(self, "eventwise_cache"):
            self.eventwise_cache = {}
        if not hasattr(self, "eventwise_features_cache"):
            self.eventwise_features_cache = {}

        batch = [None] * len(indices)
        for file_index, positions in grouped.items():
            h5_file = self._get_handle(file_index)
            first_indices, counts, count_field = self._get_eventwise_metadata(
                file_index, h5_file
            )
            eventwise_dtype = [(count_field, "i8")]
            start_field = (
                "firstEgammaIndex" if "firstEgammaIndex" in h5_file["eventwise"].dtype.names
                else "firstMuonIndex" if "firstMuonIndex" in h5_file["eventwise"].dtype.names
                else "firstTauIndex" if "firstTauIndex" in h5_file["eventwise"].dtype.names
                else None
            )
            if start_field is not None:
                eventwise_dtype.append((start_field, "i8"))
            eventwise = np.empty(len(counts), dtype=eventwise_dtype)
            eventwise[count_field] = counts
            if start_field is not None:
                eventwise[start_field] = first_indices
            local_indices = np.array([local_index for _, local_index in positions])
            rows, target, sort_order = self._read_aligned_rows(
                h5_file,
                local_indices,
                file_index,
                include_target=include_target,
            )
            sorted_indices = np.sort(local_indices)
            event_indices = np.searchsorted(first_indices, sorted_indices, side="right") - 1
            unique_event_indices = np.unique(event_indices)
            eventwise_rows = self._read_eventwise_rows(h5_file, unique_event_indices)
            eventwise_features = {
                int(event_index): self._structured_to_array(
                    eventwise_rows[position], "eventwise"
                )
                for position, event_index in enumerate(unique_event_indices)
            }

            for sorted_position, original_position in enumerate(np.argsort(sort_order)):
                batch[positions[original_position][0]] = self._read_one(
                    h5_file,
                    int(sorted_indices[sorted_position]),
                    eventwise,
                    first_indices,
                    {name: rows[name][sorted_position] for name in rows},
                    target[sorted_position],
                    eventwise_features,
                )

        return batch
    
    #Closes all open file handles and clears the handles dictionary to free up resources.
    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

class npyDataset(Dataset):
    def __init__(self, npy_file_path):
        self.data = np.load(npy_file_path)
        self.features = self.data[:, :-1]
        self.targets = self.data[:, -1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.targets[idx]
        return torch.from_numpy(features).float(), torch.tensor(target, dtype=torch.float32)
    
    def __getitems__(self, indices):
        features = torch.from_numpy(self.features[indices])
        targets = torch.from_numpy(self.targets[indices])
        return list(zip(features, targets))

class ZPairDataset(H5EgammaDataset):
    """Dataset containing every unordered lepton pair from each event.

    A returned item is ``(lepton1, lepton2, eventwise, target)``. Events
    with fewer than two selected leptons are omitted.
    """

    def __init__(self, files, **kwargs):
        super().__init__(files=files, **kwargs)

        self.pair_file_indices = []
        self.pair_rows = []
        self.pair_event_indices = []
        self.pair_offsets = [0]

        for file_index, valid_rows in enumerate(self.valid_rows):
            with h5py.File(self.files[file_index], "r") as h5_file:
                eventwise = h5_file["eventwise"][:]
                row_count = len(h5_file[self.features[0]])
            event_starts = self._eventwise_first_indices(
                eventwise, row_count
            )
            event_ends = event_starts + self._eventwise_counts(eventwise)

            # valid_rows is sorted, so locate each event's selected rows with
            # two binary searches instead of scanning the full array per event.
            event_row_starts = np.searchsorted(valid_rows, event_starts, side="left")
            event_row_ends = np.searchsorted(valid_rows, event_ends, side="left")
            selected_counts = event_row_ends - event_row_starts
            pair_counts = selected_counts * (selected_counts - 1) // 2
            pair_array = np.empty((int(pair_counts.sum()), 2), dtype=np.int64)
            event_array = np.empty(len(pair_array), dtype=np.int64)

            pair_start = 0
            for event_index, (row_start, row_end, pair_count) in enumerate(
                zip(event_row_starts, event_row_ends, pair_counts)
            ):
                if pair_count == 0:
                    continue
                event_rows = valid_rows[row_start:row_end]
                first_indices, second_indices = np.triu_indices(len(event_rows), k=1)
                pair_end = pair_start + int(pair_count)
                pair_array[pair_start:pair_end, 0] = event_rows[first_indices]
                pair_array[pair_start:pair_end, 1] = event_rows[second_indices]
                event_array[pair_start:pair_end] = event_index
                pair_start = pair_end

            self.pair_file_indices.append(
                np.full(len(pair_array), file_index, dtype=np.int64)
            )
            self.pair_rows.append(pair_array)
            self.pair_event_indices.append(event_array)
            self.pair_offsets.append(self.pair_offsets[-1] + len(pair_array))

        self.pair_offsets = np.asarray(self.pair_offsets, dtype=np.int64)
        self.fields = (
            [f"lepton1_{field}" for field in self.fields if not field.startswith("eventwise_")]
            + [f"lepton2_{field}" for field in self.fields if not field.startswith("eventwise_")]
            + [field for field in self.fields if field.startswith("eventwise_")]
        )

    def __len__(self):
        return int(self.pair_offsets[-1])

    def _get_pair(self, index):
        pair_index = int(index)
        if pair_index < 0 or pair_index >= len(self):
            raise IndexError(
                f"Pair index {pair_index} is out of range for length {len(self)}"
            )
        file_index = int(np.searchsorted(self.pair_offsets, pair_index, side="right") - 1)
        local_pair_index = pair_index - int(self.pair_offsets[file_index])
        first_row, second_row = self.pair_rows[file_index][local_pair_index]
        event_index = self.pair_event_indices[file_index][local_pair_index]
        return file_index, int(first_row), int(second_row), int(event_index)

    def __getitem__(self, index):
        file_index, first_row, second_row, event_index = self._get_pair(index)
        h5_file = self._get_handle(file_index)
        if not hasattr(self, "eventwise_cache"):
            self.eventwise_cache = {}
            self.eventwise_features_cache = {}
        if file_index not in self.eventwise_cache:
            self.eventwise_cache[file_index] = h5_file["eventwise"][:]
            self.eventwise_features_cache[file_index] = {}
        eventwise = self.eventwise_cache[file_index]
        first_indices = self._get_eventwise_first_indices(
            file_index, h5_file, eventwise
        )
        eventwise_features = self.eventwise_features_cache[file_index]
        first = self._read_one(
            h5_file, first_row, eventwise, first_indices,
            eventwise_features=eventwise_features,
        )
        second = self._read_one(
            h5_file, second_row, eventwise, first_indices,
            eventwise_features=eventwise_features,
        )
        return (
            first[0],
            second[0],
            first[1],
            torch.stack((first[2], second[2])),
        )

    def __getitems__(self, indices):
        indices = list(indices)
        global_indices = np.asarray(indices, dtype=np.int64)
        file_indices = np.searchsorted(
            self.pair_offsets, global_indices, side="right"
        ) - 1
        local_pair_indices = global_indices - self.pair_offsets[file_indices]
        grouped = {
            int(file_index): np.flatnonzero(file_indices == file_index)
            for file_index in np.unique(file_indices)
        }

        if not hasattr(self, "eventwise_cache"):
            self.eventwise_cache = {}
        if not hasattr(self, "eventwise_features_cache"):
            self.eventwise_features_cache = {}


        batch = [None] * len(indices)
        for file_index, output_positions in grouped.items():
            h5_file = self._get_handle(file_index)
            if file_index not in self.eventwise_cache:
                self.eventwise_cache[file_index] = h5_file["eventwise"][:]
            eventwise = self.eventwise_cache[file_index]
            if file_index not in self.eventwise_features_cache:
                self.eventwise_features_cache[file_index] = None
            eventwise_features = self.eventwise_features_cache[file_index]
            first_indices = self._get_eventwise_first_indices(
                file_index, h5_file, eventwise
            )

            pair_rows = self.pair_rows[file_index][local_pair_indices[output_positions]]
            pair_event_indices = self.pair_event_indices[file_index][
                local_pair_indices[output_positions]
            ]
            unique_rows = np.unique(pair_rows)
            rows, targets, _ = self._read_aligned_rows(
                h5_file,
                unique_rows,
                file_index,
                include_target=True,
                prefer_contiguous_span=True,
            )

            # Flatten all selected structured fields in one vectorized pass.
            # This avoids calling _read_one once per unique electron row.
            electron_features = np.concatenate(
                [
                    np.concatenate(
                        [
                            np.asarray(rows[name][field], dtype=np.float32).reshape(
                                len(unique_rows), -1
                            )
                            for field in self._selected_field_names(
                                name, rows[name].dtype.names
                            )
                        ],
                        axis=1,
                    )
                    for name in self.features
                ],
                axis=1,
            )
            row_event_indices = np.searchsorted(
                first_indices, unique_rows, side="right"
            ) - 1

            if eventwise_features is None:
                eventwise_rows = eventwise
                eventwise_names = self._selected_field_names(
                    "eventwise", eventwise_rows.dtype.names
                )
                eventwise_features = np.concatenate(
                    [
                        np.asarray(eventwise_rows[field], dtype=np.float32).reshape(
                            len(eventwise_rows), -1
                        )
                        for field in eventwise_names
                    ],
                    axis=1,
                )
                self.eventwise_features_cache[file_index] = eventwise_features

            if self.y_source == "eventwise" and self.y_fields:
                targets = np.stack(
                    [
                        np.asarray(
                            eventwise[field][row_event_indices], dtype=np.float32
                        )
                        for field in self.y_fields
                    ],
                    axis=-1,
                )

            first_positions = np.searchsorted(unique_rows, pair_rows[:, 0])
            second_positions = np.searchsorted(unique_rows, pair_rows[:, 1])
            first_features = torch.from_numpy(electron_features[first_positions])
            second_features = torch.from_numpy(electron_features[second_positions])
            pair_eventwise = torch.from_numpy(eventwise_features[pair_event_indices])
            pair_targets = torch.from_numpy(
                np.stack((targets[first_positions], targets[second_positions]), axis=1)
            )

            for batch_position, output_position in enumerate(output_positions):
                batch[output_position] = (
                    first_features[batch_position],
                    second_features[batch_position],
                    pair_eventwise[batch_position],
                    pair_targets[batch_position],
                )

        return batch
        