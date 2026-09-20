import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import nn, optim
import time

class H5EgammaDataset_fully_batched(Dataset):
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
        self.y_field = y_field
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
            if self.y_source not in feature_names:
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
                target = np.asarray(h5_file[self.y_source][self.y_field][:])
                valid_rows = np.flatnonzero(np.isfinite(target))
                self.lengths.append(len(valid_rows))
                self.valid_rows.append(valid_rows)

        if self.mix_files:
            self._apply_mixture(mixture_ratio, mixture_seed)

        self.offsets = np.concatenate(([0], np.cumsum(self.lengths)))

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
        file_index = np.searchsorted(self.offsets, index, side="right") - 1
        filtered_index = int(index - self.offsets[file_index])
        local_index = int(self.valid_rows[file_index][filtered_index])
        if file_index not in self.handles:
            self.handles[file_index] = h5py.File(self.files[file_index], "r")
        return file_index, self.handles[file_index], local_index

    def _selected_field_names(self, dataset_name, field_names):
        excluded = set(self.exclude_fields)
        if dataset_name == self.y_source:
            excluded.add(self.y_field)

        if self.include_fields is not None:
            return [name for name in field_names if name in self.include_fields and name not in excluded]
        return [name for name in field_names if name not in excluded]

    #Converts a structured row to a flat array using the same field selection as the header.
    def _structured_to_array(self, rows, dataset_name):
        names = self._selected_field_names(dataset_name, rows.dtype.names)
        values = [np.asarray(rows[name], dtype=np.float32).reshape(-1) for name in names]
        return np.concatenate(values) if values else np.empty(0, dtype=np.float32)

    def _eventwise_first_indices(self, eventwise):
        count_field = eventwise.dtype.names[0]
        counts = np.asarray(eventwise[count_field], dtype=np.int64)
        return np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))

    #Tries to read aligned rows in few contiguous slices if possible, otherwise reads them individually. Returns the rows and the sort order of the indices.
    def _read_aligned_rows(self, h5_file, indices, file_index=None):
        indices = np.asarray(indices, dtype=np.int64)
        sort_order = np.argsort(indices)
        sorted_indices = indices[sort_order]
        runs = np.flatnonzero(np.diff(sorted_indices) > 1) + 1
        run_starts = np.r_[0, runs]
        run_ends = np.r_[runs, len(sorted_indices)]
        use_slices = len(run_starts) <= max(1, len(sorted_indices) // 2)

        #Reverts to reading individual rows if the number of contiguous slices is too high, otherwise reads them in slices.
        if not use_slices:
            if self.mix_files:
                target = np.full(len(sorted_indices), self.file_labels[file_index], dtype=np.float32)
            elif self.y_source == "eventwise":
                target = [None] * len(sorted_indices)
            else:
                target = h5_file[self.y_source][self.y_field][sorted_indices]
            return {name: h5_file[name][sorted_indices] for name in self.features}, target, sort_order

        #This part takes care of the contigous reads. Reading a batch of 32 rows in one go reduces computation from 8s to 0.2s
        rows = {name: [] for name in self.features}
        target = []
        for start, end in zip(run_starts, run_ends):
            first = int(sorted_indices[start])
            last = int(sorted_indices[end - 1]) + 1
            relative_indices = sorted_indices[start:end] - first
            for name in self.features:
                rows[name].append(h5_file[name][first:last][relative_indices])
            if self.y_source != "eventwise" and not self.mix_files:
                target.append(h5_file[self.y_source][self.y_field][first:last][relative_indices])
        if self.mix_files:
            target = np.full(len(sorted_indices), self.file_labels[file_index], dtype=np.float32)
            return {name: np.concatenate(rows[name]) for name in self.features}, target, sort_order
        if self.y_source == "eventwise":
            return {name: np.concatenate(rows[name]) for name in self.features}, [None]*len(sorted_indices), sort_order
        return {name: np.concatenate(rows[name]) for name in self.features}, np.concatenate(target), sort_order

    #This function groups a single data row into a torch tensor of features and a torch tensor of the target variable. It can also read the h5 file if no data is set yet.
    def _read_one(self, h5_file, local_index, eventwise=None, first_indices=None, rows=None, target=None):
        
        if eventwise is None:
            eventwise = h5_file["eventwise"][:]
        eventwise_names = eventwise.dtype.names
        if first_indices is None:
            first_indices = self._eventwise_first_indices(eventwise)

        event_index = int(np.searchsorted(first_indices, local_index, side="right") - 1)
        if event_index < 0 or local_index >= first_indices[event_index] + eventwise[eventwise_names[0]][event_index]:
            raise IndexError(f"Egamma row {local_index} is not covered by eventwise")

        if rows is None:
            rows = {name: h5_file[name][local_index] for name in self.features}

        feature_arrays = [
            self._structured_to_array(rows[name], name)
            for name in self.features
        ]

        feature_arrays.append(self._structured_to_array(eventwise[event_index], "eventwise"))
        features = np.concatenate(feature_arrays)
        
               
        if self.y_source not in h5_file:
            raise KeyError(f"Unknown y source: {self.y_source}")
        target_index = event_index if self.y_source == "eventwise" else local_index
        if target is None:
            target = np.float32(h5_file[self.y_source][self.y_field][target_index])
        return torch.from_numpy(features), torch.tensor(target)
    
    #Depreciated single row read function. Torch dataloader will call __getitems__ instead of this function.
    def __getitem__(self, index):
        file_index, h5_file, local_index = self._get_file_and_row(index)
        target = self.file_labels[file_index] if self.mix_files else None
        return self._read_one(h5_file, local_index, target=target)

    #Main read function for the fully batched dataset. It groups the indices by file, reads the aligned rows in batches, and returns a list of feature-target pairs.
    def __getitems__(self, indices):
        indices = list(indices)
        grouped = {}
        for output_position, index in enumerate(indices):
            file_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
            filtered_index = int(index - self.offsets[file_index])
            local_index = int(self.valid_rows[file_index][filtered_index])
            grouped.setdefault(file_index, []).append((output_position, local_index))

        if not hasattr(self, "eventwise_cache"):
            self.eventwise_cache = {}

        batch = [None] * len(indices)
        for file_index, positions in grouped.items():
            if file_index not in self.handles:
                self.handles[file_index] = h5py.File(self.files[file_index], "r")
            h5_file = self.handles[file_index]
            if file_index not in self.eventwise_cache:
                self.eventwise_cache[file_index] = h5_file["eventwise"][:]
            eventwise = self.eventwise_cache[file_index]
            local_indices = np.array([local_index for _, local_index in positions])
            rows, target, sort_order = self._read_aligned_rows(h5_file, local_indices, file_index)
            sorted_indices = np.sort(local_indices)
            first_indices = self._eventwise_first_indices(eventwise)

            for sorted_position, original_position in enumerate(np.argsort(sort_order)):
                batch[positions[original_position][0]] = self._read_one(
                    h5_file,
                    int(sorted_indices[sorted_position]),
                    eventwise,
                    first_indices,
                    {name: rows[name][sorted_position] for name in rows},
                    target[sorted_position],
                )

        return batch
    
    #Closes all open file handles and clears the handles dictionary to free up resources.
    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

class ShuffledContiguousBatchSampler(Sampler):
    #Initializes the sampler with the dataset size and batch size. It will create batches of contiguous indices and shuffle them for each epoch.
    def __init__(self, dataset_size, batch_size, dataset=None):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.dataset = dataset

    #This function generates batches of contiguous indices, shuffles the starting indices of the batches, and yields the indices for each batch.
    def __iter__(self):
        if self.dataset is not None and self.dataset.mix_files:
            yield from self._mixed_batches()
            return

        starts = np.arange(0, self.dataset_size, self.batch_size)
        np.random.shuffle(starts)

        for start in starts:
            stop = min(start + self
                       .batch_size, self.dataset_size)
            yield list(range(start, stop))

    def _mixed_batches(self):
        source_indices = [
            np.concatenate([
                np.arange(self.dataset.offsets[file_index], self.dataset.offsets[file_index + 1])
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
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetwork, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(last_size, hidden_size))
            layers.append(torch.nn.ReLU())
            last_size = hidden_size
        layers.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(model, optimizer, loss_function, train_loader, val_loader, num_epochs=10, device='cpu', means=None, stds=None):
    model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    means = means.to(device) if means is not None else None
    stds = stds.to(device) if stds is not None else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if not torch.isfinite(targets).all():
                raise ValueError(f"Warning: Non-finite values detected in targets")
            if means is not None and stds is not None:
                inputs = (inputs - means) / stds  # Normalize inputs using provided means and stds
            elif means is not None or stds is not None:
                raise ValueError("Both means and stds must be provided for normalization.")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            print(f'Batch {batch+1}/{len(train_loader)}, Loss: {loss.item():.4f}', end='\r')

        epoch_loss = running_loss / len(train_loader.dataset)

        val_loss = validate_model(model, loss_function, val_loader, device, means, stds)
        val_losses.append(val_loss)
        train_losses.append(epoch_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, val_losses, train_losses


def validate_model(model, loss_function, val_loader, device='cpu', means=None, stds=None):
        # Validation phase
        model.eval()
        val_loss = 0.0
        means = means.to(device) if means is not None else None
        stds = stds.to(device) if stds is not None else None
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                if not torch.isfinite(targets).all():
                    raise ValueError(f"Warning: Non-finite values detected in targets")
                if means is not None and stds is not None:
                    inputs = (inputs - means) / stds  # Normalize inputs using provided means and stds
                elif means is not None or stds is not None:
                    raise ValueError("Both means and stds must be provided for normalization.")
                inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0

                outputs = model(inputs)
                loss = loss_function(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        return val_loss

def predict_model(model, test_loader, device='cpu', means=None, stds=None):
    model.eval()
    predictions = []
    means = means.to(device) if means is not None else None
    stds = stds.to(device) if stds is not None else None
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            
            if means is not None and stds is not None:
                inputs = (inputs - means) / stds  # Normalize inputs using provided means and stds
            elif means is not None or stds is not None:
                raise ValueError("Both means and stds must be provided for normalization.")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0

            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
        print("Prediction completed")
        print(f"Predictions shape: {[p.shape for p in predictions]}")
    return np.concatenate(predictions)

def h5_to_csv(h5_file_path, csv_file_path, y_source, y_field, exclude_features=None, exclude_fields=None, include_features=None, include_fields=None, batch_size=256, sample_size=100_000, mix_h5_file_path=None, mixture_ratio=None, mixture_seed=0):
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

    dataset = H5EgammaDataset_fully_batched(
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
    means, stds = compute_mean_std(dataset, sample_size=sample_size, batch_size=batch_size)

    with open(csv_file_path, 'w', newline='') as csv_file:
        # Write header
        target_name = "source" if mix_h5_files is not None else f"{y_source}_{y_field}"
        csv_file.write(','.join(dataset.fields + [target_name]) + '\n')

        # Write data rows
        Writing_time_start = time.perf_counter()
        for batch, (features, target) in enumerate(loader):
            for feature_row, target_value in zip(features.numpy(), target.numpy()):
                if np.isnan(target_value):  # Check for NaN values in target
                    continue  # Skip rows with NaN target values
                feature_row = np.nan_to_num((feature_row-means)/stds, nan=0.0)  # Replace NaN values in features with 0.0
                row = np.concatenate((feature_row, [target_value]))
                csv_file.write(','.join(map(str, row)) + '\n')
            elapsed_time = time.perf_counter() - Writing_time_start
            completion_percentage = (batch + 1) / len(loader)
            estimated_total_time = elapsed_time / completion_percentage
            estimated_time_remaining = estimated_total_time - elapsed_time
            print(f'Writing batch {batch+1}/{len(loader)}, estimated time to completion: {estimated_time_remaining:.0f} seconds', end='\r')

    dataset.close()

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

