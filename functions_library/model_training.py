import torch
import copy
import numpy as np

def _move_to_device(values, device):
    return (
        tuple(value.to(device) for value in values)
        if values is not None
        else (None, None)
    )

def _function_on_inputs(function, inputs, args, kwargs=None):
    if kwargs is None:
        kwargs = {}
    if isinstance(inputs, tuple):
        return tuple(function(value, *args, **kwargs) for value in inputs)
    return function(inputs, *args, **kwargs)

def _normalize_inputs(inputs, means, stds):
    if means is not None and stds is not None:
        return _function_on_inputs(lambda x, m, s: (x - m) / s, inputs, (means, stds))
    elif means is not None or stds is not None:
        raise ValueError("Both means and stds must be provided for normalization.")
    else:
        return inputs

def _tuple_unpack_to_device(data, device):
    eventwise_features, targets = data[-2], data[-1]
    eventwise_features, targets = eventwise_features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    if len(data) == 3:
        inputs = data[0]
        inputs = inputs.to(device, non_blocking=True)
    else:
        inputs_1, inputs_2 = data[0], data[1]
        inputs_1, inputs_2 = inputs_1.to(device, non_blocking=True), inputs_2.to(device, non_blocking=True)
        inputs = (inputs_1, inputs_2)
    return inputs, eventwise_features, targets

def _prepare_binary_targets(targets, binary_value):
    targets = torch.eq(targets, binary_value).float()
    if targets.ndim >= 3 and targets.shape[-2] == 2:
        # Pair datasets store targets as (batch, two_particles, target_fields).
        targets = targets.all(dim=(-2, -1))
    elif targets.shape[-1] == 2:
        targets = targets[:, 0] * targets[:, 1]
    return targets.reshape(-1)

def train_model(model, optimizer, loss_function, train_loader, val_loader, num_epochs=10, device='cpu', means=None, stds=None, log_target=False, binary_target=False):
    model.to(device)
    if isinstance(loss_function, torch.nn.Module):
        loss_function.to(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    train_losses = []
    val_losses = []
    batch_losses = {}
    best_val_loss = float('inf')
    best_model_state = None

    means_i, means_e = _move_to_device(means, device)
    stds_i, stds_e = _move_to_device(stds, device)

    binary_value = torch.as_tensor(binary_target, dtype=torch.float32, device=device) if binary_target is not False else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_losses[epoch] = []
        for batch, data in enumerate(train_loader):
            inputs, eventwise_features, targets = _tuple_unpack_to_device(data, device)

            if not torch.isfinite(targets).all():
                raise ValueError(f"Warning: Non-finite values detected in targets")
            if log_target:
                targets = torch.log1p(targets)  # Add 1 to avoid log(0)
            if binary_target is not False:
                targets = _prepare_binary_targets(targets, binary_value)

            inputs = _normalize_inputs(inputs, means_i, stds_i)
            eventwise_features = _normalize_inputs(eventwise_features, means_e, stds_e)
            inputs = _function_on_inputs(torch.nan_to_num, inputs, (), {'nan': 0.0, 'posinf': 0.0, 'neginf': 0.0})
            eventwise_features = torch.nan_to_num(eventwise_features, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0

            optimizer.zero_grad()
            outputs = model(*inputs, eventwise_features)

            targets = targets.to(dtype=outputs.dtype)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            batch_losses[epoch].append(loss.item())

            print(f'Batch {batch+1}/{len(train_loader)}, Loss: {loss.item():.4f}', end='\r')

        epoch_loss = running_loss / len(train_loader.dataset)

        val_loss = validate_model(model, loss_function, val_loader, device, means, stds, log_target=log_target, binary_target=binary_target)
        val_losses.append(val_loss)
        train_losses.append(epoch_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, batch_losses



def validate_model(model, loss_function, val_loader, device='cpu', means=None, stds=None, log_target=False, binary_target=False):
    # Validation phase
    model.eval()
    val_loss = 0.0
    means_i, means_e = _move_to_device(means, device)
    stds_i, stds_e = _move_to_device(stds, device)
    binary_value = torch.as_tensor(binary_target, dtype=torch.float32, device=device) if binary_target is not False else None
    with torch.no_grad():
        for data in val_loader:
            inputs, event_features, targets = _tuple_unpack_to_device(data, device)
            if log_target:
                targets = torch.log1p(targets)
            if binary_target is not False:
                targets = _prepare_binary_targets(targets, binary_value)

            if not torch.isfinite(targets).all():
                raise ValueError(f"Warning: Non-finite values detected in targets")
            inputs = _normalize_inputs(inputs, means_i, stds_i)
            event_features = _normalize_inputs(event_features, means_e, stds_e)

            inputs = _function_on_inputs(torch.nan_to_num, inputs, (), {'nan': 0.0, 'posinf': 0.0, 'neginf': 0.0})
            event_features = torch.nan_to_num(event_features, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0
            
            outputs = model(*inputs, event_features)

            targets = targets.to(dtype=outputs.dtype)
            loss = loss_function(outputs, targets)
            val_loss += loss.item() * targets.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss

def predict(model, test_loader, device='cpu', means=None, stds=None, log_target=False):
    model.eval()
    predictions = []
    means_i, means_e = _move_to_device(means, device)
    stds_i, stds_e = _move_to_device(stds, device)
    with torch.no_grad():
        for data in test_loader:
            inputs, event_features, _ = _tuple_unpack_to_device(data, device)
            inputs = _normalize_inputs(inputs, means_i, stds_i)
            event_features = _normalize_inputs(event_features, means_e, stds_e)

            inputs = _function_on_inputs(torch.nan_to_num, inputs, (), {'nan': 0.0, 'posinf': 0.0, 'neginf': 0.0})  # Replace NaN values with 0.0
            event_features = torch.nan_to_num(event_features, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0

            outputs = model(*inputs, event_features)
            if log_target:
                outputs = torch.expm1(outputs)  # Apply inverse of log1p if log_target is True
            predictions.append(outputs.cpu().numpy())
        print("Prediction completed")
    return np.concatenate(predictions)

