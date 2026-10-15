import torch
import copy
import numpy as np
from typing import Callable

def _move_to_device(values : tuple, device : torch.device) -> tuple:
    """
    Move a tuple of values to the specified device.

    Arguments:
    - values (tuple): A tuple of values to be moved.
    - device (torch.device): The device to move the values to.

    Returns:
    - tuple: A tuple of the same structure as `values`, but with all tensors moved to the specified device.
    """
    return (
        tuple(value.to(device) for value in values)
        if values is not None
        else (None, None)
    )

def _function_on_inputs(function : callable, inputs : torch.Tensor | tuple, args : tuple, kwargs : dict = {}) -> torch.Tensor | tuple:
    """
    Apply a function to inputs, which can be a single tensor or a tuple of tensors.

    Arguments:
    - function (callable): The function to apply.
    - inputs (torch.Tensor or tuple): The input tensor(s) to apply the function to.
    - args (tuple): The arguments to pass to the function.
    - kwargs (dict): The keyword arguments to pass to the function.

    Returns:
    - torch.Tensor or tuple: The result of applying the function to the inputs.
    """
    if isinstance(inputs, tuple):
        return tuple(function(value, *args, **kwargs) for value in inputs)
    return function(inputs, *args, **kwargs)

def _normalize_inputs(inputs : torch.Tensor | tuple, means : torch.Tensor, stds : torch.Tensor) -> torch.Tensor | tuple:
    """
    Normalize inputs using provided means and standard deviations.

    Arguments:
    - inputs (torch.Tensor or tuple): The input tensor(s) to normalize.
    - means torch.Tensor: The means to use for normalization.
    - stds torch.Tensor: The standard deviations to use for normalization.

    Returns:
    - torch.Tensor or tuple: The normalized input tensor(s).
    """
    if means is not None and stds is not None:
        return _function_on_inputs(lambda x, m, s: (x - m) / s, inputs, (means, stds))
    elif means is not None or stds is not None:
        raise ValueError("Both means and stds must be provided for normalization.")
    else:
        return inputs

def _tuple_unpack_to_device(data : tuple, device : torch.device) -> tuple:
    """
    Unpack a tuple of data and move each element to the specified device.

    Arguments:
    - data (tuple): A tuple containing the data to unpack and move.
    - device (torch.device): The device to move the data to.

    Returns:
    - inputs (torch.Tensor or tuple): The input tensor(s) moved to the specified device.
    - eventwise_features (torch.Tensor): The eventwise features moved to the specified device.
    - targets (torch.Tensor): The target tensor moved to the specified device.
    - domain_targets (torch.Tensor): The domain target tensor moved to the specified device.
    """
    domain_targets = None
    
    if len(data) == 3:
        eventwise_features, targets = data[-2], data[-1]
    elif len(data) in (4, 5):
        # HDF5 datasets append a domain target after the physics target.
        eventwise_features, targets, domain_targets = data[-3], data[-2], data[-1]
    else:
        raise ValueError(f"Unexpected batch structure with {len(data)} elements.")
    eventwise_features, targets = eventwise_features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    if domain_targets is not None:
        domain_targets = domain_targets.to(device, non_blocking=True)
    if len(data) in (3, 4):
        inputs = data[0]
        inputs = inputs.to(device, non_blocking=True)
    elif len(data) == 5:
        inputs_1, inputs_2 = data[0], data[1]
        inputs_1, inputs_2 = inputs_1.to(device, non_blocking=True), inputs_2.to(device, non_blocking=True)
        inputs = (inputs_1, inputs_2)
    return inputs, eventwise_features, targets, domain_targets

def _prepare_binary_targets(targets : torch.Tensor, domain_targets : torch.Tensor, binary_value : float, domain : str = "bool") -> torch.Tensor:
    """
    Prepare binary targets for training by converting them to a binary format based on the specified binary value.

    Arguments:
    - targets (torch.Tensor): The target tensor to prepare.
    - domain_targets (torch.Tensor): The domain target tensor, used to determine which samples are considered for binary conversion.
    - binary_value (float): The value to consider as the positive class (1.0) in the binary target representation.

    Keywords:
    - domain (str): The target of the secondary domain, either "bool" or "int". Default is "bool".

    Returns:
    - torch.Tensor: The prepared binary target tensor.
    """
    if domain_targets is None:
        targets = torch.eq(targets, binary_value).float()
    else:
        targets_0 = torch.eq(targets[domain_targets == 0], binary_value).float()
        if domain == "bool":
            targets_1 = targets[domain_targets == 1].float()
        elif domain == "int":
            targets_1 = torch.eq(targets[domain_targets == 1], binary_value).float()
        targets = torch.cat((targets_0, targets_1), dim=0)
    if targets.ndim >= 3 and targets.shape[-2] == 2:
        # Pair datasets store targets as (batch, two_particles, target_fields).
        targets = targets.all(dim=(-2, -1))
    elif targets.shape[-1] == 2:
        targets = targets[:, 0] * targets[:, 1]
    return targets.reshape(-1)

def _domain_loss(
    loss_function: Callable,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Calculate the unweighted adversarial domain loss."""
    unreduced_loss = _unreduced_loss(loss_function)
    if logits.ndim == 1 or logits.shape[-1] == 1:
        logits = logits.reshape(-1)
        targets = targets.to(dtype=logits.dtype).reshape(-1)
    else:
        targets = targets.to(dtype=torch.long).reshape(-1)
    return unreduced_loss(logits, targets).mean()

def _unreduced_loss(loss_function: Callable) -> Callable:
    if not hasattr(loss_function, "reduction"):
        raise TypeError(
            "The loss function must be a torch loss module with a "
            "'reduction' attribute."
        )
    unreduced_loss = copy.deepcopy(loss_function)
    unreduced_loss.reduction = "none"
    return unreduced_loss

def _weighted_task_loss(
    loss_function: Callable,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    domain_targets: torch.Tensor | None,
    mixed_target_loss_weight: float,
) -> torch.Tensor:
    """Calculate the task loss with reduced weight for weak mixed targets."""
    if domain_targets is None:
        return loss_function(outputs, targets)
    unreduced_loss = _unreduced_loss(loss_function)
    losses = unreduced_loss(outputs, targets)
    losses = losses.reshape(losses.shape[0], -1).mean(dim=1)
    weights = torch.where(
        domain_targets.reshape(-1) == 1,
        torch.as_tensor(
            mixed_target_loss_weight,
            device=losses.device,
            dtype=losses.dtype,
        ),
        torch.ones_like(losses),
    )
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)

def _domain_logits(
    model: torch.nn.Module,
    domain_adversary: torch.nn.Module,
    inputs: torch.Tensor | tuple,
    eventwise_features: torch.Tensor,
    gradient_reversal: torch.nn.Module | None,
) -> torch.Tensor:
    if not hasattr(model, "encode"):
        raise TypeError(
            "Adversarial training requires the main model to provide "
            "an encode(...) method."
        )
    embedding = model.encode(*inputs, eventwise_features)
    if gradient_reversal is not None:
        embedding = gradient_reversal(embedding)
    return domain_adversary(embedding)

def train_model(
        model : torch.nn.Module, 
        optimizer : torch.optim.Optimizer, 
        loss_function : callable, 
        train_loader : torch.utils.data.DataLoader, 
        val_loader : torch.utils.data.DataLoader, 
        num_epochs : int = 10, 
        device : str | torch.device = 'cpu', 
        means : tuple | None = None,
        stds : tuple | None = None, 
        log_target : bool = False, 
        binary_target : bool | float = False,
        domain_adversary : torch.nn.Module | None = None,
        domain_loss_function : Callable | None = None,
        gradient_reversal : torch.nn.Module | None = None,
        domain_loss_weight : float = 1.0,
        mixed_target_loss_weight : float = 0.25,
    ) -> tuple[torch.nn.Module, list, list]:
    """
    Train a PyTorch model using the provided training and validation data loaders, optimizer, and loss function.

    Arguments:
    - model (torch.nn.Module): The PyTorch model to train.
    - optimizer (torch.optim.Optimizer): The optimizer to use for training.
    - loss_function (callable): The loss function to use for training.
    - train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
    - val_loader (torch.utils.data.DataLoader): The data loader for the validation dataset.

    Keywords:
    - num_epochs (int): The number of epochs to train the model. Default is 10.
    - device (str or torch.device): The device to use for training (e.g., 'cpu' or 'cuda'). Default is 'cpu'.
    - means (tuple): A tuple containing the means for input normalization. Default is None.
    - stds (tuple): A tuple containing the standard deviations for input normalization. Default is None.
    - log_target (bool): Whether to apply a logarithmic transformation to the target values to smoothen loss function values. Default is False.
    - binary_target (bool or float): Whether to convert the target values to a binary format. If a float is provided, it will be used as the value to consider as the positive class (1.0). Default is False.
    - domain_adversary (torch.nn.Module, optional): Domain classifier applied
      to the model embedding. Default is None.
    - domain_loss_function (callable, optional): Unreduced-capable domain
      loss, such as ``BCEWithLogitsLoss``. Default is None.
    - gradient_reversal (torch.nn.Module, optional): Gradient-reversal module
      applied before the domain classifier. Required for adversarial training.
    - domain_loss_weight (float): Weight of the domain loss. Default is 1.0.
    - mixed_target_loss_weight (float): Relative weight for the physics loss
      for samples with domain target 1, whose targets are weak labels.
      Default is 0.25.

    Returns:
    - model (torch.nn.Module): The trained PyTorch model.
    - train_losses (list): A list of training losses for each epoch.
    - val_losses (list): A list of validation losses for each epoch.
    """
    if domain_adversary is not None and domain_loss_function is None:
        raise ValueError(
            "domain_loss_function is required when domain_adversary is provided."
        )
    if domain_adversary is not None and gradient_reversal is None:
        raise ValueError(
            "gradient_reversal is required for adversarial training."
        )
    if domain_adversary is None and (
        domain_loss_function is not None or gradient_reversal is not None
    ):
        raise ValueError(
            "domain_adversary is required when configuring adversarial training."
        )
    if domain_loss_weight < 0 or mixed_target_loss_weight < 0:
        raise ValueError("Loss weights must be non-negative.")

    model.to(device)
    if domain_adversary is not None:
        domain_adversary.to(device)
        optimizer_param_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        missing_parameters = [
            parameter
            for parameter in domain_adversary.parameters()
            if id(parameter) not in optimizer_param_ids
        ]
        if missing_parameters:
            optimizer.add_param_group({"params": missing_parameters})
    if gradient_reversal is not None:
        gradient_reversal.to(device)
    if isinstance(loss_function, torch.nn.Module):
        loss_function.to(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    best_domain_adversary_state = None

    means_i, means_e = _move_to_device(means, device)
    stds_i, stds_e = _move_to_device(stds, device)

    binary_value = torch.as_tensor(binary_target, dtype=torch.float32, device=device) if binary_target is not False else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        samples_seen = 0
        expected_batches = len(train_loader)
        for batch, data in enumerate(train_loader):
            inputs, eventwise_features, targets, domain_targets = _tuple_unpack_to_device(data, device)

            if not torch.isfinite(targets).all():
                raise ValueError(f"Warning: Non-finite values detected in targets")
            if log_target:
                targets = torch.log1p(targets)  # Add 1 to avoid log(0)
            if binary_target is not False:
                targets = _prepare_binary_targets(targets, domain_targets, binary_value, domain = "bool")

            inputs = _normalize_inputs(inputs, means_i, stds_i)
            eventwise_features = _normalize_inputs(eventwise_features, means_e, stds_e)
            inputs = _function_on_inputs(torch.nan_to_num, inputs, (), {'nan': 0.0, 'posinf': 0.0, 'neginf': 0.0})
            eventwise_features = torch.nan_to_num(eventwise_features, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0

            optimizer.zero_grad()
            outputs = model(*inputs, eventwise_features)

            targets = targets.to(dtype=outputs.dtype)
            task_loss = _weighted_task_loss(
                loss_function,
                outputs,
                targets,
                domain_targets,
                mixed_target_loss_weight,
            )
            loss = task_loss
            if domain_adversary is not None:
                if domain_targets is None:
                    raise ValueError(
                        "The data loader must provide domain targets for "
                        "adversarial training."
                    )
                domain_outputs = _domain_logits(
                    model,
                    domain_adversary,
                    inputs,
                    eventwise_features,
                    gradient_reversal,
                )
                loss = loss + domain_loss_weight * _domain_loss(
                    domain_loss_function,
                    domain_outputs,
                    domain_targets,
                )
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            samples_seen += batch_size

            average_loss = running_loss / samples_seen
            print(
                f'Batch {batch + 1}/{expected_batches}, '
                f'Loss: {average_loss:.4f}',
                end='\r',
            )

        if samples_seen != len(train_loader.dataset):
            raise RuntimeError(
                f"Training loader yielded {samples_seen} samples, "
                f"but its dataset contains {len(train_loader.dataset)}."
            )
        epoch_loss = running_loss / samples_seen

        val_loss = validate_model(
            model,
            loss_function,
            val_loader,
            device,
            means,
            stds,
            log_target=log_target,
            binary_target=binary_target,
            domain_adversary=domain_adversary,
            domain_loss_function=domain_loss_function,
            gradient_reversal=gradient_reversal,
            domain_loss_weight=domain_loss_weight,
            mixed_target_loss_weight=mixed_target_loss_weight,
        )
        val_losses.append(val_loss)
        train_losses.append(epoch_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            if domain_adversary is not None:
                best_domain_adversary_state = copy.deepcopy(
                    domain_adversary.state_dict()
                )

    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_domain_adversary_state is not None:
        domain_adversary.load_state_dict(best_domain_adversary_state)

    return train_losses, val_losses



def validate_model(
        model : torch.nn.Module, 
        loss_function : callable, 
        val_loader : torch.utils.data.DataLoader, 
        device : str | torch.device = 'cpu', 
        means : tuple | None = None, 
        stds : tuple | None = None, 
        log_target : bool = False, 
        binary_target : bool | float = False,
        domain_adversary : torch.nn.Module | None = None,
        domain_loss_function : Callable | None = None,
        gradient_reversal : torch.nn.Module | None = None,
        domain_loss_weight : float = 1.0,
        mixed_target_loss_weight : float = 0.25,
        ) -> float:
    """
    Validate a PyTorch model using the provided validation data loader and loss function.

    Arguments:
    - model (torch.nn.Module): The PyTorch model to validate.
    - loss_function (callable): The loss function to use for validation.
    - val_loader (torch.utils.data.DataLoader): The data loader for the validation dataset.

    Keywords:
    - device (str or torch.device): The device to use for validation (e.g., 'cpu' or 'cuda'). Default is 'cpu'.
    - means (tuple): A tuple containing the means for input normalization. Default is None.
    - stds (tuple): A tuple containing the standard deviations for input normalization. Default is None.
    - log_target (bool): Whether to apply a logarithmic transformation to the target values to smoothen loss function values. Default is False.
    - binary_target (bool or float): Whether to convert the target values to a binary format. If a float is provided, it will be used as the value to consider as the positive class (1.0). Default is False.
    - domain_adversary (torch.nn.Module, optional): Domain classifier applied
      to the model embedding. Default is None.
    - domain_loss_function (callable, optional): Domain loss function.
    - gradient_reversal (torch.nn.Module, optional): Gradient-reversal module
      applied before the domain classifier.
    - domain_loss_weight (float): Weight of the domain loss. Default is 1.0.
    - mixed_target_loss_weight (float): Relative weight for the physics loss
      for domain target 1, whose targets are weak labels. Default is 0.25.

    Returns:
    - float: The average validation loss over the entire validation dataset.
    """
    if domain_adversary is not None and domain_loss_function is None:
        raise ValueError(
            "domain_loss_function is required when domain_adversary is provided."
        )
    if domain_adversary is not None and gradient_reversal is None:
        raise ValueError(
            "gradient_reversal is required for adversarial validation."
        )
    if domain_adversary is None and (
        domain_loss_function is not None or gradient_reversal is not None
    ):
        raise ValueError(
            "domain_adversary is required when configuring adversarial validation."
        )
    if domain_loss_weight < 0 or mixed_target_loss_weight < 0:
        raise ValueError("Loss weights must be non-negative.")

    # Validation phase
    model.eval()
    if domain_adversary is not None:
        domain_adversary.eval()
    val_loss = 0.0
    samples_seen = 0
    means_i, means_e = _move_to_device(means, device)
    stds_i, stds_e = _move_to_device(stds, device)
    binary_value = torch.as_tensor(binary_target, dtype=torch.float32, device=device) if binary_target is not False else None
    with torch.no_grad():
        for data in val_loader:
            inputs, event_features, targets, domain_targets = _tuple_unpack_to_device(data, device)
            if log_target:
                targets = torch.log1p(targets)
            if binary_target is not False:
                targets = _prepare_binary_targets(targets, domain_targets, binary_value, domain = "bool")

            if not torch.isfinite(targets).all():
                raise ValueError(f"Warning: Non-finite values detected in targets")
            inputs = _normalize_inputs(inputs, means_i, stds_i)
            event_features = _normalize_inputs(event_features, means_e, stds_e)

            inputs = _function_on_inputs(torch.nan_to_num, inputs, (), {'nan': 0.0, 'posinf': 0.0, 'neginf': 0.0})
            event_features = torch.nan_to_num(event_features, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0
            
            outputs = model(*inputs, event_features)

            targets = targets.to(dtype=outputs.dtype)
            task_loss = _weighted_task_loss(
                loss_function,
                outputs,
                targets,
                domain_targets,
                mixed_target_loss_weight,
            )
            loss = task_loss
            if domain_adversary is not None:
                if domain_targets is None:
                    raise ValueError(
                        "The data loader must provide domain targets for "
                        "adversarial validation."
                    )
                domain_outputs = _domain_logits(
                    model,
                    domain_adversary,
                    inputs,
                    event_features,
                    gradient_reversal,
                )
                loss = loss + domain_loss_weight * _domain_loss(
                    domain_loss_function,
                    domain_outputs,
                    domain_targets,
                )
            batch_size = targets.size(0)
            val_loss += loss.item() * batch_size
            samples_seen += batch_size

    if samples_seen != len(val_loader.dataset):
        raise RuntimeError(
            f"Validation loader yielded {samples_seen} samples, "
            f"but its dataset contains {len(val_loader.dataset)}."
        )
    val_loss /= samples_seen
    return val_loss

def predict(
    model : torch.nn.Module, 
    test_loader : torch.utils.data.DataLoader, 
    device : str | torch.device = 'cpu', 
    means : tuple | None = None, 
    stds : tuple | None = None, 
    log_target : bool = False
) -> np.ndarray:
    """
    Predict using a trained PyTorch model on the provided test data loader.

    Arguments:
    - model (torch.nn.Module): The trained PyTorch model to use for prediction.
    - test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Keywords:
    - device (str or torch.device): The device to use for prediction (e.g., 'cpu' or 'cuda'). Default is 'cpu'.
    - means (tuple): A tuple containing the means for input normalization. Default is None.
    - stds (tuple): A tuple containing the standard deviations for input normalization. Default is None.
    - log_target (bool): Whether a logarithmic transformation was applied to the target values during training to smoothen loss function values. Default is False.
    
    Returns:
    - predictions (numpy.ndarray): An array of predicted values.
    """
    model.eval()
    predictions = []
    means_i, means_e = _move_to_device(means, device)
    stds_i, stds_e = _move_to_device(stds, device)
    with torch.no_grad():
        for data in test_loader:
            inputs, event_features, _, _ = _tuple_unpack_to_device(data, device)
            inputs = _normalize_inputs(inputs, means_i, stds_i)
            event_features = _normalize_inputs(event_features, means_e, stds_e)

            inputs = _function_on_inputs(torch.nan_to_num, inputs, (), {'nan': 0.0, 'posinf': 0.0, 'neginf': 0.0})  # Replace NaN values with 0.0
            event_features = torch.nan_to_num(event_features, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0

            outputs = model(*inputs, event_features)
            outputs = outputs.to(dtype=torch.float32)
            if log_target:
                outputs = torch.expm1(outputs)  # Apply inverse of log1p if log_target is True
            predictions.append(outputs.cpu().numpy())
        print("Prediction completed")
    return np.concatenate(predictions)
