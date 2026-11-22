import h5py
import torch
import copy
import numpy as np
from typing import Callable
from . import dataset_classes, neural_models, misc_functions, data_loading
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

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

def _call_with_inputs(function : callable, inputs : torch.Tensor | tuple, eventwise_features : torch.Tensor) -> torch.Tensor:
    """Call a model or encoder with either single-particle or pair inputs."""
    if isinstance(inputs, tuple):
        return function(*inputs, eventwise_features)
    return function(inputs, eventwise_features)

def _predict_mask(
    mask_model: torch.nn.Module,
    inputs: torch.Tensor | tuple,
    eventwise_features: torch.Tensor,
) -> torch.Tensor:
    """Return a boolean mask from one positive logit per input pair."""
    with torch.inference_mode():
        mask_output = _call_with_inputs(mask_model, inputs, eventwise_features)
    if mask_output.numel() != eventwise_features.shape[0]:
        raise ValueError(
            "mask_model must return exactly one logit per input pair; "
            f"got shape {tuple(mask_output.shape)} for batch size "
            f"{eventwise_features.shape[0]}."
        )
    return mask_output.reshape(-1) > -2

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

    Mixed-domain targets retain the original batch order so they remain
    aligned with the corresponding inputs.
    """
    if (domain_targets is None or domain == "int"):
        targets = torch.eq(targets, binary_value).float()
    elif domain == "bool":
        binary_targets = torch.eq(targets, binary_value).float()
        domain_mask = domain_targets.reshape(
            domain_targets.shape[0],
            *([1] * (targets.ndim - 1)),
        )
        targets = torch.where(domain_mask == 0, binary_targets, targets)
    else:
        raise ValueError(f"Invalid domain value: {domain}. Must be 'bool' or 'int'.")
    if (targets.ndim >= 3 and targets.shape[-2] == 2):
        # Pair datasets store targets as (batch, two_particles, target_fields).
        targets = targets.all(dim=(-2, -1))
    elif (targets.shape[-1] == 2 and targets.ndim == 2):
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
    variance_target_base_value: torch.Tensor | None = None
) -> torch.Tensor:
    """Calculate the task loss with reduced weight for weak mixed targets."""
    if domain_targets is None:
        return loss_function(outputs, targets)
    unreduced_loss = _unreduced_loss(loss_function)
    if variance_target_base_value is not None:
        losses = unreduced_loss(outputs, targets, variance_target_base_value)
    else:
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
    embedding = _call_with_inputs(model.encode, inputs, eventwise_features)
    if gradient_reversal is not None:
        embedding = gradient_reversal(embedding)
    return domain_adversary(embedding)

def _prepare_z_targets(targets : torch.Tensor) -> torch.Tensor:
    """
    Prepare z targets for training by converting them to a z target format.

    Arguments:
    - targets (torch.Tensor): The target tensor to prepare.

    Returns:
    - torch.Tensor: The prepared z target tensor.
    """
    if targets.shape[-1] < 3 or targets.shape[1] != 2:
        raise ValueError(
            "z_target requires the last dimensions of targets to be (2, 3) or (2, 4)."
        )
    
    pt1 = targets[:, 0, 0]
    pt2 = targets[:, 1, 0]
    eta1 = targets[:, 0, 1]
    eta2 = targets[:, 1, 1]
    phi1 = targets[:, 0, 2]
    phi2 = targets[:, 1, 2] 

    if targets.shape[-1] < 4:
        e1 = pt1 * torch.cosh(eta1)
        e2 = pt2 * torch.cosh(eta2)
    
    x1 = pt1 * torch.cos(phi1)
    y1 = pt1 * torch.sin(phi1)
    z1 = pt1 * torch.sinh(eta1)
    x2 = pt2 * torch.cos(phi2)
    y2 = pt2 * torch.sin(phi2)
    z2 = pt2 * torch.sinh(eta2)
    z_mass = torch.sqrt(torch.abs((e1 + e2)**2 - (x1 + x2)**2 - (y1 + y2)**2 - (z1 + z2)**2))
    return z_mass

def _prepare_predict_outputs(outputs: torch.Tensor, device: torch.device, norm_target: tuple | None, variance_target: bool, gev_target: bool = False) -> torch.Tensor:
    """
    Prepare the outputs of the model for prediction by applying inverse normalization and transformations based on the provided flags.

    Arguments:
    - outputs (torch.Tensor): The output tensor from the model.
    - device (torch.device): The device to move the outputs to.
    - norm_target (tuple or None): A tuple containing the mean and standard deviation for target normalization. If provided, the outputs will be denormalized.

    Returns:
    - torch.Tensor: The prepared output tensor for prediction.
    """
    if gev_target:
        outputs = outputs * 1000.0

    if norm_target:
        mean_t, std_t = _move_to_device(norm_target, device)
        if variance_target:
            outputs = outputs + 2 * torch.log(std_t)
        else:
            outputs = outputs * std_t + mean_t

    elif variance_target:
        outputs = torch.exp(0.5 * outputs)



    return outputs

def _prepare_training_inputs(data: tuple[torch.Tensor, ...], 
                             device: torch.device, 
                             means : tuple | None,
                             stds : tuple | None,
                             norm_target: tuple | None = None, 
                             gev_target: bool = False,
                             z_pid: tuple[bool, torch.nn.Module, torch.nn.Module] = (False, None, None),
                             binary_value: int | None = None, 
                             mask_model: torch.nn.Module | None = None, 
                             z_target: bool = False, 
                             domain_input: bool = False,
                             verbosity: int = 1,
                             domain: str = "bool",
                             batch_numbers : tuple[int, int] = (0, 0)
                             ) -> tuple:
    inputs, eventwise_features, targets, domain_targets = _tuple_unpack_to_device(data, device)
    means_i, means_e = means
    stds_i, stds_e = stds
    skip_batch = False

    if z_pid[0]:
        print("using z_pid", end='\r')
        origin_inputs = targets[:, :, 4:]
        pt_eta_phi = targets[:, :, 1:4]
        targets = targets[:, :, 0]

        z_masses = misc_functions._compute_z_mass(pt_eta_phi[:, 0, 0], pt_eta_phi[:, 0, 1], pt_eta_phi[:, 0, 2], pt_eta_phi[:, 1, 0], pt_eta_phi[:, 1, 1], pt_eta_phi[:, 1, 2])
        z_mass_mask = abs(z_masses/1000.0 - 91.2) < 10.0
        z_mass_mask = z_mass_mask.to(dtype=torch.float32)
        targets[domain_targets == 1, 0], targets[domain_targets == 1, 1] = z_mass_mask[domain_targets == 1], z_mass_mask[domain_targets == 1]

        mean_t, std_t = _move_to_device(norm_target, device)
        mean_t, std_t = mean_t[4:-1], std_t[4:-1]

    elif mask_model is not None:
        if targets.ndim != 2 or targets.shape[-1] < 2:
            raise ValueError(
                "mask_model requires at least two target columns: "
                "[physics_target, mask_target]."
            )
        mask_targets = targets[:, -1]
        targets = targets[:, :-1]
        if targets.shape[-1] == 1:
            targets = targets.reshape(-1)

    if not torch.isfinite(targets).all():
        raise ValueError(f"Warning: Non-finite values detected in targets")
    if z_target:
        print("using z_target", end='\r')
        targets = _prepare_z_targets(targets)

    if (binary_value == 13) & (domain_targets is not None) & (not z_pid[0]):
        pt_eta_phi = targets[:, :, 1:4]
        targets = targets[:, :, 0]
        targets = _prepare_binary_targets(targets, domain_targets, binary_value, domain = domain)

        z_masses = misc_functions._compute_z_mass(pt_eta_phi[:, 0, 0], pt_eta_phi[:, 0, 1], pt_eta_phi[:, 0, 2], pt_eta_phi[:, 1, 0], pt_eta_phi[:, 1, 1], pt_eta_phi[:, 1, 2])
        z_mass_mask = (abs(z_masses/1000.0 - 91.2) < 50.0) | (abs(z_masses/1000.0 - 91.2) > 10.0) | (domain_targets == 0)

        inputs = inputs[0][z_mass_mask], inputs[1][z_mass_mask]
        targets = targets[z_mass_mask]
        domain_targets = domain_targets[z_mass_mask]
        eventwise_features = eventwise_features[z_mass_mask]
        z_masses = z_masses[z_mass_mask]
    
        z_mask_1 = (abs(z_masses/1000.0 - 91.2) < 10.0) & (domain_targets == 1)
        z_mask_0 = (abs(z_masses/1000.0 - 91.2) > 50.0) & (domain_targets == 1)

        targets[z_mask_1] = targets[z_mask_1] * 1.0
        targets[z_mask_0] = targets[z_mask_0] * 0.0
    elif (binary_value == 2) & (domain_targets is not None):
        passid_loose = targets[:, 1]
        targets = targets[:, 0]
        targets = _prepare_binary_targets(targets, domain_targets, binary_value, domain = domain)

        id_mask = (targets > 0.5) | (passid_loose < 0.5) | (domain_targets == 0)

        inputs = inputs[id_mask]
        targets = targets[id_mask]
        domain_targets = domain_targets[id_mask]
        eventwise_features = eventwise_features[id_mask]
        passid_loose = passid_loose[id_mask]
    
        id_mask_1 = (targets > 0.5) & (domain_targets == 1)
        id_mask_0 = (passid_loose < 0.5) & (domain_targets == 1)

        targets[id_mask_1] = targets[id_mask_1] * 1.0
        targets[id_mask_0] = targets[id_mask_0] * 0.0

    elif binary_value:
        targets = _prepare_binary_targets(targets, domain_targets, binary_value, domain = domain)
    if norm_target and not z_pid[0]:
        print("using norm_target", end='\r')
        mean_t, std_t = _move_to_device(norm_target, device)
        if mask_model is not None:
            mean_t, std_t = mean_t[0], std_t[0]
        targets = (targets - mean_t) / std_t
    if gev_target:
        targets = targets / 40000.0
    if targets.shape[-1] == 1:
        targets = targets.reshape(-1)

    inputs = _normalize_inputs(inputs, means_i, stds_i)
    eventwise_features = _normalize_inputs(eventwise_features, means_e, stds_e)
    inputs = _function_on_inputs(torch.nan_to_num, inputs, (), {'nan': 0.0, 'posinf': 0.0, 'neginf': 0.0})
    eventwise_features = torch.nan_to_num(eventwise_features, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0

    if domain_input:
        eventwise_features = torch.column_stack((eventwise_features, domain_targets.to(dtype=eventwise_features.dtype)))

    if z_pid[0]:
        origin_inputs = _normalize_inputs(origin_inputs, mean_t, std_t)
        origin_inputs = _function_on_inputs(torch.nan_to_num, origin_inputs, (), {'nan': 0.0, 'posinf': 0.0, 'neginf': 0.0})

        type_inputs = (_call_with_inputs(z_pid[1], inputs[0], eventwise_features) / 8.0, _call_with_inputs(z_pid[1], inputs[1], eventwise_features) / 8.0)
        isolation_inputs = (_call_with_inputs(z_pid[2], inputs[0], eventwise_features) / 8.0, _call_with_inputs(z_pid[2], inputs[1], eventwise_features) / 8.0)
        inputs = (torch.column_stack((origin_inputs[0], type_inputs[0], isolation_inputs[0])), torch.column_stack((origin_inputs[1], type_inputs[1], isolation_inputs[1])))

    elif mask_model is not None:
        mask = _predict_mask(mask_model, inputs, eventwise_features)
        if mask.sum() == 0:
            target_mask = _prepare_binary_targets(mask_targets, domain_targets, 2, domain = domain)
            target_mask = target_mask.to(dtype=torch.bool)
            if verbosity > 0 and batch_numbers[1] > 0:
                print(200*" ", end='\r')  # Clear the line
                print(f"Validation batch {batch_numbers[0]+1}/{batch_numbers[1]}: No valid samples from prediction, {target_mask.sum()} valid samples from truth", end='\r')
            skip_batch = True

        mask_targets = mask_targets[mask]
        if domain_targets is not None:
            domain_targets = domain_targets[mask]

        target_mask = _prepare_binary_targets(mask_targets, domain_targets, 2, domain = domain)
        target_mask = target_mask.to(dtype=torch.bool)
        if target_mask.sum() == 0:
            if verbosity > 0 and batch_numbers[1] > 0:
                print(200*" ", end='\r')  # Clear the line
                print(f"Validation batch {batch_numbers[0]+1}/{batch_numbers[1]}: No valid samples from truth, {mask.sum()} valid samples from prediction", end='\r')
            skip_batch = True

        targets = targets[mask][target_mask]

        if domain_targets is not None:
            domain_targets = domain_targets[target_mask]

        if isinstance(inputs, tuple):
            inputs = (inputs[0][mask][target_mask], inputs[1][mask][target_mask])
        else:
            inputs = inputs[mask][target_mask]
        eventwise_features = eventwise_features[mask][target_mask]

    return inputs, eventwise_features, targets, domain_targets, skip_batch

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
        binary_target : bool | float = False,
        z_target : bool = False,
        norm_target : tuple | None = None,
        gev_target : bool = False,
        domain : str = "bool",
        domain_adversary : torch.nn.Module | None = None,
        domain_loss_function : Callable | None = None,
        gradient_reversal : torch.nn.Module | None = None,
        domain_loss_weight : float = 1.0,
        mixed_target_loss_weight : float = 0.25,
        domain_input : bool = False,
        discard_remainder : bool = True,
        periodic_save_interval : int | None = None,
        periodic_save_path : str | None = None,
        mask_model : torch.nn.Module | None = None,
        variance_target : torch.nn.Module | None = None,
        z_pid : tuple[bool, torch.nn.Module, torch.nn.Module] = (False, None, None),
        verbosity : int = 1

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
    - discard_remainder (bool): Whether to discard rows that do not keep the wanted mixture ratio for the dataset.
    - z_target (bool): Whether to use a z target for training. This will need the correct y_fields to be set in the dataset. Default is False.
    - norm_target (tuple): A tuple containing the mean and standard deviation for target normalization. Default is None.

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
    if discard_remainder and not isinstance(discard_remainder, bool):
        raise TypeError("discard_remainder must be a boolean.")
    if z_pid[0] and norm_target is None:
        raise ValueError("norm_target must be provided when z_pid is True.")

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
    if isinstance(gradient_reversal, torch.nn.Module):
        gradient_reversal.to(device)
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
    if mask_model is not None:
        mask_model.to(device)
        mask_model.eval()
        for parameter in mask_model.parameters():
            parameter.requires_grad_(False)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        samples_seen = 0
        expected_batches = len(train_loader)
        for batch, data in enumerate(train_loader):
            inputs, eventwise_features, targets, domain_targets, skip_batch = _prepare_training_inputs(data, device, (means_i, means_e), (stds_i, stds_e), norm_target = norm_target, gev_target = gev_target, z_pid = z_pid, binary_value = binary_value, mask_model = mask_model, z_target = z_target, domain_input = domain_input, verbosity = verbosity, domain = domain, batch_numbers = (batch, expected_batches))
            if skip_batch:
                continue

            optimizer.zero_grad()
            outputs = _call_with_inputs(model, inputs, eventwise_features)
            targets = targets.to(dtype=outputs.dtype)

            if variance_target:
                base_values = _call_with_inputs(variance_target, inputs, eventwise_features)
                loss = _weighted_task_loss(
                    loss_function,
                    outputs,
                    targets,
                    domain_targets,
                    mixed_target_loss_weight,
                    variance_target_base_value=base_values
                )
            else:
                loss = _weighted_task_loss(
                    loss_function,
                    outputs,
                    targets,
                    domain_targets,
                    mixed_target_loss_weight,
                )

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
            if verbosity > 0:
                print(
                    f'Batch {batch + 1}/{expected_batches}, '
                    f'Loss: {average_loss:.4f}',
                    f'Target Mean/Std: {targets.mean().item():.4f}/{targets.std().item():.4f}',
                    end='\r',
                )

        if samples_seen == 0:
            raise ValueError("All samples were rejected.")
        #if samples_seen != len(train_loader.dataset) and not (
        #    discard_remainder or mask_model is not None
        #):
        #    raise RuntimeError(
        #        f"Training loader yielded {samples_seen} samples, "
        #        f"but its dataset contains {len(train_loader.dataset)}."
        #    )
        epoch_loss = running_loss / samples_seen

        val_loss = validate_model(
            model,
            loss_function,
            val_loader,
            device,
            means,
            stds,
            binary_target=binary_target,
            domain_adversary=domain_adversary,
            domain_loss_function=domain_loss_function,
            gradient_reversal=gradient_reversal,
            domain_loss_weight=domain_loss_weight,
            mixed_target_loss_weight=mixed_target_loss_weight,
            domain_input=domain_input,
            domain=domain,
            z_target=z_target,
            norm_target=norm_target,
            mask_model=mask_model,
            variance_target=variance_target,
            z_pid=z_pid,
            verbosity=verbosity,
            gev_target=gev_target,
        )
        val_losses.append(val_loss)
        train_losses.append(epoch_loss)
        if verbosity > 0:
            print(200*" ", end='\r')  # Clear the line
        if verbosity > -1:
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            if domain_adversary is not None:
                best_domain_adversary_state = copy.deepcopy(
                    domain_adversary.state_dict()
                )

        if periodic_save_interval is not None and periodic_save_path is not None:
            if (epoch + 1) % periodic_save_interval == 0:
                torch.save(best_model_state, periodic_save_path)
                if verbosity > -1:
                    print(f"Model saved at epoch {epoch + 1} to {periodic_save_path}")

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
        binary_target : bool | float = False,
        domain : str = "bool",
        domain_adversary : torch.nn.Module | None = None,
        domain_loss_function : Callable | None = None,
        gradient_reversal : torch.nn.Module | None = None,
        domain_loss_weight : float = 1.0,
        mixed_target_loss_weight : float = 0.25,
        domain_input : bool = False,
        z_target : bool = False,
        norm_target : tuple | None = None,
        gev_target : bool = False,
        mask_model : torch.nn.Module | None = None,
        variance_target : torch.nn.Module | None = None,
        z_pid : tuple[bool, torch.nn.Module, torch.nn.Module] = (False, None, None),
        verbosity : int = 1
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
    - binary_target (bool or float): Whether to convert the target values to a binary format. If a float is provided, it will be used as the value to consider as the positive class (1.0). Default is False.
    - domain_adversary (torch.nn.Module, optional): Domain classifier applied
      to the model embedding. Default is None.
    - domain_loss_function (callable, optional): Domain loss function.
    - gradient_reversal (torch.nn.Module, optional): Gradient-reversal module
      applied before the domain classifier.
    - domain_loss_weight (float): Weight of the domain loss. Default is 1.0.
    - mixed_target_loss_weight (float): Relative weight for the physics loss
      for domain target 1, whose targets are weak labels. Default is 0.25.
    - z_target (bool): Whether to use a z target for validation. This will need the correct y_fields to be set in the dataset. Default is False.

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
    if domain_adversary:
        domain_adversary.eval()
    val_loss = 0.0
    samples_seen = 0
    means_i, means_e = _move_to_device(means, device)
    stds_i, stds_e = _move_to_device(stds, device)
    binary_value = torch.as_tensor(binary_target, dtype=torch.float32, device=device) if binary_target is not False else None
    with torch.no_grad():
        expected_batches = len(val_loader)
        for batch, data in enumerate(val_loader):
            inputs, eventwise_features, targets, domain_targets, skip_batch = _prepare_training_inputs(data, device, (means_i, means_e), (stds_i, stds_e), norm_target = norm_target, gev_target = gev_target, z_pid = z_pid, binary_value = binary_value, mask_model = mask_model, z_target = z_target, domain_input = domain_input, verbosity = verbosity, domain = domain, batch_numbers = (batch, expected_batches))
            if skip_batch:
                continue
            
            outputs = _call_with_inputs(model, inputs, eventwise_features)
            targets = targets.to(dtype=outputs.dtype)

            if variance_target:
                base_values = _call_with_inputs(variance_target, inputs, eventwise_features)
                task_loss = _weighted_task_loss(
                    loss_function,
                    outputs,
                    targets,
                    domain_targets,
                    mixed_target_loss_weight,
                    variance_target_base_value=base_values
                )
            else:
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
                    eventwise_features,
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
            if verbosity > 0:
                print(200*" ", end='\r')  # Clear the line
                print(f"Validation batch {batch+1}/{expected_batches}: Loss = {loss.item():.4f}", end='\r')

    if samples_seen == 0:
        raise ValueError("All samples were rejected.")
    #if samples_seen != len(val_loader.dataset) and mask_model is None:
    #    raise RuntimeError(
    #        f"Validation loader yielded {samples_seen} samples, "
    #        f"but its dataset contains {len(val_loader.dataset)}."
    #    )
    if verbosity > 0:
        print(200*" ", end='\r')  # Clear the line
    val_loss /= samples_seen
    return val_loss

def predict(
    model : torch.nn.Module, 
    test_loader : torch.utils.data.DataLoader, 
    device : str | torch.device = 'cpu', 
    means : tuple | None = None, 
    stds : tuple | None = None, 
    norm_target : tuple | None = None,
    variance_target : bool = False,
    gev_target : bool = False,
    domain_input : bool | int = False,
    verbosity : int = 1
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
    
    Returns:
    - predictions (numpy.ndarray): An array of predicted values.
    """
    model.eval()
    predictions = []
    means_i, means_e = _move_to_device(means, device)
    stds_i, stds_e = _move_to_device(stds, device)
    dataset = getattr(test_loader, "dataset", None)
    previous_skip_target_reads = None
    if dataset is not None and hasattr(dataset, "__getitems__"):
        previous_skip_target_reads = getattr(dataset, "skip_target_reads", False)
        dataset.skip_target_reads = True
    try:
        with torch.inference_mode():
            expected_batches = len(test_loader)
            for batch, data in enumerate(test_loader):
                inputs, eventwise_features, _, domain = _tuple_unpack_to_device(data, device)
                inputs = _normalize_inputs(inputs, means_i, stds_i)
                eventwise_features = _normalize_inputs(eventwise_features, means_e, stds_e)

                inputs = _function_on_inputs(torch.nan_to_num, inputs, (), {'nan': 0.0, 'posinf': 0.0, 'neginf': 0.0})  # Replace NaN values with 0.0
                eventwise_features = torch.nan_to_num(eventwise_features, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0
                
                if domain_input or type(domain_input) == int:
                    if type(domain_input) == int:
                        domain = torch.ones_like(domain) * domain_input
                    eventwise_features = torch.column_stack((eventwise_features, domain.to(dtype=eventwise_features.dtype)))

                outputs = _call_with_inputs(model, inputs, eventwise_features)
                outputs = outputs.to(dtype=torch.float32)

                outputs = _prepare_predict_outputs(outputs, device, norm_target, variance_target, gev_target)

                predictions.append(outputs.cpu().numpy())
                if verbosity > 0:
                    print(200*" ", end='\r')  # Clear the line
                    print(f"Prediction batch {batch+1}/{expected_batches}: Predictions shape = {outputs.shape}", end='\r')
            if verbosity > 0:
                print(200*" ", end='\r')  # Clear the line
            if verbosity > -1:
                print("Prediction completed")
    finally:
        if dataset is not None and previous_skip_target_reads is not None:
            dataset.skip_target_reads = previous_skip_target_reads
    return np.concatenate(predictions)

def compressed_training_function(
    h5_files : list[str],
    dataset_class : type[Dataset],
    nn_model : type[torch.nn.Module],
    model_kwargs : dict,
    loss_function : torch.nn.Module,
    optimizer_class : type[torch.optim.Optimizer],
    num_epochs : int = 10,
    mix_files : list[str] | None = None,
    mixture_ratio : float = 0.5,
    exclude_features : list[str] | None = None,
    exclude_fields : dict[str, list[str]] | None = None,
    y_source : str = "egammas",
    y_field : str = "truthOrigin",
    fallback_source : str | None = None,
    fallback_field : list[str] | None = None,
    valid_fields : dict[str, list[str]] | None = None,
    batch_size : int = 4096,
    num_workers : int = 8,
    prefetch_factor : int = 2,
    persistent_workers : bool = True,
    pin_memory : bool = True,
    domain_adversary : bool = False,
    gradient_reversal_strength : float = 0.1,
    domain_loss_weight : float = 0.20,
    domain_loss_function : Callable | None = None,
    mixed_target_loss_weight : float = 0.20,
    domain_input : bool = False,
    binary_target : bool | float = False,
    validation_split : float = 0.25,
    use_old_model : torch.nn.Module | bool = False,
    discard_remainder : bool = True,
    domain : str = "bool",
    z_target : bool = False,
    norm_target : bool | None = None,
    gev_target : bool = False,
    periodic_save_interval : int | None = None,
    periodic_save_path : str | None = None,
    use_mask : bool | str = False,
    variance_target : torch.nn.Module | None = None,
    means_library_path : str | Path = "./means_library",
    z_pid : tuple[bool, str, str] = (False, "", ""),
    verbosity : int = 1
) -> tuple[torch.nn.Module, list, list]:
    train_size_egam = int((1 - validation_split) * len(h5_files))
    train_size_data = int((1 - validation_split) * len(mix_files))

    mean_dataset = dataset_classes.H5EgammaDataset(
        h5_files[0],
        #mix_files=mix_files[:train_size_data],
        #mixture_ratio=mixture_ratio,
        exclude_features = "truth",
        #exclude_fields = valid_fields,
        y_source = "tracks",
        fallback_source = "egammas",
        y_field = None,
    )

    means, stds = data_loading.load_means_stds(means_library_path, "egammas", exclude_features = exclude_features, exclude_fields = exclude_fields, dataset = mean_dataset)
    #means, stds = misc_functions.compute_mean_std(mean_dataset, sample_size = 300_000, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)

    if norm_target:
        if z_target:
            norm_target = (55_000, 40_000) #Temporary solution for z_target means/stds.
        else:   
            means_t, stds_t = data_loading.load_means_stds(means_library_path, "egammas", include_fields = {"egammas": y_field}, dataset = mean_dataset)
            norm_target = (means_t[0], stds_t[0])
            #norm_target = misc_functions.compute_target_mean_std(mean_dataset, sample_size = 1_000_000, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)

    train_dataset = dataset_class(
        h5_files[:train_size_egam],
        mix_files = mix_files[:train_size_data],
        exclude_features = exclude_features,
        exclude_fields = exclude_fields,
        y_source = y_source,
        y_field = y_field,
        fallback_source = fallback_source,
        fallback_field = fallback_field,
        mixture_ratio = mixture_ratio,
        valid_fields = valid_fields,
    )

    val_dataset = dataset_class(
        h5_files[train_size_egam:],
        mix_files = mix_files[train_size_data:],
        exclude_features = exclude_features,
        exclude_fields = exclude_fields,
        y_source = y_source,
        y_field = y_field,
        fallback_source = fallback_source,
        fallback_field = fallback_field,
        mixture_ratio = mixture_ratio,
        valid_fields = valid_fields,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    train_sampler = data_loading.ShuffledContiguousBatchSampler(len(train_dataset), batch_size=batch_size, dataset=train_dataset, discard_remainder=discard_remainder)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory=pin_memory)

    particle_fields = [field for field in train_dataset.fields if not field.startswith("eventwise_")]
    event_fields = [field for field in train_dataset.fields if field.startswith("eventwise_")]

    if domain_input:
        event_size = len(event_fields) + 1
    else:
        event_size = len(event_fields)

    if z_pid[0]:
        type_model = neural_models.ParticleOriginNetwork(
            len(particle_fields) // 2,
            event_size,
            **model_kwargs
        ).to(device)
        isolation_model = neural_models.ParticleOriginNetwork(
            len(particle_fields) // 2,
            event_size,
            **model_kwargs
        ).to(device)
        type_model.load_state_dict(torch.load(z_pid[1], map_location=device))
        isolation_model.load_state_dict(torch.load(z_pid[2], map_location=device))

        target_fields = [field for field in train_dataset.y_fields]
        particle_size = len(target_fields) - 3
        model_kwargs = {"particle_hidden_sizes": [64, 64, 16, 16, 8], "event_hidden_sizes" : [16, 8], "consolidation_hidden_sizes" : [16, 8], "output_size" : 1}
        embedding_size = model_kwargs["particle_hidden_sizes"][-1] * 2 + model_kwargs["event_hidden_sizes"][-1]
    elif dataset_class == dataset_classes.ZPairDataset:
        particle_size = len(particle_fields) // 2
        embedding_size = model_kwargs["particle_hidden_sizes"][-1] * 2 + model_kwargs["event_hidden_sizes"][-1]
    else:
        particle_size = len(particle_fields)
        embedding_size = model_kwargs["particle_hidden_sizes"][-1] + model_kwargs["event_hidden_sizes"][-1]

    if use_old_model == False:
        model = nn_model(
            particle_size,
            event_size,
            **model_kwargs
        ).to(device)
    else:
        model = use_old_model.to(device)



    if use_mask is not False and use_mask is not None:
        if not isinstance(use_mask, (str, bytes)):
            raise TypeError("use_mask must be a checkpoint path or False.")
        mask_kwargs = model_kwargs.copy()
        mask_kwargs["output_size"] = 1
        mask_model = nn_model(
            particle_size,
            event_size,
            **mask_kwargs
        ).to(device)
        mask_model.load_state_dict(torch.load(use_mask, map_location=device))
    else:
        mask_model = None

    if domain_adversary:
        domain_adversary = neural_models.DomainAdversary(
            embedding_size=embedding_size,
            domain_count=1,
        )
        gradient_reversal = neural_models.GradientReversal(strength = gradient_reversal_strength)
        optimizer = optimizer_class(list(model.parameters()) + list(domain_adversary.parameters()), lr=1e-5, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 1e-4)
    else:
        domain_adversary = None
        gradient_reversal = None
        optimizer = optimizer_class(list(model.parameters()), lr=1e-6, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 1e-4)
    train_losses, val_losses = train_model(model, 
                                           optimizer, 
                                           loss_function, 
                                           train_loader, 
                                           val_loader, 
                                           num_epochs=num_epochs, 
                                           device=device, 
                                           means=means, 
                                           stds=stds, 
                                           binary_target=binary_target, 
                                           domain_adversary = domain_adversary, 
                                           gradient_reversal = gradient_reversal,
                                           domain_loss_function = domain_loss_function, 
                                           domain_loss_weight = domain_loss_weight, 
                                           mixed_target_loss_weight = mixed_target_loss_weight,
                                           domain_input = domain_input,
                                           discard_remainder = discard_remainder,
                                           domain = domain,
                                           z_target = z_target,
                                           norm_target = norm_target,
                                           gev_target = gev_target,
                                           periodic_save_interval = periodic_save_interval,
                                           periodic_save_path = periodic_save_path,
                                           mask_model = mask_model,
                                           variance_target = variance_target,
                                           z_pid = (z_pid[0], type_model, isolation_model) if z_pid[0] else (False, None, None),
                                           verbosity = verbosity,
                                           )
    return model, train_losses, val_losses, means, stds, norm_target