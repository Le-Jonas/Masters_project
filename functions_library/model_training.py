import torch
import numpy as np

def train_model(model, optimizer, loss_function, train_loader, val_loader, num_epochs=10, device='cpu', means=None, stds=None, log_target=False):
    model.to(device)
    train_losses = []
    val_losses = []
    batch_losses = {}
    best_val_loss = float('inf')
    best_model_state = None
    means = means.to(device) if means is not None else None
    stds = stds.to(device) if stds is not None else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_losses[epoch] = []
        for batch, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if not torch.isfinite(targets).all():
                raise ValueError(f"Warning: Non-finite values detected in targets")
            if log_target:
                targets = torch.log1p(targets)  # Add 1 to avoid log(0)
            if means is not None and stds is not None:
                inputs = (inputs - means) / stds  # Normalize inputs using provided means and stds
            elif means is not None or stds is not None:
                raise ValueError("Both means and stds must be provided for normalization.")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0

            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.shape[-1] == 1:
                outputs = outputs.reshape(-1)  # Flatten outputs if they have a single output dimension
                targets = targets.reshape(-1)  # Flatten targets if they have a single output dimension
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            batch_losses[epoch].append(loss.item())

            print(f'Batch {batch+1}/{len(train_loader)}, Loss: {loss.item():.4f}', end='\r')

        epoch_loss = running_loss / len(train_loader.dataset)

        val_loss = validate_model(model, loss_function, val_loader, device, means, stds, log_target=log_target)
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

    return model, train_losses, val_losses, batch_losses



def validate_model(model, loss_function, val_loader, device='cpu', means=None, stds=None, log_target=False):
        # Validation phase
        model.eval()
        val_loss = 0.0
        means = means.to(device) if means is not None else None
        stds = stds.to(device) if stds is not None else None
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                if log_target:
                    targets = torch.log1p(targets)

                if not torch.isfinite(targets).all():
                    raise ValueError(f"Warning: Non-finite values detected in targets")
                if means is not None and stds is not None:
                    inputs = (inputs - means) / stds  # Normalize inputs using provided means and stds
                elif means is not None or stds is not None:
                    raise ValueError("Both means and stds must be provided for normalization.")
                inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0

                outputs = model(inputs)
                if outputs.shape[-1] == 1:
                    outputs = outputs.reshape(-1)  # Flatten outputs if they have a single output dimension
                    targets = targets.reshape(-1)  # Flatten targets if they have a single output dimension
                loss = loss_function(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        return val_loss

def predict(model, test_loader, device='cpu', means=None, stds=None, log_target=False):
    model.eval()
    predictions = []
    means = means.to(device) if means is not None else None
    stds = stds.to(device) if stds is not None else None
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            
            if means is not None and stds is not None:
                inputs = (inputs - means) / stds  # Normalize inputs using provided means and stds
            elif means is not None or stds is not None:
                raise ValueError("Both means and stds must be provided for normalization.")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN values with 0.0

            outputs = model(inputs)
            if log_target:
                outputs = torch.expm1(outputs)  # Apply inverse of log1p if log_target is True
            predictions.append(outputs.cpu().numpy())
        print("Prediction completed")
    return np.concatenate(predictions)

def training_setup(model,  train_loader, val_loader, optimizer=None, loss_function=None, num_epochs=10, device='cpu', means=None, stds=None):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if loss_function is None:
        loss_function = torch.nn.MSELoss()
    
    trained_model, val_losses, train_losses = train_model(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        means=means,
        stds=stds
    )
    
    return trained_model, val_losses, train_losses