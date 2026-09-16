import torch
import numpy as np

def train_model_from_hdf5(model, optimizer, loss_function, train_loader, val_loader, num_epochs=10, device='cpu', means=None, stds=None):
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

def train_model_from_csv(model, optimizer, loss_function, train_loader, val_loader, num_epochs=10, device='cpu', means=None, stds=None):
    model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

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