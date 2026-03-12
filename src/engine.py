"""
This file contains functions to train the model.
"""
import copy
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix

from tqdm.auto import tqdm # For progress bar

def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> dict[str, float]:
    """Train the model for one epoch.

    This function trains the model for one epoch, calculates the loss
    for the epoch, calculates the accuracy for the epochs.

    Args:
        model: model to train
        dataloader: PyTorch DataLoader that the model will trained on.
        loss_fn: PyTorch loss function to calculate loss
        optimizer: PyTorch Optimizer to update weights
        device: PyTorch device.

    Returns:
        Returns a dictionary of float values with keys 'loss' and 'acc'.
        'loss' represents the average loss for epoch and 'acc' represents the accuracy.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device, dtype=torch.long)

        optimizer.zero_grad()

        logits = model(X)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == y).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


@torch.inference_mode()
def validation_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> dict[str, Any]:
    """Validates the model for one epoch.
    This function is used to validate the data on unseen data (data other than training data).
    It calculates the loss, accuracy, confusion matrix, and classification report.

    Args:
        model: model to validate.
        dataloader: PyTorch dataloader.
        loss_fn: Loss function to calculate loss
        device: PyTorch device

    Returns:
        A dictionary of 4 keys: 'loss', 'acc', 'classification_report', 'confusion_matrix'.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device, dtype=torch.long)

        logits = model(X)
        loss = loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == y).sum().item()
        total_samples += batch_size

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    report = classification_report(
        all_targets.numpy(),
        all_preds.numpy(),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_targets.numpy(), all_preds.numpy())

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
        "classification_report": report,
        "confusion_matrix": cm
    }


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | str,
    monitor: str = "val_loss",
    load_best_at_end: bool = True,
) -> dict[str, Any]: 
    """Train the model for given number of epochs.
    This function uses train_step and validation_step functions to train and validate
    the model for a given number of epochs.

    Args:
        model: model to train
        train_dataloader: PyTorch DataLoader that the model will be trained on.
        val_dataloader: PyTorch DataLoader to validate the model on unseen data.
        loss_fn: PyTorch loss function
        optimizer: PyTorch optimizer to update weights.
        epochs: Integer represents the number of epochs.
        device: The PyTorch device that the operations are performed on (e.g. "cuda")
        monitor: A metric that will be used to determine the best model.
        load_best_at_end: If you set this true, given model's parameters automatically converted to the best model parameters.

    Returns:
        A dictionary that has the keys:
            train_loss               : A list that stores the train loss values for each epoch
            train_acc                : A list that stores the train accuracy values for each epoch
            val_loss                 : A list that stores the validation loss values for each epoch
            val_acc                  : A list that stores the validation accuracy values for each epoch
            val_classification_report: A list of classification reports produces for each epoch in validation step.
    """
    model = model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_classification_report": [],
    }

    best_model_state_dict = None
    best_epoch = -1

    if "loss" in monitor:
        best_score = float("inf")
    else:
        best_score = float("-inf")

    for epoch in tqdm(range(epochs)):

        train_results = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_results = validation_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_loss"].append(train_results["loss"])
        history["train_acc"].append(train_results["acc"])
        history["val_loss"].append(val_results["loss"])
        history["val_acc"].append(val_results["acc"])
        history["val_classification_report"].append(
            val_results["classification_report"]
        )

        current_score = {
            "train_loss": train_results["loss"],
            "train_acc": train_results["acc"],
            "val_loss": val_results["loss"],
            "val_acc": val_results["acc"],
        }[monitor]

        if "loss" in monitor:
            is_better = current_score < best_score
        else:
            is_better = current_score > best_score

        if is_better:
            best_score = current_score
            best_epoch = epoch + 1
            best_model_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"train_loss: {train_results['loss']:.4f} | "
            f"train_acc: {train_results['acc']:.4f} | "
            f"val_loss: {val_results['loss']:.4f} | "
            f"val_acc: {val_results['acc']:.4f}"
        )

    if load_best_at_end and best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)

    return {
        "model": model,
        "best_model_state_dict": best_model_state_dict,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
    }
