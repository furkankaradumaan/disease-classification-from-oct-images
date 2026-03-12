"""
This file contains functions to evaluate the model performance and
to generate metrics.
"""

import torch
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, dataloader, criterion, device, class_names=None):
    """
    Trained classification modelini evaluate eder.

    Args:
        model: Eğitilmiş PyTorch model
        dataloader: Test dataloader
        criterion: Loss function (örn: nn.CrossEntropyLoss())
        device: "cuda" veya "cpu"
        class_names: Sınıf isimleri listesi (opsiyonel)

    Returns:
        results (dict):
            {
                "loss": float,
                "accuracy": float,
                "confusion_matrix": np.ndarray,
                "classification_report": dict,
                "y_true": list,
                "y_pred": list
            }
    """
    model.eval()

    running_loss = 0.0
    total_samples = 0
    correct = 0

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total_samples
    accuracy = correct / total_samples

    cm = confusion_matrix(all_labels, all_preds)

    if class_names is not None:
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    else:
        report = classification_report(
            all_labels,
            all_preds,
            output_dict=True,
            zero_division=0,
        )

    results = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": all_labels,
        "y_pred": all_preds
    }

    return results
