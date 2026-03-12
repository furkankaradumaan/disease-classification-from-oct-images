import torch
from pathlib import Path
import json
from typing import Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def save_model_state_dict(
    state_dict,
    save_dir: str = "models",
    filename: str = "model.pth"
):
    """Save the model state_dict into a file.
    Args:
        state_dict: model.state_dict()
        save_dir: model directory
        filename: file name
    """

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    full_path = save_path / filename
    torch.save(state_dict, full_path)

    print(f"[INFO] Model saved to: {full_path}")


def _to_serializable(obj: Any) -> Any:
    """
    JSON'a uygun hale çevirir.
    """
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    return obj

def save_training_metrics(
    metrics: dict[str, Any],
    save_dir: str = "experiments",
    filename: str = "training_metrics.json",
) -> Path:
    """Daves the train/validation metrics into a file.
    Expected keys:
        - train_loss
        - train_acc
        - val_loss
        - val_acc
        - val_classification_report (istersen)

    Args:
        metrics: kaydedilecek metrik dict'i
        save_dir: kayıt klasörü
        filename: dosya adı

    Returns:
        Kaydedilen dosyanın yolu
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    full_path = save_path / filename

    serializable_metrics = _to_serializable(metrics)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Training metrics saved to: {full_path}")
    return full_path

def save_test_metrics(
    metrics: dict[str, Any],
    save_dir: str = "experiments",
    filename: str = "test_metrics.json",
) -> Path:
    """
    Test metriklerini kaydeder.

    Beklenen örnek anahtarlar:
        - loss
        - acc
        - classification_report
        - confusion_matrix

    Args:
        metrics: kaydedilecek test metrik dict'i
        save_dir: kayıt klasörü
        filename: dosya adı

    Returns:
        Kaydedilen dosyanın yolu
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    full_path = save_path / filename

    serializable_metrics = _to_serializable(metrics)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Test metrics saved to: {full_path}")
    return full_path

def save_loss_acc_figures(metrics: dict[str, Any],
                          save_dir: str) -> None:
    """Save the figures for loss and accuracy plots.
    This function creates a figure for loss and accuracy metrics.
    seperately. Then saves each figure into a file in save_dir.
    
    Args:
        metrics : A dictionary containing keys train_loss, train_acc, val_loss, val_acc
        save_dir: The name of the directory to save the figures. (e.g figures/convnext_tiny/)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    loss_path = save_path / "loss.png"
    acc_path = save_path / "acc.png"

    # Plot the train and val loss and save it.
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]
    
    plt.figure(figsize=(15, 8))
    plt.title("Loss Functions")
    plt.plot(train_loss, color="blue", label="Train Loss")
    plt.plot(val_loss, color="red", label="Validation Loss")
    plt.legend()
    plt.savefig(str(loss_path.resolve()))
        
    # Plot accuracy and save it.
    train_acc = metrics["train_acc"]
    val_acc = metrics["val_acc"]

    plt.figure(figsize=(15, 8))
    plt.title("Loss Functions")
    plt.plot(train_acc, color="blue", label="Train Acc")
    plt.plot(val_acc, color="red", label="Validationa Acc")
    plt.legend()
    plt.savefig(str(acc_path.resolve()))

def save_confusion_matrix(cm, save_dir):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    image_path = save_path / "confusion_matrix.png"

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(str(image_path.resolve()))

def load_metric(
    save_dir: str = "experiments",
    filename: str = "metrics.json"
) -> dict[str, Any]:
    """
    Kaydedilmiş metrics dosyasını yükler.
    """
    full_path = Path(save_dir) / filename

    with open(full_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics
