"""
This file contains functions to create a PyTorch model
using timm library.
"""
import timm
import torch

def load_model(
    model_name: str,
    num_classes: int,
    inchans: int,
    device: str|None
):
    """
    timm kütüphanesinden pretrained model yükler ve döndürür.

    Args:
        model_name (str): timm model adı (örn: "resnet18", "vit_base_patch16_224")
        num_classes (int): classifier head'i değiştirmek için
        device (str): "cuda" veya "cpu"

    Returns:
        torch.nn.Module
    """

    model = timm.create_model(
        model_name,
        num_classes=num_classes,
        in_chans=inchans,
        pretrained=False,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    return model
