import argparse
from pathlib import Path

import torch
from torchvision import transforms
from torch import nn

from src import engine
from src import data_setup
from src import model_builder
from src import save
from src import evaluate

import yaml

parser = argparse.ArgumentParser(prog="Disease Classification From OCT Images")
parser.add_argument("--config_path", "-cp", type=Path, default="config.yaml")
args = parser.parse_args()

config_path: Path = args.config_path

# Load configurations
with config_path.open(mode="r") as f:
    config = yaml.safe_load(f)

# Setup device agnostic code.
device = "cuda" if torch.cuda.is_available() else "cpu"

# define transforms
train_transform = transforms.Compose([
    transforms.Resize((224 ,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224 ,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading data and creating dataloaders...")
dataset = data_setup.get_dataset(config["dataset"]["dataset_name"])
train_dataloader, val_dataloader, test_dataloader = data_setup.create_dataloaders(
        dataset, train_transform, test_transform, config["training"]["batch_size"])


# Create a model
print("Creating deep learning model...")
model = model_builder.load_model(config["model"]["model_name"],
                                 config["dataset"]["num_classes"],
                                 config["dataset"]["inchans"],
                                 device=device)

# Create loss function and optimizer
loss_fn_name = config["training"]["loss_fn"]
if loss_fn_name == "cross-entropy-loss":
    loss_fn = nn.CrossEntropyLoss()
else:
    raise ValueError(f"Unsupported loss function name: {loss_fn_name}")

optimizer_name = config["training"]["optimizer"]
learning_rate = config["training"]["learning_rate"]

if optimizer_name == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_name == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
elif optimizer_name == "rmsprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
else:
    raise ValueError(f"Invalid optimizer name: {optimizer_name}")

print("Training the model...")
# Train the model
results = engine.train_model(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=config["training"]["num_epochs"],
    device=device,
)

# save the best model seen
best_model_state_dict = results["best_model_state_dict"]
save.save_model_state_dict(best_model_state_dict,
                           save_dir=config["output"]["model_save_dir"],
                           filename=config["output"]["model_save_name"])
save.save_loss_acc_figures(results["history"], config["output"]["train_output_save_dir"])

# Evaluate the best model
print("Evaluating the best model...")
evaluation_result = evaluate.evaluate_model(results["model"], test_dataloader, loss_fn, device)

print(f"Evalution Loss: {evaluation_result['loss']:.4f}")
print(f"Evalution Accuracy: {evaluation_result['accuracy']:.2f}")
save.save_confusion_matrix(evaluation_result['confusion_matrix'], config["output"]["test_output_save_dir"])
