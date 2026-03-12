from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from src.HFDatasetWrapper import HFDatasetWrapper

def get_dataset(
    dataset_name: str,
    ) -> DatasetDict:
    """Loads the datasets containing train, validation, test splits.
    This function loads the dataset from Huggingface,
    then returns it.
    
    Args:
        dataset_name: The name of the dataset that will be downloaded.
    Returns:
        A DatasetDict containing train, validation, and test splits
    """
    dataset = load_dataset(
        dataset_name,
    )
    
    return dataset

def create_datasets(dataset: DatasetDict,
                    train_transform: transforms.Compose,
                    test_transform: transforms.Compose) -> tuple[Dataset, Dataset, Dataset]:
    """Create train, validation and test datasets.
    This function creates PyTorch datasets for train, validation, and test splits in given dataset.

    Args:
        dataset: The DatasetDict contains train, validation, and test splits.
        train_transform: Transforms to apply to train data.
        test_transform: Transforms to apply to validation and test data.

    Returns:
        A tuple of three PyTorch Dataset objects in the form (train_dataset, validation_dataset, test_dataset)
    """
    train_data, val_data, test_data = dataset["train"], dataset["validation"], dataset["test"]

    train_dataset = HFDatasetWrapper(train_data, transform=train_transform)
    val_dataset = HFDatasetWrapper(val_data, transform=test_transform)
    test_dataset = HFDatasetWrapper(test_data, transform=test_transform)

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(dataset: DatasetDict,
                       train_transform: transforms.Compose,
                       test_transform: transforms.Compose,
                       batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, validation, and test dataloaders.
    This function creates PyTorch datasets for train, val, and test splits in dataset.
    Then it creates PyToch DataLoaders for train, val, and test.

    Args:
        dataset: dataset that contains the train, validation, and test splits.
        train_transform: transforms to apply to train data
        test_transform: transforms to apply to train data
        batch_size: An integer that indicates the batch size.
    Returns:
        A tuple of three PyTorch DataLoaders in the form (train_dataloader, val_dataloader, test_dataloader)
    """
    train_dataset, val_dataset, test_dataset = create_datasets(dataset, train_transform, test_transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader
