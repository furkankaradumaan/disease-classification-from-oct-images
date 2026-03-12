from torch.utils.data import Dataset

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, image_key="image", label_key="label", transform=None):
        self.dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample[self.image_key]
        label = sample[self.label_key]

        if self.transform:
            image = self.transform(image)

        return image, label
