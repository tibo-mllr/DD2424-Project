import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torch.nn.functional as F
import torchvision.transforms

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class CIFAR10Custom(Dataset):
    validation_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor()
            # Add any other transformations here (e.g., transforms.Normalize(mean, std))
        ]
    )

    # For Task 6 Normalize the data
    # It is quite convenient since the mean and standard dev is known for CIFAR-10
    # If we want to compute do we need to use dataloaders.
    normalized_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),  # Convert images to tensor and scale to [0, 1]
            # Should do the trick.s
            torchvision.transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )

    image_transforms = torchvision.transforms.Compose(
        [
            # should we transform to first. Investigate the order of to tensor and the horizontal flip.
            # This might not be right, or not.
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1)
            ),  # degrees = 0 to deactivate rotation.
            torchvision.transforms.ToTensor(),
        ]
    )

    @staticmethod
    def encode_one_hot(label, num_classes):
        return F.one_hot(torch.tensor(label), num_classes=num_classes).float()

    def __init__(self, train=False, num_classes=10):
        self.dataset = datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=self.validation_transforms,
        )

        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    # This function
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y_one_hot = CIFAR10Custom.encode_one_hot(y, self.num_classes)
        return x, y_one_hot
