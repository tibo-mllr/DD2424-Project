from torch.utils.data import DataLoader
from ..datasets import CIFAR10Custom


def main(batch_size=64):
    train_dataset = CIFAR10Custom(train=True)
    test_dataset = CIFAR10Custom(train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for X, Y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {Y.shape} {Y.dtype}")
        break

    return train_dataloader, test_dataloader
