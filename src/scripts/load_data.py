from torch.utils.data import DataLoader
from ..datasets import CIFAR10Custom


def main(batch_size=64, normalize=False, transform=False):

    train_dataset = CIFAR10Custom(train=True, normalize=normalize, transform=transform)
    test_dataset = CIFAR10Custom(train=False, normalize=normalize, transform=False)


    # Can add workers here.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers= 1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 1)

    for X, Y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {Y.shape} {Y.dtype}")
        break

    return train_dataloader, test_dataloader
