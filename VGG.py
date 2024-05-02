
"""
Building a basic classifier 

Will do it based on pytorch
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import torchvision.transforms


import ssl
ssl._create_default_https_context = ssl._create_unverified_context



# Transforms the image and scales it (). 
image_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # Add any other transformations here (e.g., transforms.Normalize(mean, std))
])

# Function to one-hot encode the labels
def encode_one_hot(label, num_classes):
    return F.one_hot(torch.tensor(label), num_classes=num_classes).float()

class CIFAR10Custom(Dataset):
    def __init__(self, train=True, num_classes=10):
        self.dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=image_transforms)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)
    
    # This function
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y_one_hot = encode_one_hot(y, self.num_classes)
        return x, y_one_hot

# Example usage
train_dataset = CIFAR10Custom(train=True)
test_dataset  = CIFAR10Custom(train=False)


# Defining batc size
batch_size = 32

# Create data loaders. Shuffle the data
# We would need to create a custom collate function for one-hot encoding. 
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)



# in the datta loader can we then just dor 
# for images, labels in train_dataloader:
#   the images  will then eb 

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# setting the device
device= (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"using {device} device")



"""
Here is an implementation of the network itself. 
Evert model has it inherit from nn.module.
"""
class CIFAR10Model(nn.Module): # It inherits from Module
    def __init__(self): 
        super().__init__()
        # Start of with a convolutional layer followed by a rely and some dropout
        self.conv1 = nn.conv2d(3, 32, kernel_size=(3,3), stride = 1, padding = 1)
        self.act1 = nn.ReLU()
        # a bit of dropout.
        
        