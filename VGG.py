"""
Building a basic classifier 

Will do it based on pytorch
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
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



"""
Here is an implementation of the network itself. 
Evert model has it inherit from nn.module.
"""
class CIFAR10Model(nn.Module): # It inherits from Module
    def __init__(self): 
        super().__init__()
        # Start of with a convolutional layer followed by a rely and some dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride = 1, padding = 1)
        self.act1 = nn.ReLU()
        

        
        
if __name__=="__main__":
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


    
        
model = CIFAR10Model().to(device)
print(model)


"""
Now lets define a loss and an optimizer
"""
loss_fn = nn.CrossEntropyLoss()
# standard optimizer with ordinary optimizer.
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)


"""
Function for training. 
What to consider. 
"""
def train(dataloader, model, loss_fn, optimizer): 
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #compute prediction error
        pred = model(X)
        loss = loss_fn(pred,y)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

"""
Check the performance aginst the test dataset.s
"""
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    # hmmm. Hur fungerar detta
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    # hmm, what is no_grad
    with torch.no_grad():
        for X,y in dataloader: 
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == 1).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


"""
Doing training over several epochs should be done. 
"""
    
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    # now do the stages of the training
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done")
