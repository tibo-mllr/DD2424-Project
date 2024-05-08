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
import matplotlib.pyplot as plt

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# Transforms the image and scales it ().
image_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # Add any other transformations here (e.g., transforms.Normalize(mean, std))
    ]
)
# Function to one-hot encode the labels
def encode_one_hot(label, num_classes):
    return F.one_hot(torch.tensor(label), num_classes=num_classes).float()


class CIFAR10Custom(Dataset):
    def __init__(self, train=True, num_classes=10):
        self.dataset = datasets.CIFAR10(
            root="./data", train=train, download=True, transform=image_transforms
        )
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    # This function
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y_one_hot = encode_one_hot(y, self.num_classes)
        return x, y_one_hot


class CIFAR10Model(nn.Module):  # 

    def __init__(self):
        super(CIFAR10Model, self).__init__()
        # Start of with a convolutional layer followed by a rely and some dropout

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3), stride = 1, padding = "same"),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=1)
        ) 
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) 
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=1)
        )      
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 10), 
            nn.ReLU()  
        ) 
   #     self.fc3 = nn.Sequential(
            # 10 is the number of classses.
   #         nn.Linear(4096, 10)
   #     )


        # we have 10 classes which means that in the finals

    def forward(self, output): 
        # stack the outputs.
        out = self.layer1(output)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.fc(out)
        out = self.fc2(out)
     #   out = self.fc3(out)

        return out
        # might need some reshaping before the last layer.


"""
Starting off with the 
"""

class BaseLineModel(nn.Module):  # 

    def __init__(self):
        super(BaseLineModel, self).__init__()
        # Start of with a convolutional layer followed by a rely and some dropout


        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3), stride = 1, padding = "same"),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=1)
        ) 
      
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30752, 128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 10), 
            nn.ReLU() 
        )
    def forward(self, output): 
        # stack the outputs.
        out = self.layer1(output)
        out = self.layer2(out)
        out = self.fc(out)
        out = self.fc2(out)
     #   out = self.fc3(out)

        return out


class BaseLineDropout(nn.Module):
    def __in
    


def train(dataloader, model, loss_fn, optimizer, device):
    """
    Function for training.
    What to consider.
    """

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
      
      
        # zero grad before. 
        optimizer.zero_grad()

        # Backprop
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    """
    Check the performance aginst the test dataset.s
    """
    # should simply be the lenght of the epoch


    size = len(dataloader.dataset)
    # hmmm. Hur fungerar detta
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    # hmm, what is no_grad
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return correct*100, test_loss



def graphFunctions(testAcc, testLoss):

	# plot loss
    

    plt.plot(range(1,len(testAcc) + 1), testAcc, label = 'Test accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('The Accuracy')
    plt.title(f'The accuracy for the test data ')
    plt.legend()
    plt.savefig("Accura", bbox_inches='tight')

    plt.plot(range(1,len(testLoss) + 1), testLoss, label = 'Test loss')

    plt.xlabel('Epoch')
    plt.ylabel('The cross entropy loss')
    plt.title(f'The Loss for the test data ')
    plt.legend()
    plt.savefig("Loss", bbox_inches='tight')






if __name__ == "__main__":
    # Example usage
    train_dataset = CIFAR10Custom(train=True)
    test_dataset = CIFAR10Custom(train=False)

    # Defining batc size
    batch_size = 32

    # Create data loaders. Shuffle the data
    # We would need to create a custom collate function for one-hot encoding.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # in the datta loader can we then just dor
    # for images, labels in train_dataloader:
    #   the images  will then eb

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # setting the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"using {device} device")

    model = BaseLineModel().to(device)
    print(model)

    
    # Now lets define a loss and an optimizer
    loss_fn = nn.CrossEntropyLoss()
    # standard optimizer with ordinary optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9 )

    # Doing training over several epochs should be done.
    

    testAcc = []
    testLoss = []
    epochs = 2
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # now do the stages of the training
        train(train_dataloader, model, loss_fn, optimizer, device)
        test_Accuracy, test_Loss = test(test_dataloader, model, loss_fn, device)
        testAcc.append(test_Accuracy)
        testLoss.append(test_Loss)
    print("Done")

    print(testAcc)
    print(testLoss)

    graphFunctions(testAcc, testLoss)