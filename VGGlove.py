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
import torch.optim.lr_scheduler as lr_scheduler

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# Transforms the image and scales it ()
#
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


# We only need to apply these
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


# Function to one-hot encode the labels
def encode_one_hot(label, num_classes):
    return F.one_hot(torch.tensor(label), num_classes=num_classes).float()


class CIFAR10Custom(Dataset):
    def __init__(self, train=True, num_classes=10):

        # Task 4
        # If it is the training dataset should the modifications be applied. Not to validation version.
        # if train == True:
        self.dataset = datasets.CIFAR10(
            root="./data", train=train, download=True, transform=validation_transforms
        )
        #  else:
        #     self.dataset = datasets.CIFAR10(
        #      root="./data", train=train, download=True, transform=validation_transforms
        #    )

        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    # This function
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y_one_hot = encode_one_hot(y, self.num_classes)
        return x, y_one_hot


"""
Starting off with the baseLineModel
"""


class BaseLineModel(nn.Module):  #

    def __init__(self):
        super(BaseLineModel, self).__init__()
        # Start of with a convolutional layer followed by a rely and some dropout

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"), nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(2048, 128), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(128, 10)
            # removed a relu down here. May or may not be necessary.
        )
        self._initialize_weights()

    # Ok this was apparently the problem. How does it work.

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # enligt instruktioner skall det vara He init för bägge, He = kaiming
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.xavier_normal_(m.weight)
                # not sure about this one.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    """            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    """

    def forward(self, output):
        out = self.layer1(output)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.fc(out)
        out = self.fc2(out)

        return out


class BaseLineModelDropout(nn.Module):  #

    def __init__(self):
        super(BaseLineModelDropout, self).__init__()
        # Start of with a convolutional layer followed by a rely and some dropout

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"), nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 10)
            # removed a relu down here. May or may not be necessary.
        )
        self._initialize_weights()

    # Ok this was apparently the problem. How does it work.

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # enligt instruktioner skall det vara He init för bägge, He = kaiming
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.xavier_normal_(m.weight)
                # not sure about this one.
                # should they have bias or not.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    """            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    """

    def forward(self, output):
        out = self.layer1(output)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.fc(out)
        out = self.fc2(out)

        return out


class EveryThingModel(nn.Module):  #

    def __init__(self):
        super(EveryThingModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            # should Apparently be 1D here
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 10)
            # removed a relu down here. May or may not be necessary.
        )
        self._initialize_weights()

    # Ok this was apparently the problem. How does it work.

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # enligt instruktioner skall det vara He init för bägge, He = kaiming
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.xavier_normal_(m.weight)
                # not sure about this one.
                # should they have bias or not.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    """            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    """

    def forward(self, output):
        out = self.layer1(output)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.fc(out)
        out = self.fc2(out)

        return out


"""
Task 9 part 1
Start with batchNorm then drop out
Unlike for the everything model is the probablity  the same for dropout. 
"""


class BatchDropout(nn.Module):  #

    def __init__(self):
        super(BatchDropout, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            # should Apparently be 1D here
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc2 = nn.Sequential(nn.Linear(128, 10))
        self._initialize_weights()

    # Ok this was apparently the problem. How does it work.

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # enligt instruktioner skall det vara He init för bägge, He = kaiming
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.xavier_normal_(m.weight)
                # not sure about this one.
                # should they have bias or not.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, output):
        out = self.layer1(output)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.fc(out)
        out = self.fc2(out)
        return out


"""
Task 9 part two. 
Start with dropout then batch. Keep the same probability over all dropout. 
"""


class DropoutBatch(nn.Module):  #

    def __init__(self):
        super(DropoutBatch, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            # should Apparently be 1D here
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.fc2 = nn.Sequential(nn.Linear(128, 10))
        self._initialize_weights()

    # Ok this was apparently the problem. How does it work.

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # could maybe customize the mode if we wanted to.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # enligt instruktioner skall det vara He init för bägge, He = kaiming
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, output):
        out = self.layer1(output)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.fc(out)
        out = self.fc2(out)
        return out


def train(
    dataloader, model, loss_fn, optimizer, device, num_epoch=None, scheduler=None
):
    """
    Function for training.
    What to consider.
    """

    size = len(dataloader.dataset)
    model.train()
    num_batches = len(dataloader)

    train_loss, correct = 0, 0
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

        # part of task 8 with the CosineAnnealingWarmRestarts
        # use the definition from the
        # scheduler.step(num_epoch + batch / len(dataloader))

        # some changes to ret
        train_loss += loss.item()
        # should bet the same here
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        if batch % 100 == 0:

            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Doing it per batch here makes a lot of sense since we are averageing over the batches
    train_loss /= num_batches
    # here we do not really care how we we divied it into batchs.
    train_acc = 100 * (correct / size)

    return train_acc, train_loss, scheduler


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
    return correct * 100, test_loss


def graphFunctions(trainAcc, testAcc, trainLoss, testLoss):

    # plot loss

    plt.plot(range(1, len(testAcc) + 1), testAcc, label="Test accuracy")
    plt.plot(range(1, len(trainAcc) + 1), trainAcc, label="Train accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("The Accuracy")
    plt.title(f"The accuracy for the test data ")
    plt.legend()
    plt.savefig("Accura", bbox_inches="tight")

    plt.clf()

    plt.plot(range(1, len(testLoss) + 1), testLoss, label="Test loss")
    plt.plot(range(1, len(trainLoss) + 1), trainLoss, label="Train loss")

    plt.xlabel("Epoch")
    plt.ylabel("The cross entropy loss")
    plt.title(f"The Loss for the test data ")
    plt.legend()
    plt.savefig("Loss", bbox_inches="tight")


# to get warmup + cosine anneling to work do we need to define a custom class for the learning rate that inhets from lr_scheduler._LRScheduler
# Fairly certain it works as intended


class WarmUpScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, initial_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # the learning
            return [
                self.initial_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr for base_lr in self.base_lrs]


if __name__ == "__main__":
    train_dataset = CIFAR10Custom(train=True)
    test_dataset = CIFAR10Custom(train=False)

    batch_size = 64

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    # Change to model if
    model = BatchDropout().to(device)
    print(model)

    # Now lets define a loss and an optimizer
    loss_fn = nn.CrossEntropyLoss()
    # standard optimizer with ordinary optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # If we want to use L2 normalization is the following step used.
    #  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay= 0.001)

    # Task 7. If we want a baseline model, but with adam instead
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # if we want adamW instead.
    # AdamW is adam with weight decay. Can experiment with the weight.
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay = 0.001)

    """
    Task 8 with differnt learning rate schedulers. I guess we can try all of the tree mentioned. 

    Warm up = very small learning rate intially that then increases fast . 
    """

    # learning rate warmup + cosine anealing.

    # Can be combined
    warmupSteps = 5
    epochs = 100
    warmup_scheduler = WarmUpScheduler(optimizer, warmupSteps, initial_lr=1e-3)
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmupSteps
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmupSteps],
    )
    # Then use scheduler.step after an epoch.

    # cosine annealing with restart is built into pyTorch.
    # The parameters are.  T_0 = 10 makes sense. eta_min too.
    # Not sure how important the initial learning rate is.
    #   optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    #   scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult = 1, eta_min = 0.001)

    # step decay is also built in. The step_size should probably be quite small, but the initial learning rate should be big.
    # gamma should be around 0.1
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.1)

    # Could make a final task with adam, all of the regularization and the whole shabang over 400+ epochs and see if we can get 90+% accuracy. Is certainty within the realm of possibility.

    # record both the test version and the training version.s
    testAcc = []
    trainAcc = []
    testLoss = []
    trainLoss = []
    #
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # now do the stages of the training
        # I gues they must return the scheduler. hmm. Yes. They are obviusly only needed during the training process.
        train_Acc, train_Loss, scheduler = train(
            train_dataloader, model, loss_fn, optimizer, device, t, scheduler
        )
        test_Accuracy, test_Loss = test(test_dataloader, model, loss_fn, device)
        testAcc.append(test_Accuracy)
        testLoss.append(test_Loss)
        trainAcc.append(train_Acc)
        trainLoss.append(train_Loss)

        # Part of task 8 when useing the step learning rate or cosince scheduler with warmup since it is updated per batch.
        scheduler.step()
        print(f"Epoch {t+1}, LR: {scheduler.get_last_lr()[0]}")

    print("Done")

    print(testAcc)
    print(testLoss)

    graphFunctions(trainAcc, testAcc, trainLoss, testLoss)
