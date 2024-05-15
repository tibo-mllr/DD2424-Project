"""
Building a basic classifier 

Will do it based on pytorch
"""

import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
from src.scripts import load_data, train, test
from src.models import BatchDropoutModel
from src.utils import WarmUpScheduler, plot_graphs


if __name__ == "__main__":
    # Load the data
    batch_size = 64
    train_dataloader, test_dataloader = load_data(batch_size)

    # Automatically set the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")

    # Choose the model
    model = BatchDropoutModel().to(device)

    # Define a loss and an optimizer
    loss_fn = nn.CrossEntropyLoss()
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
    # Learning rate warmup + cosine anealing can be combined
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

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # now do the stages of the training
        # I gues they must return the scheduler. hmm. Yes. They are obviusly only needed during the training process.
        train_Acc, train_Loss, scheduler = train(
            train_dataloader, model, loss_fn, optimizer, device, scheduler
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

    plot_graphs(trainAcc, testAcc, trainLoss, testLoss)
