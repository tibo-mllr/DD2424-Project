"""
Building a basic classifier 

Will do it based on pytorch
"""

import argparse
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
from src.scripts import load_data, train, test
from src.models import (
    BaseLineModel,
    BaseLineDropoutModel,
    BatchDropoutModel,
    DropoutBatchModel,
    CompleteModel,
)
from src.utils import WarmUpScheduler, plot_graphs


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        metavar="",
        help="number of total epochs to run (default: 100)",
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        metavar="",
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="baseline",
        type=str,
        choices=[
            "baseline",
            "baseline-dropout",
            "batch-dropout",
            "dropout-batch",
            "complete",
        ],
        metavar="",
        help="model to use; options: [baseline, baseline-dropout, batch-dropout, dropout-batch, complete] (default: baseline)",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help="normalize the data",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        default="SGD",
        type=str,
        choices=["sgd", "decay", "adam", "qdamW"],
        metavar="",
        help="optimizer to use; options: [sgd, decay, adam, adamW] (default: sgd)",
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        default="none",
        type=str,
        choices=["cosine", "step", "compose", "none"],
        metavar="",
        help="scheduler to use; options: [cosine, step, compose, none] (default: none)",
    )
    parser.add_argument(
        "-t",
        "--transform",
        action="store_true",
        help="apply transformations to the data",
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        default="none",
        type=str,
        choices=["cosine", "step", "compose", "none"],
        metavar="",
        help="scheduler to use; options: [cosine, step, compose, none] (default: none)",
    )
    parser.add_argument(
        "-t",
        "--transform",
        action="store_true",
        help="apply transformations to the data",
    )
    parser.add_argument(
        "-w",
        "--warmup-steps",
        default=5,
        type=int,
        metavar="",
        help="number of warmup steps (default: 5)",
    )

    args = parser.parse_args()

    batch_size = args.batch_size
    chosen_model = args.model
    chosen_optimizer = args.optimizer
    chosen_scheduler = args.scheduler
    epochs = args.epochs
    lr = args.lr
    normalize = args.normalize
    transform = args.transform
    warmup_steps = args.warmup_steps

    return (
        batch_size,
        epochs,
        lr,
        chosen_model,
        chosen_optimizer,
        chosen_scheduler,
        normalize,
        transform,
        warmup_steps,
    )


def main(
    batch_size,
    epochs,
    lr,
    chosen_model,
    chosen_optimizer,
    chosen_scheduler,
    normalize,
    transform,
    warmup_steps,
):
    # Load the data
    train_dataloader, test_dataloader = load_data(batch_size, normalize, transform)

    # Automatically set the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")

    # Choose the model
    models = {
        "baseline": BaseLineModel,
        "baseline-dropout": BaseLineDropoutModel,
        "batch-dropout": BatchDropoutModel,
        "dropout-batch": DropoutBatchModel,
        "complete": CompleteModel,
    }
    model = models[chosen_model]().to(device)

    # Define a loss and an optimizer
    loss_fn = nn.CrossEntropyLoss()
    if chosen_optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif chosen_optimizer == "decay":
        # Decay is the same as SGD with L2 normalization
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001
        )
    elif chosen_optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif chosen_optimizer == "adamW":
        # AdamW is adam with weight decay. Can experiment with the weight.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    else:
        raise ValueError("Unknown optimizer")

    # Task 8 with differnt learning rate schedulers. I guess we can try all of the tree mentioned.
    # Warm up = very small learning rate intially that then increases fast .
    if chosen_optimizer == "compose":
        # Learning rate warmup + cosine anealing can be combined
        warmup_scheduler = WarmUpScheduler(optimizer, warmup_steps, initial_lr=lr)
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_steps
        )
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    elif chosen_scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=0.001
        )
    elif chosen_scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif chosen_scheduler == "none":
        scheduler = None
    else:
        raise ValueError("Unknown scheduler")

    # Could make a final task with adam, all of the regularization and the whole shabang over 400+ epochs and see if we can get 90+% accuracy. Is certainty within the realm of possibility.

    # record both the test version and the training version.s
    testAcc = []
    trainAcc = []
    testLoss = []
    trainLoss = []

    print(
        "===============================\n",
        f"Training model {models[chosen_model].__name__} with optimizer {optimizer.__class__.__name__},",
        f"learning rate {lr} and batch size {batch_size} on {epochs} epochs.",
        "\n===============================",
    )

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # now do the stages of the training
        # I gues they must return the scheduler. hmm. Yes. They are obviusly only needed during the training process.
        train_Acc, train_Loss = train(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            device,
            t,
            scheduler,
            chosen_scheduler,
        )
        test_Accuracy, test_Loss = test(test_dataloader, model, loss_fn, device)
        testAcc.append(test_Accuracy)
        testLoss.append(test_Loss)
        trainAcc.append(train_Acc)
        trainLoss.append(train_Loss)

        # Part of task 8 when useing the step learning rate or cosince scheduler with warmup since it is updated per batch.
        if scheduler:
            scheduler.step()
            print(f"Epoch {t+1}, LR: {scheduler.get_last_lr()[0]}")

    print("Done")

    print(testAcc)
    print(testLoss)

    plot_graphs(trainAcc, testAcc, trainLoss, testLoss)


if __name__ == "__main__":
    (
        batch_size,
        epochs,
        lr,
        chosen_model,
        chosen_optimizer,
        chosen_scheduler,
        normalize,
        transform,
        warmup_steps,
    ) = get_args()

    main(
        batch_size,
        epochs,
        lr,
        chosen_model,
        chosen_optimizer,
        chosen_scheduler,
        normalize,
        transform,
        warmup_steps,
    )
