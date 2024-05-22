import torch


def main(
    dataloader,
    model,
    loss_fn,
    optimizer,
    device,
    num_epoch,
    scheduler=None,
    chosen_scheduler=None,
):
    """
    Function for training
    """

    size = len(dataloader.dataset)
    model.train()
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    for batch, (X, Y) in enumerate(dataloader):

        X, Y = X.to(device), Y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, Y)

        # Don't keep the grads from previous steps
        optimizer.zero_grad()

        # Backprop
        loss.backward()
        optimizer.step()

        if scheduler and (
            chosen_scheduler == "cosine" or chosen_scheduler == "compose"
        ):
            scheduler.step(num_epoch + batch / len(dataloader))

        # Update the loss
        train_loss += loss.item()
        correct += (pred.argmax(1) == Y.argmax(1)).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Doing it per batch here makes a lot of sense since we are averageing over the batches
    train_loss /= num_batches
    # Here we do not really care how we divied it into batches
    train_acc = 100 * (correct / size)

    return train_acc, train_loss
