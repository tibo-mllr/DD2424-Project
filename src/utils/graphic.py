import matplotlib.pyplot as plt


def plot_graphs(trainAcc, testAcc, trainLoss, testLoss):
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
