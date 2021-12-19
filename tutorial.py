import matplotlib.pyplot as plt
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        internal_layer_size = 1024
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, internal_layer_size),
            nn.ReLU(),
            nn.Linear(internal_layer_size, internal_layer_size),
            nn.ReLU(),
            nn.Linear(internal_layer_size, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.cuda()
    gap_time = time.time()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.cuda()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float32).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return correct, test_loss


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor(),
)

# breakpoint()


with open("out.txt", "w") as f:

    start_time = time.time()

    batch_size = 128

    # Output batch size
    print(f"Batch Size: {batch_size} ")
    f.write(f"Batch Size: {batch_size}")

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    # Create data loaders.
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # breakpoint()

    # Display info about the shape of the data loaders
    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    # Print model
    model = NeuralNetwork().to(device)
    # model = model.cuda()
    print(model)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Run model for 25 epochs
    epochs = 3

    for t in range(epochs):

        # print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        correct, test_loss = test(test_dataloader, model, loss_fn)

        if t == epochs - 1:
            print_str = f"Test Error - [Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}] "
            print(print_str)
            f.write(print_str)

    time_str = f"Time taken: {time.time() - start_time} \n"
    print(time_str)
    f.write(time_str)


print("Done!")
