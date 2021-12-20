import matplotlib.pyplot as plt
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Model parameters for tweaking
batch_size = 16
epochs = 500
internal_layer_size = 28 * 28
learning_rate = 0.0001
momentum = 0.5

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, internal_layer_size),
            nn.ReLU(),
            nn.Linear(internal_layer_size, internal_layer_size),
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
    if device == "cuda":
        model = model.cuda()

    size = len(dataloader.dataset)

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
    if device == "cuda":
        model = model.cuda()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

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
training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=ToTensor(),
)


with open("out.txt", "w") as f:

    start_time = time.time()
    torch.backends.cudnn.benchmark = True

    # Output model tweaking parameters
    param_str = (
        f"Batch Size: {batch_size}\nEpochs: {epochs}\nLR: {learning_rate}\n"
        f"Momentum: {momentum}\n"
    )
    print(param_str)

    # I am pretty sure pinned memory somehow refers to storing on the GPU when True
    pinned_memory = True if device == "cuda" else False

    # Create data loaders.
    train_dataloader = DataLoader(
        training_data, pin_memory=pinned_memory, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, pin_memory=[pinned_memory], batch_size=batch_size, shuffle=True
    )

    # Display info about the shape of the data loaders
    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    # Print model
    model = NeuralNetwork().to(device)
    if device == "cuda":
        model = model.cuda()
    print(model, "\n")

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Run model
    for t in range(epochs):

        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        correct, test_loss = test(test_dataloader, model, loss_fn)

        if t == epochs - 1:
            final_correct_value = correct
            final_test_loss = test_loss

    # Print out summary of the run at the end
    error_and_loss_str = (
        f"Test Error - [Accuracy: {(100*correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f}]\n"
    ) 

    runtime_str = time.strftime(
        "%H hours, %M minutes, and %S", time.gmtime(time.time() - start_time)
    )
    runtime_str = f"Time taken: {runtime_str} sec \n"

    summary_str = param_str + error_and_loss_str + runtime_str

    print(f"\nSummary\n-------------------------------")
    print(summary_str)
    f.write(summary_str + "\n")

    # Save the trained model to a file
    torch.save(model, "model.pt")

print("Done!")
