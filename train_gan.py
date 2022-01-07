from numpy import tan, tanh
import torch
from torch import nn
from torch._C import _get_custom_class_python_wrapper
from torch.nn.modules.activation import Tanh
from torch.utils import data
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from src.nn_module_subclasses import NeuralNetwork
from src.nn_module_subclasses import Discriminator
from src.nn_module_subclasses import Generator 
import torch.optim as optim 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
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
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )






epochs = 10
img_dim = 28*28*1
z_dim = 32
disc = Discriminator(img_dim=img_dim)
gen = Generator(z_dim=z_dim, img_dim=img_dim)
criterion = nn.BCELoss()
disc_optim = optim.Adam(disc.parameters(), lr=1e-5)
gen_optim = optim.Adam(gen.parameters(), lr=1e-5)


def train_gan(dataloader, generator, discriminator, loss_fn, generator_optimizer, discriminator_optimizer, epochs, z_dim):
    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.view(-1, 28*28*1).to(device)
            batch_size = real.shape[0]

            #generate random noise
            z = torch.randn(batch_size, z_dim)

            #generate fake image
            fake = generator(z)

            #discriminator real image loss
            disc_real = discriminator(real).view(-1)
            disc_real_loss = loss_fn(disc_real, torch.ones_like(disc_real))

            #discriminator fake image loss
            disc_fake = discriminator(fake).view(-1)
            disc_fake_loss = loss_fn(disc_fake, torch.zeros_like(disc_fake))

            total_loss = (disc_real_loss + disc_fake_loss)/2

            discriminator.zero_grad()
            total_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            #generator loss
            gen_output = discriminator(fake).view(-1)
            gen_loss = loss_fn(gen_output, torch.ones_like(gen_output))

            generator.zero_grad()
            gen_loss.backward(retain_graph=True)
            generator_optimizer.step()
            print(gen_output)

            if batch_idx == 0:
                print(
                    f'Epoch: {epoch}/{epochs}, \
                    Loss Discriminator: {total_loss:.4f},\
                 Loss Generator: {gen_loss:.4f}'
                )








        







train_gan(train_dataloader, gen,disc, criterion, gen_optim, disc_optim, 2, z_dim)


#or t in range(epochs):
   # print(f"Epoch {t+1}\n-------------------------------")
   # train(train_dataloader, model, loss_fn, optimizer)
   # test(test_dataloader, model, loss_fn)
#print("Done!")
