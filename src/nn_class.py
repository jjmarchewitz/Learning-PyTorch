import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

# I moved this to its own file because both train_model.py and run_model.py will need
# to use this class. Good OOP design and all that


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.internal_layer_size = 28 * 28

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, self.internal_layer_size),
            nn.ReLU(),
            nn.Linear(self.internal_layer_size, self.internal_layer_size),
            nn.ReLU(),
            nn.Linear(self.internal_layer_size, self.internal_layer_size),
            nn.ReLU(),
            nn.Linear(self.internal_layer_size, 10),
        )

    def forward(self, x):

        # x comes in as 4-dimensional array/batch of 28x28 images. If you have x[A][B][C][D]
        # then A is the image number in the batch, idk what B is but it only has 1 possible
        # value, C is either row or column and D is the other one (column or row respectively)
        x = self.flatten(x)

        # After flattening, x is a list of flattened images. x[A] will give you image number
        # A in the list
        logits = self.linear_relu_stack(x)
        return logits
