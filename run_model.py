from dataclasses import dataclass
import matplotlib.pyplot as plt
import pygame as pg
from src.nn_class import NeuralNetwork
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from src.nn_class import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
# Load trained model from models folder

state_dict = torch.load("models/98_percent_model.pt", map_location=torch.device(device))
model.load_state_dict(state_dict)


@dataclass
class WindowProperties:
    width: int = 500
    height: int = 500


pg.init()
running = True
window_properties = WindowProperties()

while running:
    # Initialize display surface as a window
    display_surface = pg.display.set_mode(
        (window_properties.width, window_properties.height)
    )
