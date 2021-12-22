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
model.load_state_dict(
    torch.load("models/98_percent_model.pt", map_location=torch.device(device))
)

breakpoint()
