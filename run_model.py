import matplotlib.pyplot as plt
import pygame as pg
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
