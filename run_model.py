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

# TODO: Fix this
# state_dict = torch.load("models/98_percent_model.pt", map_location=torch.device(device))
# model.load_state_dict(state_dict)


@dataclass
class WindowProperties:
    width: int = 600
    height: int = 600

    # A "pixel block" is a word that I made up to refer to a group of pixels on screen that
    # represent one greyscale pixel in an image in the MNIST dataset. So there would be a
    # 20x20 region displaying on screen that really represents a single pixel being sent to
    # the NN.

    # Number of rows/columns in the array of "pixel blocks"
    pixel_block_rows: int = 28
    pixel_block_columns: int = 28

    # While each pixel block represents one pixel in the output, the pixel block will be
    # displayed as more than just one pixel to make it easier to see. This is the scaling
    # factor from the original single pixel to what is displayed to the end user.
    pixel_block_scalar: int = 20

    # RGB color definitions
    black: tuple = (0, 0, 0)
    white: tuple = (255, 255, 255)
    scarlet: tuple = (170, 10, 10)
    grey: tuple = (100, 100, 100)


class PixelBlock:
    def __init__(self, left, top, width, height, color):

        self.top_left = [left, top]
        self.color = color

        self.collision_box = pg.Rect(left, top, width, height)

        # The extra parenthesis are on purpose, Surface() takes a tuple as input
        self.pixel_surface = pg.Surface((width, height))
        self.pixel_surface.fill(color)

    def draw(self, display_surface):
        display_surface.blit(self.pixel_surface, self.top_left)

    def update_color(self, color):
        self.color = color
        self.pixel_surface.fill(self.color)


class DigitImage:
    def __init__(self, pixel_width, pixel_height, scalar):

        # This is called a "list comprehension". It allows you to define lists (or in this
        # case nested lists) using in-line for-loops. Here's a simple example:
        # list_a = [i for i in range(5)]   -->  [0, 1, 2, 3, 4]
        self.pixel_array = [
            [
                PixelBlock(
                    i * scalar,
                    j * scalar,
                    scalar,
                    scalar,
                    (0, 0, 0),  # cool visual --> (i + j * 5, i + j * 5, i + j * 5),
                )
                for i in range(pixel_width)
            ]
            for j in range(pixel_height)
        ]

    def draw(self, display_surface, left, top, width, height):
        # The extra parenthesis are on purpose, Surface() takes a tuple as input
        digit_surface = pg.Surface((width, height))

        for x in self.pixel_array:
            for y in x:
                digit_surface.blit(y.pixel_surface, y.top_left)

        display_surface.blit(digit_surface, [left, top])


pg.init()
done = False
window_properties = WindowProperties()

# Initialize display surface as a window
display_surface = pg.display.set_mode(
    (window_properties.width, window_properties.height)
)

# Initialize the image array class
current_digit = DigitImage(
    window_properties.pixel_block_columns,
    window_properties.pixel_block_rows,
    window_properties.pixel_block_scalar,
)

# print(current_drawing.pixel_array)

while not done:

    # Check through unprocessed events in the pygame event queue
    for event in pg.event.get():
        # If the event is a quit event (closing the window)
        if event.type == pg.QUIT:
            done = True

    # Fill the background of the window in
    display_surface.fill(window_properties.scarlet)

    current_digit.draw(
        display_surface,
        20,
        0,
        window_properties.pixel_block_columns * window_properties.pixel_block_scalar,
        window_properties.pixel_block_rows * window_properties.pixel_block_scalar,
    )

    pg.display.flip()

pg.quit()
