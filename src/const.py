import torch

EPOCHS = 15
SPLIT = 0.15

# parameters, valeur standard en MNIST
H = 64
W = 64
mean = (0.5,) # classical value for grayscale
std = (0.5,)

GENERATOR = torch.Generator().manual_seed(42)