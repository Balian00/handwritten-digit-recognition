import torch

EPOCHS = 30
SPLIT = 0.2
LR = 0.01

# parameters, valeur standard en MNIST
H = 28
W = 28
mean = (0.5,) # classical value for grayscale
std = (0.5,)

GENERATOR = torch.Generator().manual_seed(42)