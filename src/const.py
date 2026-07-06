import torch

EPOCHS = 30
SPLIT = 0.2
LR = 0.01
AUGMENTATION = True
CROP = 4
ROTATE = 10
BRIGHTNESS = 0.3
CONTRAST = 0.3
H = 28
W = 28

mean = (0.5,) # classical value for grayscale
std = (0.5,)
