import torch
import os
from utils.dataset import getDataset

def train():
    print('Loading dataset..')

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    root = os.path.join(BASE_DIR, "dataset")

    dataset = getDataset(root)

    print('Dataset loaded.')