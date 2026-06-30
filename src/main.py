import click
import torch
from engine import train, test, test_one
from model import DigitCNN
from const import SPLIT, GENERATOR
from torch.utils.data import random_split
from data import getDataset
import random


def get_random_index(dataset):
    return random.randint(0, len(dataset) - 1)

@click.command()
@click.option('--mode', type=click.Choice(['train', 'test', 'test_one']))
@click.option('--image', type=str, default=None)

def main(mode, image=None):
    print(f'Code running in {mode} mode...')
    dataset = getDataset("../dataset")
    if mode == 'train':
        train()
    elif mode == 'test':
        model = DigitCNN()
        model.load_state_dict(torch.load("model.pth"))
        train_dataset, test_dataset = random_split(dataset, [1-SPLIT, SPLIT], generator=GENERATOR)
        test(model, test_dataset)
    elif mode == 'test_one':
        model = DigitCNN()
        model.load_state_dict(torch.load("model.pth"))
        if image is not None :
            test_one(model, image)
        if image is None :
            image = dataset[get_random_index(dataset)]
            test_one(model, image)
    else:
        return 0

if __name__ == '__main__':
    main()