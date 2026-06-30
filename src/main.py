import click
import torch
import time
from engine import train, test, test_one
from model import DigitCNN
from const import SPLIT, GENERATOR
from torch.utils.data import random_split
from data import getDataset
import random
from PIL import Image
from data import transform


def get_random_index(dataset):
    return random.randint(0, len(dataset) - 1)

@click.command()
@click.option('--mode', type=click.Choice(['train', 'test', 'test_one']))
@click.option('--image', type=str, default=None)

def main(mode, image):
    t_start = time.time()
    print(f"\n  ── Handwritten Digit Recognizer ──  mode: {mode}") #print by ai
    dataset = getDataset("../dataset")

    if mode == 'train':
        train()

    elif mode == 'test':
        model = DigitCNN()
        model.load_state_dict(torch.load("model.pth"))
        train_dataset, test_dataset = random_split(dataset, [1-SPLIT, SPLIT], generator=GENERATOR)
        accuracy = test(model, test_dataset)
        elapsed = time.time() - t_start
        print("─" * 40) #print by ai
        print(f"  Accuracy : {accuracy:.2f}%") #print by ai
        print(f"  Total time : {elapsed:.1f}s") #print by ai
        print("─" * 40) #print by ai

    elif mode == 'test_one':
        model = DigitCNN()
        model.load_state_dict(torch.load("model.pth"))
        if image is not None :
            # image store dans suppdata, au même format que dataset
            img = Image.open(image)
            tensor = transform(img)
            test_one(model, tensor)
        elif image is None :
            image, label = dataset[get_random_index(dataset)]
            test_one(model, image, label)
        elapsed = time.time() - t_start
        print(f"  Total time : {elapsed:.1f}s") #print by ai
    else:
        return 0

if __name__ == '__main__':
    main()