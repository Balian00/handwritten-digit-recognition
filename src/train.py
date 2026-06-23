import torch
import os
from utils.dataset import getDataset
from torchvision.utils import save_image


def train():
    print("Loading dataset..")

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    root = os.path.join(BASE_DIR, "dataset")

    dataset = getDataset(root)

    print("Dataset loaded.")

    # test: save one sample
    save_sample_as_png(dataset, index=0)


def save_sample_as_png(dataset, index=0, output_path="sample.png"):
    img, label = dataset[index]

    print("Label:", label)
    print("Shape:", img.shape)

    # si grayscale (1 channel), on duplique pour sauvegarde RGB
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    # IMPORTANT:
    # si tu as Normalize(0.5, 0.5), il faut dénormaliser
    img = img * 0.5 + 0.5

    # clamp pour éviter valeurs hors [0,1]
    img = torch.clamp(img, 0, 1)

    # sauvegarde PNG
    save_image(img, output_path)

    print(f"Image saved to {output_path}")