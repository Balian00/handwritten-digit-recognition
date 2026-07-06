import click
import torch
import time
from engine import train, test, test_one
from model import DigitCNN
from const import SPLIT, GENERATOR, AUGMENTATION
from torch.utils.data import random_split, Dataset
from data import get_dataset, get_transform
import random
from PIL import Image, UnidentifiedImageError


DATASET_ROOT = "../dataset"


def get_random_index(dataset : Dataset) -> int :
    return random.randint(0, len(dataset) - 1)


def load_model() -> DigitCNN :
    model = DigitCNN()
    print("  Loading model from 'model.pth'...", end="", flush=True) #print by ai
    try :
        model.load_state_dict(torch.load("model.pth"))
    except FileNotFoundError:
        print(" échec") #print by ai
        print("  Erreur : le fichier model.pth est introuvable. Lance d'abord --mode train.") #print by ai
        raise
    except RuntimeError:
        print(" échec") #print by ai
        print("  Erreur : model.pth est corrompu ou incompatible avec l'architecture actuelle.") #print by ai
        raise
    print(" ok") #print by ai
    return model


@click.command()
@click.option('--mode', type=click.Choice(['train', 'test', 'test_one']))
@click.option('--image', type=str, default=None)

def main(mode: str, image: str | None) -> None:
    t_start = time.time()
    print("") #print by ai
    print("─" * 40) #print by ai
    print(f"  Handwritten Digit Recognizer  —  mode: {mode}") #print by ai
    print("─" * 40) #print by ai

    if mode in ('train', 'test'):
        print(f"  Loading dataset from '{DATASET_ROOT}'...", end="", flush=True) #print by ai
        try :
            dataset_train = get_dataset(DATASET_ROOT, augmentation=AUGMENTATION)
            dataset_test = get_dataset(DATASET_ROOT, augmentation=False)
        except FileNotFoundError:
            print(" échec") #print by ai
            print(f"  Erreur : le dossier dataset '{DATASET_ROOT}' est introuvable.") #print by ai
            raise
        print(" ok") #print by ai

        indices = list(range(len(dataset_train)))
        train_indices, test_indices = random_split(indices, [1-SPLIT, SPLIT])

        train_dataset = torch.utils.data.Subset(dataset_train, train_indices.indices)
        test_dataset = torch.utils.data.Subset(dataset_test, test_indices.indices)
        print(f"  Dataset ready  ({len(train_dataset)} train / {len(test_dataset)} test)") #print by ai
        print("─" * 40) #print by ai

        if mode == 'train':
            train(train_dataset, test_dataset)

        elif mode == 'test':
            model = load_model()
            accuracy = test(model, test_dataset)
            elapsed = time.time() - t_start
            print("─" * 40) #print by ai
            print(f"  Accuracy   : {accuracy:.2f}%") #print by ai
            print(f"  Total time : {elapsed:.1f}s") #print by ai
            print("─" * 40) #print by ai

    elif mode == 'test_one':
        model = load_model()
        if image is not None :
            # image store dans suppdata, au même format que dataset
            print(f"  Loading image '{image}'...", end="", flush=True) #print by ai
            try :
                img = Image.open(image)
            except FileNotFoundError:
                print(" échec") #print by ai
                print(f"  Erreur : l'image '{image}' est introuvable. Vérifie le chemin fourni.") #print by ai
                raise
            except UnidentifiedImageError:
                print(" échec") #print by ai
                print(f"  Erreur : le fichier '{image}' n'est pas reconnu comme une image valide.") #print by ai
                raise
            print(" ok") #print by ai
            tensor = get_transform(False)(img)
            test_one(model, tensor)
        elif image is None :
            print(f"  Loading dataset from '{DATASET_ROOT}'...", end="", flush=True) #print by ai
            try :
                dataset = get_dataset(DATASET_ROOT, False)
            except FileNotFoundError:
                print(" échec") #print by ai
                print(f"  Erreur : le dossier dataset '{DATASET_ROOT}' est introuvable.") #print by ai
                raise
            print(" ok") #print by ai
            image, label = dataset[get_random_index(dataset)]
            test_one(model, image, label)
        elapsed = time.time() - t_start
        print(f"  Total time : {elapsed:.1f}s") #print by ai
        print("─" * 40) #print by ai

if __name__ == '__main__':
    main()
