import click
import torch
import time
from src.engine import train, test, test_one
from src.model import DigitCNN
from src.const import SPLIT, AUGMENTATION, GENERATOR
from torch.utils.data import random_split, Dataset
from src.data import get_dataset, get_transform
import random
from PIL import Image, UnidentifiedImageError
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "src" / "model.pth"


DATASET_ROOT = "dataset"


def get_random_index(dataset: Dataset) -> int:
    """Returns a valid random index for the given dataset.

    Args:
        dataset: The dataset to pick an index from.

    Returns:
        A random integer between 0 and len(dataset) - 1 inclusive.
    """
    return random.randint(0, len(dataset) - 1)


def load_model() -> DigitCNN:
    """Loads the model from model.pth and returns the instance ready for inference.

    Args:
        None — the path is defined by MODEL_PATH in this module.

    Returns:
        The model with loaded weights, evaluation mode not enabled
        (the caller handles the mode via test() or test_one()).
    """
    model = DigitCNN()
    print("  Loading model from 'model.pth'...", end="", flush=True)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    except FileNotFoundError:
        print(" failed")
        print("  Error: model.pth not found. Run --mode train first.")
        raise
    except RuntimeError:
        print(" failed")
        print(
            "  Error: model.pth is corrupted or incompatible with the current architecture."
        )
        raise
    print(" ok")
    return model


@click.command()
@click.option("--mode", type=click.Choice(["train", "test", "test_one"]), required=True)
@click.option("--image", type=str, default=None)
def main(mode: str, image: str | None) -> None:
    """Main entry point of the program, routed by Click.

    Args:
        mode: Execution mode among 'train', 'test', 'test_one'.
        image: Path to an external image for test_one. If None,
               a random image is picked from the dataset.
    """
    t_start = time.time()
    print("")
    print("─" * 40)
    print(f"  Handwritten Digit Recognizer  —  mode: {mode}")
    print("─" * 40)

    if mode in ("train", "test"):
        print(f"  Loading dataset from '{DATASET_ROOT}'...", end="", flush=True)
        try:
            dataset_train = get_dataset(DATASET_ROOT, augmentation=AUGMENTATION)
            dataset_test = get_dataset(DATASET_ROOT, augmentation=False)
        except FileNotFoundError:
            print(" failed")
            print(f"  Error: dataset folder '{DATASET_ROOT}' not found.")
            raise
        print(" ok")

        indices = list(range(len(dataset_train)))
        train_indices, test_indices = random_split(
            indices, [1 - SPLIT, SPLIT], generator=GENERATOR
        )

        train_dataset = torch.utils.data.Subset(dataset_train, train_indices.indices)
        test_dataset = torch.utils.data.Subset(dataset_test, test_indices.indices)
        print(
            f"  Dataset ready  ({len(train_dataset)} train / {len(test_dataset)} test)"
        )
        print("─" * 40)

        if mode == "train":
            train(train_dataset, test_dataset)

        elif mode == "test":
            model = load_model()
            accuracy = test(model, test_dataset)
            elapsed = time.time() - t_start
            print("─" * 40)
            print(f"  Accuracy   : {accuracy:.2f}%")
            print(f"  Total time : {elapsed:.1f}s")
            print("─" * 40)

    elif mode == "test_one":
        model = load_model()
        if image is not None:
            # image stored in suppdata, same format as dataset
            print(f"  Loading image '{image}'...", end="", flush=True)
            try:
                img = Image.open(image)
            except FileNotFoundError:
                print(" failed")
                print(f"  Error: image '{image}' not found. Check the provided path.")
                raise
            except UnidentifiedImageError:
                print(" failed")
                print(f"  Error: file '{image}' is not recognized as a valid image.")
                raise
            print(" ok")
            tensor = get_transform(False)(img)
            test_one(model, tensor)
        else :
            print(f"  Loading dataset from '{DATASET_ROOT}'...", end="", flush=True)
            try:
                dataset = get_dataset(DATASET_ROOT, False)
            except FileNotFoundError:
                print(" failed")
                print(f"  Error: dataset folder '{DATASET_ROOT}' not found.")
                raise
            print(" ok")
            image, label = dataset[get_random_index(dataset)]
            test_one(model, image, label)
        elapsed = time.time() - t_start
        print(f"  Total time : {elapsed:.1f}s")
        print("─" * 40)


if __name__ == "__main__":
    main()
