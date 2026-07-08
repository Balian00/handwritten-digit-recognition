from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.datasets as D
from const import H, W, AUGMENTATION, CROP, ROTATE, BRIGHTNESS, CONTRAST, std, mean
import torch

def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 2, pin_memory: bool = torch.cuda.is_available()) -> DataLoader:
    return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)


def get_transform(augmentation: bool = AUGMENTATION) -> T.Compose:
    """Builds the transformation pipeline applied to the images.

    Always applies: resizing, grayscale conversion, normalization.
    If augmentation is enabled, adds a random crop, a slight rotation
    and a contrast/brightness variation.

    Args:
        augmentation: If True, adds the data augmentation transforms.

    Returns:
        A ready-to-use torchvision transforms pipeline.
    """
    transforms = [
        T.Resize((H, W)),
        T.Grayscale(),
    ]
    if augmentation:
        transforms += [
            T.RandomCrop(H, padding=CROP),
            T.RandomRotation(ROTATE),
            T.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST),
        ]
    transforms += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(transforms)


def get_dataset(root: str, augmentation: bool) -> Dataset:
    """Loads the dataset from a folder structured by class.

    Expects an ImageFolder-style structure: one subfolder per class,
    named after the label (e.g. dataset/0/, dataset/1/, ...).

    Args:
        root: Path to the dataset root folder.
        augmentation: If True, applies data augmentation on loading.

    Returns:
        The loaded dataset, ready to be split or passed to a DataLoader.
    """
    return D.ImageFolder(
        root, transform=get_transform(augmentation)
    )  # create object dataset (from torch)
    


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 2, pin_memory: bool = torch.cuda.is_available()) -> DataLoader: # num_workers: parallel batch prefetch; pin_memory: faster CPU→GPU transfer
    """Creates a DataLoader from a dataset.

    Args:
        dataset: The dataset to load in batches.
        batch_size: Number of images per batch.
        shuffle: If True, shuffles the data at each epoch.

    Returns:
        An iterable DataLoader producing batches of tensors.
    """
    return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory) # num_workers: parallel batch prefetch; pin_memory: faster CPU→GPU transfer
