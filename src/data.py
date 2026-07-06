from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.datasets as D
from const import H, W, AUGMENTATION, CROP, ROTATE, BRIGHTNESS, CONTRAST, std, mean


# object of transformation
def get_transform(augmentation: bool = AUGMENTATION) -> T.Compose:
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

def get_dataset(root : str, augmentation : bool) -> Dataset :
    dataset = D.ImageFolder(root, transform=get_transform(augmentation)) # create object dataset (from torch)
    return dataset

def get_dataloader(dataset : Dataset, batch_size : int, shuffle : bool) -> DataLoader :
    dataLoader = DataLoader(dataset, batch_size, shuffle)
    return dataLoader