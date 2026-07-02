from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as D
from const import H, W, AUGMENTATION, CROP, ROTATE, BRIGHTNESS, CONTRAST, std, mean
from PIL import Image as PILImage

def rgba_loader(path):
    img = PILImage.open(path)
    if img.mode == 'RGBA':
        background = PILImage.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    return img.convert('RGB')

# object of transformation
def getTransform(augmentation=AUGMENTATION):
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

def getDataset(root) :
    dataset = D.ImageFolder(root, transform=getTransform(AUGMENTATION), loader=rgba_loader) # create object dataset (from torch)
    return dataset

def getDataLoader(dataset, batch_size, shuffle) :
    dataLoader = DataLoader(dataset, batch_size, shuffle)
    return dataLoader