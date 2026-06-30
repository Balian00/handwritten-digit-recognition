from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as D
from const import H, W, std, mean

# object of transformation
transform = T.Compose([
    T.Resize((H, W)),
    T.Grayscale(), # grayscale sufficient for digit
    T.ToTensor(),
    T.Normalize(mean=mean, std=std) 
])

def getDataset(root) :
    dataset = D.ImageFolder(root, transform=transform) # create object dataset (from torch)
    return dataset

def getDataLoader(dataset, batch_size, shuffle) :
    dataLoader = DataLoader(dataset, batch_size, shuffle)
    return dataLoader