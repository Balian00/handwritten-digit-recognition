import torchvision.transforms as T
import torchvision.datasets as D

# parameters
H = 32
W = 32
mean = (0.5,) # classical value for grayscale
std = (0.5,)

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