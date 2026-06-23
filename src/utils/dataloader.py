from torch.utils.data import DataLoader

def getDataLoader(dataset, batch_size, shuffle) :
    dataLoader = DataLoader(dataset, batch_size, shuffle)
    return dataLoader

