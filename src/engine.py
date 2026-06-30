import torch
import os
from data import getDataset
from data import getDataLoader
from torchvision.utils import save_image
from model import DigitCNN
import torch.nn as nn
from const import EPOCHS, SPLIT, GENERATOR
from torch.utils.data import random_split


#test function by AI
def save_sample_as_png(dataset, index, output_path="sample.png"):
    img, label = dataset[index]

    print("Label:", label)
    print("Shape:", img.shape)

    # si grayscale (1 channel), on duplique pour sauvegarde RGB
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    # IMPORTANT:
    # dénormalization (valeur pas modualire -> fichier de constante à définir)
    img = img * 0.5 + 0.5

    # clamp pour éviter valeurs hors [0,1] 
    img = torch.clamp(img, 0, 1)

    # sauvegarde PNG
    save_image(img, output_path)

    print(f"Image saved to {output_path}")

# test
def test(model, test_dataset) :
    model.eval()
    print('Testing...')
    correct = 0
    for images, labels in getDataLoader(test_dataset, 32, False) :
        predictions = model.forward(images)
        predicted_classes = predictions.argmax(dim=1)
        correct = correct + (predicted_classes == labels).sum()
    accuracy = correct / len(test_dataset) * 100
    model.train()
    return accuracy


# train
def train():
    print("Loading dataset..")

    # BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    # print(f'{BASE_DIR}')
    # root = os.path.join(BASE_DIR, "dataset")
    # dataset = getDataset(root)

    dataset = getDataset("../dataset")
    train_dataset, test_dataset = random_split(dataset, [1-SPLIT, SPLIT], generator=GENERATOR)

    print("Dataset loaded.")
    print("Model initializing..")
    model = DigitCNN()
    print("Model initialized.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    print("Loop on batch")
    for epoch in range(EPOCHS) :
        print(f'Start epoch {epoch}/{EPOCHS}.')
        loss = 0
        for images, labels in getDataLoader(train_dataset, 32, True) :
            predictions = model.forward(images)
            losses = criterion(predictions, labels)
            loss = loss + losses
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
        accuracy = test(model, test_dataset)        
        print(f'Epoch {epoch}/{EPOCHS} ended with an accuracy of {accuracy}')
    torch.save(model.state_dict(), "model.pth")
        


# test_one 
def test_one(model, image):
    model.eval()
    print('Testing 1...')
    tensor, label = image
    tensor = tensor.unsqueeze(0)
    guess = model.forward(tensor)
    predicted_class = guess.argmax(dim=1).item()
    passed = predicted_class == label
    print(f'Guess: {predicted_class}, true digit was {label}, so the guess is {passed}')