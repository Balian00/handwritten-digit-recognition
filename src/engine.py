import torch
import os
import time
from data import getDataset, getDataLoader
from torchvision.utils import save_image
from model import DigitCNN
import torch.nn as nn
from const import EPOCHS, SPLIT, GENERATOR, LR
from torch.utils.data import random_split

# test
def test(model, test_dataset) :
    model.eval()
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
    t_start = time.time()

    print("─" * 40) #print by ai
    print("  Loading dataset...")  #print by ai

    dataset = getDataset("../dataset_old")
    # dataset, _ = random_split(dataset, [0.1, 0.9], generator=GENERATOR)
    train_dataset, test_dataset = random_split(dataset, [1-SPLIT, SPLIT], generator=GENERATOR)

    print(f"  Dataset ready  ({len(train_dataset)} train / {len(test_dataset)} test)") #print by ai
    print("─" * 40) #print by ai
    print("  Initializing model...") #print by ai
    model = DigitCNN()
    print("  Model ready.") #print by ai
    print("─" * 40) #print by ai

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    current_accuracy = 0
    best_accuracy = 0

    for epoch in range(EPOCHS) :
        print(f"\n  Epoch [{epoch + 1:2d}/{EPOCHS}]  training...", end="", flush=True) #print by ai
        loss = 0
        for images, labels in getDataLoader(train_dataset, 32, True) :
            predictions = model.forward(images)
            losses = criterion(predictions, labels)
            loss = loss + losses
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
        current_accuracy = test(model, test_dataset)
        saved = ""
        if current_accuracy > best_accuracy :
            torch.save(model.state_dict(), "model.pth")
            best_accuracy = current_accuracy
            saved = "  ✓ saved"
        print(f"  accuracy: {current_accuracy:.2f}%{saved}") #print by ai

    elapsed = time.time() - t_start
    print("\n" + "─" * 40) #print by ai
    print(f"  Training complete.") #print by ai
    print(f"  Best accuracy : {best_accuracy:.2f}%") #print by ai
    print(f"  Total time    : {elapsed:.1f}s  ({elapsed/60:.1f} min)") #print by ai
    print("─" * 40) #print by ai


def test_one(model, tensor, label=None):
    model.eval()
    tensor = tensor.unsqueeze(0)
    guess = model.forward(tensor)
    predicted_class = guess.argmax(dim=1).item()
    print("─" * 40) #print by ai
    if label is not None:
        passed = predicted_class == label
        status = "correct ✓" if passed else "wrong ✗"
        print(f"  Prediction : {predicted_class}  |  True label : {label}  →  {status}") #print by ai
    else:
        print(f"  The network thinks the digit is a  {predicted_class}") #print by ai
    print("─" * 40) #print by ai