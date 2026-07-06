import torch
import os
import time
from data import get_dataloader
from model import DigitCNN
import torch.nn as nn
from const import EPOCHS, SPLIT, GENERATOR, LR
from torch.utils.data import Dataset

# test
def test(model : nn.Module, test_dataset : Dataset) -> float :
    
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    for images, labels in get_dataloader(test_dataset, 32, False) :
        images, labels = images.to(device), labels.to(device)
        predictions = model.forward(images)
        predicted_classes = predictions.argmax(dim=1)
        correct = correct + (predicted_classes == labels).sum()
    accuracy = correct / len(test_dataset) * 100
    model.train()
    return accuracy.item()


# train
def train(train_dataset : Dataset, test_dataset : Dataset) -> tuple[float, float] :
    t_start = time.time()

    print("  Initializing model...", end="", flush=True) #print by ai
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    print(f" ok, running on {device}.") #print by ai
    print("─" * 40) #print by ai

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    current_accuracy = 0
    best_accuracy = 0

    for epoch in range(EPOCHS) :
        print(f"\n  Epoch [{epoch + 1:2d}/{EPOCHS}]  training...", end="", flush=True) #print by ai
        loss = 0
        for images, labels in get_dataloader(train_dataset, 32, True) :
            images, labels = images.to(device), labels.to(device)
            predictions = model.forward(images)
            losses = criterion(predictions, labels)
            loss = loss + losses
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
        current_accuracy = test(model, test_dataset)
        saved = ""
        if current_accuracy > best_accuracy :
            try :
                torch.save(model.state_dict(), "model.pth")
            except OSError:
                print("  Erreur : impossible de sauvegarder model.pth. Vérifie les permissions du dossier.") #print by ai
                raise
            best_accuracy = current_accuracy
            saved = "  ✓ saved"
        print(f"  accuracy: {current_accuracy:.2f}%{saved}") #print by ai

    elapsed = time.time() - t_start
    print("\n" + "─" * 40) #print by ai
    print(f"  Training complete.") #print by ai
    print(f"  Best accuracy : {best_accuracy:.2f}%") #print by ai
    print(f"  Total time    : {elapsed:.1f}s  ({elapsed/60:.1f} min)") #print by ai
    print("─" * 40) #print by ai
    return best_accuracy, elapsed


def test_one(model : nn.Module, tensor : torch.Tensor, label : int =None) -> int :
    model.eval()
    device = next(model.parameters()).device
    tensor = tensor.to(device)
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
    return predicted_class