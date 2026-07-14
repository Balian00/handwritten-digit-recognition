import torch
import time
from src.data import get_dataloader
from src.model import DigitCNN
import torch.nn as nn
from src.const import EPOCHS, LR
from torch.utils.data import Dataset
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.pth"


def test(model: nn.Module, test_dataset: Dataset) -> float:
    """Evaluates the model on a dataset and returns its accuracy.

    Automatically switches the model to evaluation mode before inference,
    then switches it back to training mode before returning.

    Args:
        model: The model to evaluate.
        test_dataset: The test dataset, compatible with DataLoader.

    Returns:
        The accuracy as a percentage (0-100).
    """
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    with torch.no_grad():  # disables gradient graph, saves memory and speeds up inference
        data_loader = get_dataloader(test_dataset, 256, False) # larger batch at eval: no backprop, more memory available
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)  # __call__ triggers forward() + nn.Module hooks
            predicted_classes = predictions.argmax(dim=1)
            correct += (predicted_classes == labels).sum()
    accuracy = correct / len(test_dataset) * 100
    model.train()
    return accuracy.item()


def train(train_dataset: Dataset, test_dataset: Dataset) -> tuple[float, float]:
    """Trains the model and saves the best weights.

    Creates and initializes a new model on every call. Automatically
    saves the weights to model.pth only when the accuracy on the test
    set improves.

    Args:
        train_dataset: The training dataset, with augmentation.
        test_dataset: The test dataset, without augmentation.

    Returns:
        A tuple (best_accuracy, elapsed) where best_accuracy is the best
        accuracy reached as a percentage and elapsed is the total time in seconds.
    """
    t_start = time.time()

    print("  Initializing model...", end="", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    print(f" ok, running on {device}.")
    print("─" * 40)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    current_accuracy = 0
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f"\n  Epoch [{epoch + 1:2d}/{EPOCHS}]  training...", end="", flush=True)
        for images, labels in get_dataloader(train_dataset, 32, True):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)  # __call__ triggers forward() + nn.Module hooks
            losses = criterion(predictions, labels)
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
        current_accuracy = test(model, test_dataset)
        saved = ""
        if current_accuracy > best_accuracy:
            try:
                torch.save(model.state_dict(), MODEL_PATH)
            except OSError:
                print(
                    "  Error: unable to save model.pth. "
                    "Check the folder permissions."
                )
                raise
            best_accuracy = current_accuracy
            saved = "  ✓ saved"
        print(f"  accuracy: {current_accuracy:.2f}%{saved}")

    elapsed = time.time() - t_start
    print("\n" + "─" * 40)
    print("  Training complete.")
    print(f"  Best accuracy : {best_accuracy:.2f}%")
    print(f"  Total time    : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print("─" * 40)
    return best_accuracy, elapsed


def test_one(model: nn.Module, tensor: torch.Tensor, label: int = None) -> int:
    """Tests the model on a single image and prints the prediction.

    Args:
        model: The model to use for the prediction.
        tensor: The image as a tensor (C, H, W), without batch dimension.
        label: The true label if known, to display whether the prediction is correct.

    Returns:
        The predicted class (integer between 0 and 9).
    """
    model.eval()
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    tensor = tensor.unsqueeze(0)
    guess = model(tensor)
    predicted_class = guess.argmax(dim=1).item()
    print("─" * 40)
    if label is not None:
        passed = predicted_class == label
        status = "correct ✓" if passed else "wrong ✗"
        print(f"  Prediction : {predicted_class}  |  True label : {label}  →  {status}")
    else:
        print(f"  The network thinks the digit is a  {predicted_class}")
    print("─" * 40)
    return predicted_class
