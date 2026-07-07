# Handwritten Digit Classifier

A handwritten digit classifier (0-9) based on a CNN (convolutional neural network), trained with [PyTorch](https://pytorch.org/) on a [Kaggle dataset](https://www.kaggle.com/datasets/panghalvishesh/handwritten-digit).

## Use of AI

This README, as well as the messages printed to the console (print statements), were generated with the help of an AI. All content was reviewed and validated by the author. Everything else — model architecture, training and evaluation logic, data handling, debugging — was written by me.

## Current status (V3)

The CNN correctly recognizes almost all handwritten digits tested manually. The model is trained on a Kaggle handwritten digit dataset with data augmentation (random crop, slight random rotation, random contrast and brightness variation), which significantly improves generalization compared to V1.

Compared to V2, this version focuses on code quality rather than new features:
- **Security / robustness**: explicit error handling (missing dataset, invalid image, missing or corrupted `model.pth`) with clear messages instead of raw crashes.
- **Clean, PEP8 code**: consistent naming, type hints on functions, docstrings, code written in English.
- **Light optimizations**: `torch.no_grad()` during evaluation, `DataLoader` with `num_workers`/`pin_memory`, larger evaluation batch size than the training one.
- **Shorter code**: removal of the custom batch generator (made unnecessary by `DataLoader`), factored paths and CLI logic in `main.py`.

*Note: "almost all handwritten digits tested manually" refers to digits written in a way reasonably close to the dataset. Indeed, the dataset sometimes contains digit representations that differ from how the reader writes them. It is therefore suggested to compare your handwriting to the dataset's when testing manually via `test_one`.*

## How the code works

- `data.py`: loads the dataset with `ImageFolder` and builds the transform pipeline (resizing, grayscale conversion, normalization) with optional augmentation (random crop, slight rotation, contrast/brightness variation). Also provides `get_dataloader`, which creates a `DataLoader` with `num_workers` and `pin_memory` to speed up batch loading.
- `model.py`: defines the CNN architecture (two convolution blocks + ReLU activation + pooling, followed by a fully connected layer producing the 10 classification scores).
- `engine.py`: contains the training loop (`train`), batch evaluation on the test set (`test`, under `torch.no_grad()`), and single-image testing (`test_one`). Weights are only saved when accuracy improves.
- `main.py`: command-line entry point (via `click`) that selects the execution mode (`train`, `test`, `test_one`), loads/splits the dataset, and handles loading/saving the model weights (`model.pth`), with explicit error handling (missing files, invalid images).
- `const.py`: centralizes all hyperparameters (epochs, learning rate, split, image dimensions, augmentation parameters, random generator seed).

## Results

| Epochs | H×W   | Learning rate | Augmentation | Split | Accuracy (test set) | Time   |
|--------|-------|---------------|--------------|-------|---------------------|--------|
| 30     | 28×28 | 0.01          | Standard     | 0.14  | 93.00%              | 9.6min |
| 15     | 28×28 | 0.01          | Standard     | 0.14  | 90.01%              | 4.8min |
| 50     | 28×28 | 0.01          | Standard     | 0.14  | 95.23%              | 15.9min|
| 30     | 28×28 | 0.001         | Standard     | 0.14  | 70.02%              | 9.7min |
| 50     | 28×28 | 0.001         | Standard     | 0.14  | 77.18%              | 15.9min|
| 30     | 28×28 | 0.01          | False        | 0.14  | 92.44%              | 6.8min |
| 30     | 28×28 | 0.01          | Light        | 0.14  | 94.73%              | 9.6min |
| 30     | 28×28 | 0.01          | Aggressive   | 0.14  | 78.96%              | 9.5min |
| 30     | 32×32 | 0.01          | Standard     | 0.14  | 94.37%              | 9.7min |
| 50     | 32×32 | 0.01          | Standard     | 0.14  | 95.36%              | 16.1min|
| 30     | 28×28 | 0.01          | Standard     | 0.2   | 92.22%              | 9.4min |
### Epochs, dimensions, split, learning rate
In the standard MNIST setup, the following values are typically found:
- Epochs = 10 to 30, here 30 is chosen for a main test
- H = W = 28
- LR = 0.01
- Split = 0.14 (60K train + 10K test)

### Augmentation
The standard augmentation chosen corresponds to:
- a crop of 4 pixels
- a rotation of up to 10°
- a brightness and contrast variation of 0.3

A light augmentation corresponds to half these values, and an aggressive one to double them.

### Time
To allow for the least biased comparison possible, tests are standardized on one T4 GPU made accessible by the Kaggle platform. The notebook used for the tests can be found at the following link: https://www.kaggle.com/code/balianfranquet/notebook78580c1166.

## Installation

Create and activate a virtual environment, then install the dependencies listed in `requirements.txt`.

On Windows (PowerShell):
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

All the commands below should be run from the `src` folder, once the virtual environment is activated.

Train the model:
```bash
python main.py --mode train
```

Evaluate accuracy on the test set:
```bash
python main.py --mode test
```

Test on a random image from the dataset:
```bash
python main.py --mode test_one
```

Test on an external image:
```bash
python main.py --mode test_one --image adr_img.png
```
*Note: a set of test images is available in the `test_data/` folder.*

## Dataset structure
The dataset structure is as follows:
```
dataset/
0/
1/
2/
...
9/
```
This allows the use of `ImageFolder` from `torchvision.datasets`.

## Next steps

- Complete the results table with different combinations of hyperparameters

## Possible continuations

This project is an extensible base towards broader visual recognition problems:

- **Handwritten letters**: extend the classification to the 26 letters of the alphabet (or even both cases), relying on the same CNN architecture with an adapted number of output classes.
- **Digits in real-world conditions**: train the model on digits photographed in varied contexts — house numbers, license plates, digital displays, handwriting on a whiteboard — for robust recognition outside a controlled context.
- **General-purpose OCR**: combine the two extensions above to build a system capable of reading sequences of characters (words, numbers) rather than a single isolated character.
