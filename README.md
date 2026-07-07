# Handwritten Digit Classifier

A handwritten digit classifier (0-9) based on a CNN (convolutional neural network), trained with [PyTorch](https://pytorch.org/) on a [Kaggle dataset](https://www.kaggle.com/datasets/panghalvishesh/handwritten-digit).

## Use of AI

This project was built entirely by me — architecture, training logic, data pipeline, debugging, and all design decisions.

[AI](https://claude.ai/) was used in a limited, assistive capacity for the following:

- **Prints**: written by me, minor aesthetic improvements suggested by AI (alignment, symbols).
- **README**: structured and initially written by me; AI lightly rephrased some sections for fluency and corrected grammar.
- **Docstrings**: written by me, AI translated them to English and reformatted them to Google style.
- **Results table**: data and analysis are mine; AI reformatted the table into the per-parameter breakdown format.
- **Two minor optimizations**: AI suggested adding `num_workers=2` and `pin_memory=True` to the DataLoader, and using a larger batch size during evaluation — both one-line changes.

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

All experiments were run on a single T4 GPU (Kaggle). The notebook used is available at: https://www.kaggle.com/code/balianfranquet/training-cnn-model-to-digit-recognition

### Reference run

All parameters are set to their standard values. Each section below varies one parameter at a time while keeping the others fixed at these reference values.

| Epochs | H×W   | Learning rate | Augmentation | Split | Accuracy | Time   |
|--------|-------|---------------|--------------|-------|----------|--------|
| 30     | 28×28 | 0.01          | Standard     | 0.14  | 93.00%   | 9.6min |

These values follow the standard MNIST setup: 30 epochs, 28×28 images, lr=0.01, split=0.14 (equivalent to ~60K train / 10K test).

---

### Effect of epochs

The number of epochs controls how many times the model sees the entire dataset during training. More epochs generally improve accuracy, at the cost of training time.

| Epochs | Accuracy | Time    |
|--------|----------|---------|
| 15     | 90.01%   | 4.8min  |
| **30** | **93.00%** | **9.6min** |
| 50     | 95.23%   | 15.9min |

More epochs consistently improve accuracy. The gain from 30 to 50 epochs (~2%) is meaningful and the time cost remains reasonable.

---

### Effect of learning rate

The learning rate controls how large each weight update step is during training. Too high and the model overshoots; too low and it converges too slowly.

| Learning rate | Accuracy | Time   |
|---------------|----------|--------|
| **0.01**      | **93.00%** | **9.6min** |
| 0.001         | 70.02%   | 9.7min |

A learning rate of 0.001 significantly underperforms with 30 epochs — the model converges too slowly to reach a good solution in the allotted time.

---

### Effect of augmentation

Data augmentation artificially increases the diversity of training images by applying random transformations at load time. This helps the model generalize to real-world images rather than overfitting to the clean dataset.

Standard augmentation: crop=4px, rotation=±10°, brightness/contrast=0.3.  
Light: half these values. Aggressive: double these values.

| Augmentation  | Accuracy | Time   |
|---------------|----------|--------|
| False         | 92.44%   | 6.8min |
| Light         | 94.73%   | 9.6min |
| **Standard**  | **93.00%** | **9.6min** |
| Aggressive    | 78.96%   | 9.5min |

While training without augmentation yields a competitive test set accuracy (92.44%), it fails to generalize to real handwritten images — tests on images from `test_data/` show noticeably worse recognition compared to augmented models. Light augmentation achieves the best balance: it improves both test set accuracy and real-world generalization without over-distorting the training images as aggressive augmentation does.

---

### Effect of input size

The input size determines the resolution at which images are fed into the network. Larger images preserve more detail but require more computation.

| H×W     | Accuracy | Time   |
|---------|----------|--------|
| **28×28** | **93.00%** | **9.6min** |
| 32×32   | 94.37%   | 9.7min |

Slightly larger input yields a modest accuracy gain (~1.4%) with virtually no additional time cost.

---

### Effect of train/test split

The split ratio determines what fraction of the dataset is reserved for testing. A larger test set gives a more reliable accuracy estimate but leaves fewer samples for training.

| Split  | Accuracy | Time   |
|--------|----------|--------|
| **0.14** | **93.00%** | **9.6min** |
| 0.2    | 92.22%   | 9.4min |

The standard MNIST split (0.14) slightly outperforms the larger test split, likely due to having more training data available.

---

### Optimal configuration

Based on the results above, the most promising configuration combines the best-performing value from each parameter: 50 epochs, 32×32 input, lr=0.01, light augmentation, split=0.14.

| Epochs | H×W   | Learning rate | Augmentation | Split | Accuracy | Time |
|--------|-------|---------------|--------------|-------|----------|------|
| 50     | 32×32 | 0.01          | Light        | 0.14  | 95.97% | 13.9min |

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

## Possible continuations

This project is an extensible base towards broader visual recognition problems:

- **Handwritten letters**: extend the classification to the 26 letters of the alphabet (or even both cases), relying on the same CNN architecture with an adapted number of output classes.
- **Digits in real-world conditions**: train the model on digits photographed in varied contexts — house numbers, license plates, digital displays, handwriting on a whiteboard — for robust recognition outside a controlled context.
- **General-purpose OCR**: combine the two extensions above to build a system capable of reading sequences of characters (words, numbers) rather than a single isolated character.
