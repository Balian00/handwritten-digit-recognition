# Handwritten Digit Classifier

> README generated entirely by AI.

## Done

* Dataset loading with `ImageFolder`
* Image preprocessing:

  * Resize to 32×32
  * Grayscale conversion
  * Tensor conversion
  * Normalization
* Dataset loading tests
* Sample export to PNG for visual inspection
* DataLoader (`getDataLoader`): shuffling, batching (tested with shape checks)

## Dataset Structure

dataset/
├── 0/
├── 1/
├── 2/
└── ...

## Next

* CNN model
* Training loop (forward pass, loss, backward pass, optimizer step)
* Evaluation