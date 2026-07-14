import torch.nn as nn
import torch
from src.const import H, W


class DigitCNN(nn.Module):
    """CNN for handwritten digit classification (0-9).

    Architecture: two conv+relu+pooling blocks followed by a
    fully-connected layer. The size of the linear layer is computed
    automatically via a forward pass on a dummy tensor.
    """

    def __init__(self) -> None:
        """Initializes the network layers.

        The input size of the linear layer is inferred automatically
        from H and W defined in const.py, without manual computation.
        """
        super().__init__()
        # layers declared here
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

        # to compute shape of in_features
        dummy = torch.zeros(1, 1, H, W)
        dummy = self.conv1(dummy)
        dummy = self.relu1(dummy)
        dummy = self.max_pool1(dummy)
        dummy = self.conv2(dummy)
        dummy = self.relu2(dummy)
        dummy = self.max_pool2(dummy)
        in_features = dummy.shape[1] * dummy.shape[2] * dummy.shape[3]

        self.linear = nn.Linear(in_features, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass on a batch of images.

        Args:
            x: Batch of images of shape (N, 1, H, W).

        Returns:
            Tensor of raw scores (logits) of shape (N, 10),
            one score per class for each image in the batch.
        """
        # forward path of the batch x through the layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
