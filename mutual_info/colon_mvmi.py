from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch


class Colon(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_channels, n_inputs):
        """
        Initializes MLP object.
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
        """
        super(Colon, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(n_inputs, 500),
            nn.Tanh(),

            nn.Dropout(),
            nn.Linear(500, 10)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        conv = self.conv(x)
        conv = torch.flatten(conv, 1)

        logits = self.linear(conv)
        out = self.softmax(logits)

        return out