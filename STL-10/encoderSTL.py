from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch


class EncoderSTL(nn.Module):
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
        super(EncoderSTL, self).__init__()
        stride = 1
        max_s = 2
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=stride, padding=0),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),
            #
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),
            #
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),
            #
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),

        )

        self.linear = nn.Sequential(
            #nn.Dropout(0.7),
            nn.Linear(n_inputs, 800),
            nn.Tanh(),

            nn.Dropout(0.5),
            nn.Linear(800, 10),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, p1, p2, p3):
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

        linear_input = torch.cat([conv, p1, p2, p3], 1)

        logits = self.linear(linear_input)
        out = self.softmax(logits)

        return out