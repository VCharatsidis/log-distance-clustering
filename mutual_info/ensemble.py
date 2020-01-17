from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch


class Ensemble(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self):
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
        super(Ensemble, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        )

        input = 8192
        self.shared_linear = nn.Sequential(
            nn.Linear(input, 100),
            nn.Tanh(),
        )

        self.a = nn.Sequential(

            nn.Linear(input, 100),
            nn.Tanh(),

            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

        self.b = nn.Sequential(

            nn.Linear(input, 100),
            nn.Tanh(),

            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

        self.c = nn.Sequential(

            nn.Linear(input, 100),
            nn.Tanh(),

            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )


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
        flat = torch.flatten(conv, 1)
        shared = self.shared_linear(flat)

        a = self.a(x)
        b = self.b(x)
        c = self.c(x)

        return a, b, c