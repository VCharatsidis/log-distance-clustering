from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class BaseConv(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels):
        """
        Initializes ConvNet object.
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        TODO:
        Implement initialization of the network.
        """

        super(BaseConv, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        out = self.conv1(x)
        out = self.batchNorm1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        # print("conv 1: ")
        # print(out.shape)

        out = self.conv2(out)
        out = self.batchNorm2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)


        return out

