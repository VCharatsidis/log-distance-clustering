import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    def __init__(self, in_channels=3):
        super(ConvLayer, self).__init__()
        max_s = 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),
            #
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class PrimaryCaps(nn.Module):

    def __init__(self, num_capsules=8, in_channels=128, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6):
        super(PrimaryCaps, self).__init__()

        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)

        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))

        return output_tensor


class DigitCaps(nn.Module):

    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        print(x.shape)
        print(torch.__version__)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)

        print("W", W.shape)
        print("x", x.shape)
        # u_hat = W * x
        # print(u_hat.shape)
        u_hat = torch.mul(W, x)
        input()

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))

        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))

        return output_tensor


class Classifier(nn.Module):

    def __init__(self, input_width=20):
        super(Classifier, self).__init__()

        self.classification_layers = nn.Sequential(
            nn.Linear(input_width, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, self.input_height * self.input_height * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        classifications = self.classification_layers(x)

        return classifications


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()

        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()

        self.classifier1 = Classifier()
        self.classifier2 = Classifier()
        self.classifier3 = Classifier()
        self.classifier4 = Classifier()
        self.classifier5 = Classifier()
        self.classifier6 = Classifier()
        self.classifier7 = Classifier()
        self.classifier8 = Classifier()
        self.classifier9 = Classifier()
        self.classifier0 = Classifier()

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        print(output.shape)

        classification1 = self.classifier1(output)
        classification2 = self.classifier2(output)
        classification3 = self.classifier3(output)
        classification4 = self.classifier4(output)
        classification5 = self.classifier5(output)
        classification6 = self.classifier6(output)
        classification7 = self.classifier7(output)
        classification8 = self.classifier8(output)
        classification9 = self.classifier9(output)
        classification0 = self.classifier0(output)

        return output, classification1, classification2, classification3, classification4, classification5, classification6, classification7, classification8, classification9, classification0

