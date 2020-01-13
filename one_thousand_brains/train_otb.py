from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch

from torch.autograd import Variable
import matplotlib.pyplot as plt

from base_conv import BaseConv

from sklearn.datasets import fetch_openml

import statistics
from colon import Colon
from losses import IID_loss, mi

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 1e-5
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 1
HALF_BATCH = BATCH_SIZE_DEFAULT // 2
EVAL_FREQ_DEFAULT = 20

FLAGS = None


def calc_distance(out, y):
    abs_difference = torch.abs(out - y)
    eps = 1e-8
    information_loss = torch.log(1 - abs_difference + eps)

    mean = torch.mean(information_loss)

    return torch.abs(mean)


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def neighbours(convolutions, perimeter=3):
    size = convolutions.shape[0]
    inputs = []
    reception_field = 2 * perimeter + 1

    for i in range(3, size-3):
        for j in range(3, size-3):
            conv_loss_tensor = torch.zeros(reception_field, reception_field)

            for row in range(i-perimeter, i+perimeter+1):
                for col in range(j-perimeter, j+perimeter+1):
                    if row >= 0 and row < size:
                        if col >= 0 and col < size:
                            conv_loss_tensor[row - (i-perimeter), col - (j-perimeter)] = convolutions[row, col]

            flatten_input = torch.flatten(conv_loss_tensor)
            inputs.append(flatten_input)

    return inputs


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def forward_block(X, ids, colons, optimizers, train, to_tensor_size):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, to_tensor_size)
    convolutions = x_tensor/255

    inputs = neighbours(convolutions[0, 0])

    size = len(inputs)
    #print("size: ", size)

    colon_outputs = []
    empty_predictions = torch.zeros(10 * size)
    predictions = torch.zeros(size, 10)
    #print("empty predictions: ", empty_predictions.shape)

    for i in range(size):
        colon = colons[i]

        #print("inputs[i]: ", inputs[i].shape)
        #x_reduced = torch.cat([inputs[i], empty_predictions])
        x_reduced = inputs[i]
        #print("x_reduced: ", x_reduced.shape)

        prediction = colon.forward(x_reduced)
        #print("pred: ", prediction.shape)

        predictions[i] = prediction
        colon_outputs.append(prediction)

    #print("predictions: ", predictions.shape)
    #input()
    total_loss = 0

    for idx1, i in enumerate(colon_outputs):
        loss = torch.zeros([])
        for idx2, j in enumerate(colon_outputs):
            if idx1 == idx2:
                continue

            kl = kl_divergence(torch.log(i), j.detach())

            assert kl.item() > 0
            loss += kl

        if train:
            total_loss += loss.item()
            optimizers[idx1].zero_grad()
            loss.backward(retain_graph=True)
            optimizers[idx1].step()

    return colon_outputs, total_loss, torch.mean(predictions, dim=0)


def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    number_convolutions = 484

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    colons = []

    optimizers = []
    colons_paths = []

    for i in range(number_convolutions):
        filepath = 'colons\\colon_' + str(i) + '.model'
        predictor_model = os.path.join(script_directory, filepath)
        colons_paths.append(predictor_model)

        c = Colon(49)
        colons.append(c)

        optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
        optimizers.append(optimizer)

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):
        if iteration % 50 == 0:
            print("iteration: ", iteration)

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        colon_outputs, loss, mean = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            print()
            print("iteration: ", iteration)

            print(colon_outputs)
            print("mean: ", mean)
            print(ids)
            print(targets[ids])
            print(X_train[ids])

            total_loss = 0

            test_batch_size = 10

            test_ids = np.random.choice(len(X_test), size=test_batch_size, replace=False)

            for c, i in enumerate(test_ids):
                if c % 100 == 0:
                    print("test iteration: "+str(c))
                colon_outputs, loss, mean = forward_block(X_test, i, colons, optimizers, False, 1)

                total_loss += loss/number_convolutions

            total_loss = total_loss / test_batch_size

            if max_loss > total_loss:
                max_loss = total_loss
                print("models saved iter: " + str(iteration))
                for i in range(number_convolutions):
                    torch.save(colons[i], colons_paths[i])

            print("total loss " + str(total_loss))
            print("")


def to_Tensor(X, batch_size=BATCH_SIZE_DEFAULT):
    X = np.reshape(X, (batch_size, 1, 28, 28))
    X = Variable(torch.FloatTensor(X))

    return X


def show_mnist(first_image):
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def main():
    """
    Main function
    """
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')

    FLAGS, unparsed = parser.parse_known_args()

    main()