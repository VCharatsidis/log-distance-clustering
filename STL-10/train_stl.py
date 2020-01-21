from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch

from torch.autograd import Variable
import matplotlib.pyplot as plt
from stl10_input import read_all_images, read_labels
from encoderSTL import EncoderSTL
import random

from sklearn.datasets import fetch_openml
from four_variate_mi import four_variate_IID_loss
from ensemble import Ensemble
from four_variate_mi import four_variate_IID_loss
from mutual_info import IID_loss
from utils import rotate, scale, random_erease
from colon_mvmi import Colon

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 60
EVAL_FREQ_DEFAULT = 200

FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def encode_4_patches(image, colons):
    i_1, i_2, i_3, i_4 = split_image_to_4(image)

    pred_1 = colons[0](i_1)
    pred_2 = colons[0](i_2)
    pred_3 = colons[0](i_3)
    pred_4 = colons[0](i_4)

    return pred_1, pred_2, pred_3, pred_4


def forward_block(X, ids, colons, optimizers, train, to_tensor_size):
    x_train = X[ids, :]

    x_tensor = to_tensor(x_train, to_tensor_size)

    images = x_tensor/255.0

    pred_1, pred_2, pred_3, pred_4 = encode_4_patches(images, colons)

    product = pred_1 * pred_2 * pred_3 * pred_4
    log = torch.log(product)
    log_sum = log.sum(dim=1)

    loss = - log_sum.mean(dim=0)

    if train:
        torch.autograd.set_detect_anomaly(True)

        for i in optimizers:
            i.zero_grad()

        loss.backward(retain_graph=True)

        for i in optimizers:
            i.step()

    return pred_1, pred_2, pred_3, pred_4, loss


def split_image_to_4(image):
    split_at_pixel = 50
    width = image.shape[2]
    height = image.shape[3]

    image_1 = image[:, :, 0: split_at_pixel, :]
    image_2 = image[:, :, width - split_at_pixel:, :]
    image_3 = image[:, :, :, 0: split_at_pixel]
    image_4 = image[:, :, :, height - split_at_pixel:]

    # image_1 = image
    # image_2 = rotate(image, 20, BATCH_SIZE_DEFAULT)
    # image_3 = scale(image, BATCH_SIZE_DEFAULT)
    # image_4 = random_erease(image, BATCH_SIZE_DEFAULT)

    image_1 = image_1.to('cuda')
    image_2 = image_2.to('cuda')
    image_3 = image_3.to('cuda')
    image_4 = image_4.to('cuda')

    #image = image.to('cuda')

    # input()
    # print(image_1.shape)
    # print(image_2.shape)
    # print(image_3.shape)
    # print(image_4.shape)
    # input()

    return image_1, image_2, image_3, image_4


def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():

    fileName = "data\\train_X.bin"
    X_train = read_all_images(fileName)

    testFile = "data\\test_X.bin"
    X_test = read_all_images(testFile)

    # train_y_File = "data\\train_y.bin"
    # train_y = read_labels(train_y_File)

    test_y_File = "data\\test_y.bin"
    targets = read_labels(test_y_File)

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    colons = []

    optimizers = []
    colons_paths = []

    filepath = 'encoders\\encoder_' + str(0) + '.model'
    predictor_model = os.path.join(script_directory, filepath)
    colons_paths.append(predictor_model)

    input = 13312
    #input = 3840

    # c = Ensemble()
    # c.cuda()

    c = EncoderSTL(3, input)
    c.cuda()
    colons.append(c)

    # c2 = Colon(1, input)
    # c2.cuda()
    # colons.append(c2)
    #
    # c3 = Colon(1, input)
    # c3.cuda()
    # colons.append(c3)
    #
    # c4 = Colon(1, input)
    # c4.cuda()
    # colons.append(c4)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    # optimizer2 = torch.optim.Adam(c2.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer2)
    #
    # optimizer3 = torch.optim.Adam(c3.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer3)
    #
    # optimizer4 = torch.optim.Adam(c4.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer4)

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, p3, p4, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, p3, p4, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)
            print()
            print("iteration: ", iteration)

            print(p1[0])
            print(p2[0])
            print(p3[0])
            print(p4[0])

            print_info(p1, p2, p3, p4, targets, test_ids)

            test_loss = mim.item()

            if max_loss > test_loss:
                max_loss = test_loss
                print("models saved iter: " + str(iteration))
                # for i in range(number_colons):
                #     torch.save(colons[i], colons_paths[i])

            print("test loss " + str(test_loss))
            print("")


def to_tensor(X, batch_size=BATCH_SIZE_DEFAULT):
    #X = np.reshape(X, (batch_size, 1, 96, 96))
    X = Variable(torch.FloatTensor(X))

    return X


def show_mnist(first_image, w, h):
    pixels = first_image.reshape((w, h))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def print_info(p1, p2, p3, p4, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 10: ""}
    for i in range(p1.shape[0]):
        if i == 10:
            print("")

        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)
        val, index3 = torch.max(p3[i], 0)
        val, index4 = torch.max(p4[i], 0)

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + " " + \
                 str(index3.data.cpu().numpy()) + " " + str(index4.data.cpu().numpy()) + " , "

        label = targets[test_ids[i]]
        print_dict[label] += string

    for i in print_dict.keys():
        print(i, " : ", print_dict[i])


def main():
    """
    Main function
    """
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

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