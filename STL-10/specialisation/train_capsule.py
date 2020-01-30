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

import random
from encoderSpecialisation import Specialist

from sklearn.datasets import fetch_openml
from four_variate_mi import four_variate_IID_loss
from ensemble import Ensemble
from four_variate_mi import four_variate_IID_loss
from mutual_info import IID_loss
from utils import rotate, scale, to_gray, random_erease, vertical_flip
from colon_mvmi import Colon
from capsule_net import CapsNet

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 128
EVAL_FREQ_DEFAULT = 100
NUMBER_CLASSES = 1
FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def encode_4_patches(image, colons):

    #split_at_pixel = 19

    # image = np.reshape(image, (BATCH_SIZE_DEFAULT, 1, 28, 28))
    # image = torch.FloatTensor(image)

    # print(image.shape)
    # print(image.shape[2])
    # print(image.shape[3])
    # input()
    width = image.shape[2]
    height = image.shape[3]

    split_at_pixel = 50

    image = image[:, :, 20:70, 20:70]

    augments = {0: to_gray(image, BATCH_SIZE_DEFAULT),
                1: rotate(image, 20, BATCH_SIZE_DEFAULT),
                2: rotate(image, -20, BATCH_SIZE_DEFAULT),
                3: scale(image, 40, 5, BATCH_SIZE_DEFAULT),
                #3: scale(image, 32, 4, BATCH_SIZE_DEFAULT),
                4: vertical_flip(image, BATCH_SIZE_DEFAULT),
                5: scale(image, 30, 10, BATCH_SIZE_DEFAULT),
                #5: scale(image, 24, 8, BATCH_SIZE_DEFAULT),
                6: random_erease(image, BATCH_SIZE_DEFAULT),
                7: vertical_flip(image, BATCH_SIZE_DEFAULT)}

    ids = np.random.choice(len(augments), size=1, replace=False)

    image_2 = augments[ids[0]]

    # image = torch.transpose(image, 1, 3)
    # show_mnist(image[0], image[0].shape[1], image[0].shape[2])
    # image = torch.transpose(image, 1, 3)
    #
    # image_2 = torch.transpose(image_2, 1, 3)
    # show_mnist(image_2[0], image_2[0].shape[1], image_2[0].shape[2])
    # image_2 = torch.transpose(image_2, 1, 3)
    #
    # image_3 = torch.transpose(image_3, 1, 3)
    # show_mnist(image_3[0], image_3[0].shape[1], image_3[0].shape[2])
    # image_3 = torch.transpose(image_3, 1, 3)
    #
    # image_1 = torch.transpose(image_1, 1, 3)
    # show_mnist(image_1[0], image_1[0].shape[1], image_1[0].shape[2])
    # image_1 = torch.transpose(image_1, 1, 3)

    image = image.to('cuda')
    image_2 = image_2.to('cuda')

    output, c1, c2, c3, c4, c5, c6, c7, c8, c9, c0 = colons[0](image)
    output, a_c1, a_c2, a_c3, a_c4, a_c5, a_c6, a_c7, a_c8, a_c9, a_c0 = colons[0](image_2)

    comb1 = (c1 + a_c1) / 2
    comb2 = (c2 + a_c2) / 2
    comb3 = (c3 + a_c3) / 2
    comb4 = (c4 + a_c4) / 2
    comb5 = (c5 + a_c5) / 2
    comb6 = (c6 + a_c6) / 2
    comb7 = (c7 + a_c7) / 2
    comb8 = (c8 + a_c8) / 2
    comb9 = (c9 + a_c9) / 2
    comb0 = (c0 + a_c0) / 2

    new_preds =[comb1.to('cpu'), comb2.to('cpu'), comb3.to('cpu'), comb4.to('cpu'), comb5.to('cpu'), comb6.to('cpu'), comb7.to('cpu'), comb8.to('cpu'), comb9.to('cpu'), comb0.to('cpu')]

    products = []
    for prod in range(10):
        product = torch.ones([BATCH_SIZE_DEFAULT, 1])

        for idx, prediction in enumerate(new_preds):
            if idx != prod:
                product *= torch.ones(prediction.shape) - prediction
            else:
                product *= prediction

        products.append(product)



    # print(len(products))
    # print(products[0])
    # print(products[0].shape)
    #
    # input()
    # image_1 = image
    # image_2 = random_erease(image, BATCH_SIZE_DEFAULT)
    # image_2 = torch.transpose(image_2, 1, 3)
    # print(image_2.shape)
    # show_mnist(image_2[0], image_2[0].shape[1], image_2[0].shape[2])
    #
    # image_3 = scale(image, BATCH_SIZE_DEFAULT)
    # show_mnist(image_3[0], image_3[0].shape[1], image_3[0].shape[2])
    #
    # image_4 = random_erease(image, BATCH_SIZE_DEFAULT)
    # show_mnist(image_4[0], image_4.shape[1], image_4.shape[2])


    return products


def forward_block(X, ids, colons, optimizers, train, to_tensor_size):

    x_train = X[ids, :]

    x_tensor = to_tensor(x_train, to_tensor_size)

    images = x_tensor/255.0

    products = encode_4_patches(images, colons)

    losses = []
    total_loss = torch.zeros([1])

    for p in products:
        mean = p.mean(dim=0)
        log_p = -torch.log(mean)
        losses.append(log_p)
        total_loss += log_p

    total_loss = total_loss / len(products)

    if train:
        torch.autograd.set_detect_anomaly(True)

        for idx, i in enumerate(optimizers):
            i.zero_grad()

        total_loss.backward(retain_graph=True)

        for idx, i in enumerate(optimizers):
            i.step()

    return products, total_loss


def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():
    #
    fileName = "..\\data\\train_X.bin"
    X_train = read_all_images(fileName)

    testFile = "..\\data\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "..\\data\\test_y.bin"
    targets = read_labels(test_y_File)

    # mnist = fetch_openml('mnist_784', version=1, cache=True)
    # targets = mnist.target[60000:]
    #
    # X_train = mnist.data[:60000]
    # X_test = mnist.data[60000:]

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    colons = []

    optimizers = []
    colons_paths = []

    filepath = 'encoders\\encoder_' + str(0) + '.model'
    predictor_model = os.path.join(script_directory, filepath)
    colons_paths.append(predictor_model)

    c = CapsNet()
    c = c.cuda()
    colons.append(c)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    max_loss = 10000000

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        products, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            # print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
            print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}

            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            products, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)

            # test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            # products, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)
            # print_dict = gather_data(print_dict, products, targets, test_ids)
            #
            # test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            # products, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)
            # print_dict = gather_data(print_dict, products, targets, test_ids)

            print("loss 1: ", mim.item())
            products, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)

            print()
            print("iteration: ", iteration)

            print_dict = gather_data(print_dict, products, targets, test_ids)
            print_info(print_dict)

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
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


def show_mnist(first_image, w, h):
    #pixels = first_image.reshape((w, h))
    pixels = first_image
    plt.imshow(pixels)
    plt.show()

def gather_data(print_dict, products, targets, test_ids):
    for i in range(products[0].shape[0]):
        res = ""

        for specialist in products:

            index = torch.round(specialist[i])

            res += str(int(index.data.cpu().numpy()[0])) + " "

        res += ", "

        label = targets[test_ids[i]]
        print_dict[label] += res

    return print_dict


def print_info(print_dict):
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