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
from utils import rotate, scale, to_gray, random_erease, vertical_flip
from colon_mvmi import Colon

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 100
EVAL_FREQ_DEFAULT = 200
NUMBER_CLASSES = 10
FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def encode_4_patches(image, colons,
                     p1=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p2=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p3=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p4=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     ):

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

    image = image[:, :, 5:90, 5:90]
    image_1 = image[:, :, 0: split_at_pixel, 0: split_at_pixel]
    image_2 = image[:, :, 85 - split_at_pixel:, 0: split_at_pixel]
    image_3 = image[:, :, 0: split_at_pixel, 0: split_at_pixel]
    image_4 = image[:, :, 85 - split_at_pixel:, 85 - split_at_pixel:]

    patches = {0: image_1,
               1: image_2,
               2: image_3,
               3: image_4}

    patch_ids = np.random.choice(len(patches), size=4, replace=False)

    augments = {0: to_gray(patches[patch_ids[0]], BATCH_SIZE_DEFAULT),
                1: rotate(patches[patch_ids[1]], 20, BATCH_SIZE_DEFAULT),
                2: rotate(patches[patch_ids[2]], -20, BATCH_SIZE_DEFAULT),
                3: scale(patches[patch_ids[3]], 40, 5, BATCH_SIZE_DEFAULT),
                4: vertical_flip(patches[patch_ids[0]], BATCH_SIZE_DEFAULT),
                5: scale(patches[patch_ids[1]], 30, 10, BATCH_SIZE_DEFAULT),
                6: random_erease(patches[patch_ids[2]], BATCH_SIZE_DEFAULT),
                7: patches[patch_ids[3]]}

    ids = np.random.choice(len(augments), size=4, replace=False)

    image_1 = augments[ids[0]]
    image_2 = augments[ids[1]]
    image_3 = augments[ids[2]]
    image_4 = augments[ids[3]]

    # image_4 = torch.transpose(image_4, 1, 3)
    # show_mnist(image_4[0], image_4[0].shape[1], image_4[0].shape[2])
    # image_4 = torch.transpose(image_4, 1, 3)
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

    # image_a, image_b = torch.split(image, width//2, dim=2)
    #
    # image_1, image_2 = torch.split(image_a, height//2, dim=3)
    # image_3, image_4 = torch.split(image_b, height//2, dim=3)

    images = [image_1, image_2, image_3, image_4]

    p1 = p1.to('cuda')
    p2 = p2.to('cuda')
    p3 = p3.to('cuda')
    p4 = p4.to('cuda')

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

    prod = torch.ones([BATCH_SIZE_DEFAULT, NUMBER_CLASSES])
    prev_preds = [p1, p2, p3, p4]
    preds = []

    z = images[0].to('cuda')
    pred = colons[0](z, p2, p3, p4)
    preds.append(pred)
    prod *= pred.to('cpu')

    z = images[1].to('cuda')
    pred = colons[0](z, p1, p3, p4)
    preds.append(pred)
    prod *= pred.to('cpu')

    z = images[2].to('cuda')
    pred = colons[0](z, p1, p2, p4)
    preds.append(pred)
    prod *= pred.to('cpu')

    z = images[3].to('cuda')
    pred = colons[0](z, p1, p2, p3)
    preds.append(pred)
    prod *= pred.to('cpu')


    # for idx, i in enumerate(images):
    #     z = i.to('cuda')
    #
    #     pred = colons[0](z,)
    #
    #     preds.append(pred)
    #     prod *= pred.to('cpu')

    return prod, preds



def forward_block(X, ids, colons, optimizers, train, to_tensor_size,
                  p1=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p2=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p3=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p4=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  ):
    x_train = X[ids, :]

    x_tensor = to_tensor(x_train, to_tensor_size)

    images = x_tensor/255.0

    product, preds = encode_4_patches(images, colons, p1, p2, p3, p4)

    product = product.mean(dim=0)
    log_product = torch.log(product)

    loss = -log_product.mean(dim=0)

    if train:
        torch.autograd.set_detect_anomaly(True)

        for i in optimizers:
            i.zero_grad()

        loss.backward(retain_graph=True)

        for i in optimizers:
            i.step()

    pred_1, pred_2, pred_3, pred_4 = preds
    return pred_1, pred_2, pred_3, pred_4, loss


def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():
    #
    fileName = "data\\train_X.bin"
    X_train = read_all_images(fileName)

    testFile = "data\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "data\\test_y.bin"
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

    input = 4126
    #input = 1152

    # c = Ensemble()
    # c.cuda()

    c = EncoderSTL(3, input)
    c.cuda()
    colons.append(c)

    # c2 = EncoderSTL(3, input)
    # c2.cuda()
    # colons.append(c2)
    #
    # c3 = EncoderSTL(3, input)
    # c3.cuda()
    # colons.append(c3)
    #
    # c4 = EncoderSTL(3, input)
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
        p1, p2, p3, p4, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)
        p1, p2, p3, p4, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)

        if iteration % EVAL_FREQ_DEFAULT == 0:

            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)

            p1, p2, p3, p4, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)
            print("loss 1: ", mim.item())
            p1, p2, p3, p4, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)
            print("loss 2: ", mim.item())
            p1, p2, p3, p4, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)
            print("loss 3: ", mim.item())

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
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


def show_mnist(first_image, w, h):
    #pixels = first_image.reshape((w, h))
    pixels = first_image
    plt.imshow(pixels)
    plt.show()


def print_info(p1, p2, p3, p4, targets, test_ids):
    #print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
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