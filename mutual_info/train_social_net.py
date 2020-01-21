from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch

from torch.autograd import Variable
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from three_variate_mutual_info import three_variate_IID_loss
from four_variate_mi import four_variate_IID_loss
from ensemble import Ensemble
from utils import scale, rotate, random_erease

from SocialColon import SocialColon

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 100
EVAL_FREQ_DEFAULT = 100

FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def encode_4_patches(image,
                     colons,
                     p1=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p2=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p3=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p4=torch.zeros([BATCH_SIZE_DEFAULT, 10])):

    i_1, i_2, i_3, i_4 = split_image_to_4(image)

    p1 = p1.to('cuda')
    p2 = p2.to('cuda')
    p3 = p3.to('cuda')
    p4 = p4.to('cuda')

    pred_1 = colons[0](i_1, p2, p3, p4)
    pred_2 = colons[0](i_2, p1, p3, p4)
    pred_3 = colons[0](i_3, p1, p2, p4)
    pred_4 = colons[0](i_4, p1, p2, p3)

    return pred_1, pred_2, pred_3, pred_4



def forward_block(X, ids, colons, optimizers, train, to_tensor_size):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, to_tensor_size)

    images = x_tensor/255

    # pred_1, pred_2, pred_3 = encode_3_patches(images, colons)
    # loss = three_variate_IID_loss(pred_1, pred_2, pred_3)

    pred_1, pred_2, pred_3, pred_4 = encode_4_patches(images, colons)
    loss = four_variate_IID_loss(pred_1, pred_2, pred_3, pred_4)

    if train:
        optimizers[0].zero_grad()
        loss.backward(retain_graph=True)
        optimizers[0].step()

    return pred_1, pred_2, pred_3, pred_4, loss


def split_image_to_3(images):
    image_shape = images.shape

    image_a, image_b = torch.split(images, image_shape[2] // 2, dim=3)
    image_3, image_4 = torch.split(images, image_shape[2] // 2, dim=2)

    # image_a = image_a.to('cuda')
    # image_b = image_b.to('cuda')
    # image_4 = image_4.to('cuda')

    #images = images.to('cuda')

    # print(images.shape)
    # print("image a batch: ", image_a.shape)
    # print("image b batch: ", image_b.shape)
    # print("image 3 batch: ", image_3.shape)
    # print("image 4 batch: ", image_4.shape)

    return image_a, image_b, image_4

def split_image_to_4(image):
    image_1 = image
    image_2 = rotate(image, 20, BATCH_SIZE_DEFAULT)
    image_3 = scale(image, BATCH_SIZE_DEFAULT)
    image_4 = random_erease(image, BATCH_SIZE_DEFAULT)

    image_1 = image_1.to('cuda')
    image_2 = image_2.to('cuda')
    image_3 = image_3.to('cuda')
    image_4 = image_4.to('cuda')

    # image = image.to('cuda')
    # show_mnist(image_1[0], 20, 28)
    # show_mnist(image_1[1], 20, 28)
    # show_mnist(image_1[2], 20, 28)
    # show_mnist(image_1[3], 20, 28)
    #
    # show_mnist(image_2[0], 20, 28)
    # show_mnist(image_2[1], 20, 28)
    # show_mnist(image_2[2], 20, 28)
    # show_mnist(image_2[3], 20, 28)
    #
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


def second_guess(X, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3, p4):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, BATCH_SIZE_DEFAULT)

    images = x_tensor / 255

    # pred_1, pred_2, pred_3 = encode_3_patches(images, colons)
    # loss = three_variate_IID_loss(pred_1, pred_2, pred_3)

    pred_1, pred_2, pred_3, pred_4 = encode_4_patches(images, colons, p1, p2, p3, p4)
    loss = four_variate_IID_loss(pred_1, pred_2, pred_3, pred_4)

    if train:
        optimizers[0].zero_grad()
        loss.backward(retain_graph=True)
        optimizers[0].step()

    return pred_1, pred_2, pred_3, pred_4, loss


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target[60000:]

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    number_colons = 4

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    colons = []

    optimizers = []
    colons_paths = []

    filepath = 'colons\\colon_' + str(0) + '.model'
    predictor_model = os.path.join(script_directory, filepath)
    colons_paths.append(predictor_model)

    #four_split = 3200
    preds = 30
    two_split = 6400 + preds

    #two_split_3_conv = 3840

    # c = Ensemble()
    # c.cuda()

    c = SocialColon(1, two_split)
    c.cuda()
    colons.append(c)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, p3, p4, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)
        p1, p2, p3, p4, mim = second_guess(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)
        p1, p2, p3, p4, mim = second_guess(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, p3, p4, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)
            p1, p2, p3, p4, mim = second_guess(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)
            p1, p2, p3, p4, mim = second_guess(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)

            print()
            print("iteration: ", iteration)

            print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
            for i in range(100):
                if i == 10:
                    print("")

                val, index = torch.max(p1[i], 0)
                val, index2 = torch.max(p2[i], 0)
                val, index3 = torch.max(p3[i], 0)
                val, index4 = torch.max(p4[i], 0)

                string = str(index.data.cpu().numpy())+" "+ str(index2.data.cpu().numpy()) + " "+\
                         str(index3.data.cpu().numpy())+" "+ str(index4.data.cpu().numpy()) +", "

                print_dict[targets[test_ids[i]]] += string


            for i in print_dict.keys():
                print(i, " : ", print_dict[i])

            test_loss = mim.item()

            if max_loss > test_loss:
                max_loss = test_loss
                print("models saved iter: " + str(iteration))
                # for i in range(number_colons):
                #     torch.save(colons[i], colons_paths[i])

            print("test loss " + str(test_loss))
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
