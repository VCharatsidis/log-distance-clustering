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
from four_variate_mi import four_variate_IID_loss
from ensemble import Ensemble
from four_variate_mi import four_variate_IID_loss
from mutual_info import IID_loss
from utils import rotate, scale, random_erease
from colon_mvmi import Colon
from vae_dec import VaeDecoder
from vae_encoder import VaeEncoder

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 400

FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def encode_4_patches(image, colons, vae_enc, vae_dec):
    i_1, i_2, i_3, i_4 = split_image_to_4(image, vae_enc, vae_dec)

    pred_1 = colons[0](i_1)
    pred_2 = colons[1](i_2)
    pred_3 = colons[2](i_3)
    pred_4 = colons[3](i_4)

    return pred_1, pred_2, pred_3, pred_4, i_4


def forward_block(X, ids, colons, optimizers, train, to_tensor_size, vae_enc, vae_dec):
    x_train = X[ids, :]

    x_tensor = to_tensor(x_train, to_tensor_size)

    images = x_tensor/255

    # pred_1, pred_2, pred_3 = encode_3_patches(images, colons)
    # loss = three_variate_IID_loss(pred_1, pred_2, pred_3)

    # pred_1, pred_2, pred_3, pred_4 = encode_4_patches(images, colons)

    pred_1, pred_2, pred_3, pred_4, i_4 = encode_4_patches(images, colons, vae_enc, vae_dec)

    loss = four_variate_IID_loss(pred_1, pred_2, pred_3, pred_4)

    # loss_1 = IID_loss(pred_1, pred_2)
    # loss_2 = IID_loss(pred_3, pred_4)

    # loss_3 = IID_loss(pred_1, pred_3)
    # loss_4 = IID_loss(pred_1, pred_4)
    #
    # loss_5 = IID_loss(pred_2, pred_3)
    # loss_6 = IID_loss(pred_2, pred_4)

    # loss_a = loss_1  #+ loss_3 + loss_4 + loss_5 + loss_6
    # loss_b = loss_2  #+ loss_3 + loss_4 + loss_5 + loss_6

    if train:
        torch.autograd.set_detect_anomaly(True)
        for i in optimizers:
            i.zero_grad()

        loss.backward(retain_graph=True)

        for i in optimizers:
            i.step()

    return pred_1, pred_2, pred_3, pred_4, loss, i_4


def split_image_to_4(image, vae_enc, vae_dec):
    # split_at_pixel = 19
    # width = image.shape[2]
    # height = image.shape[3]
    #
    # image_1 = image[:, :, 0: split_at_pixel, :]
    # image_2 = image[:, :, width - split_at_pixel:, :]
    # image_3 = image[:, :, :, 0: split_at_pixel]
    # image_4 = image[:, :, :, height - split_at_pixel:]

    # # image_1, _ = torch.split(image, split_at_pixel, dim=3)
    # # image_3, _ = torch.split(image, split_at_pixel, dim=2)
    #
    image_1 = image
    image_2 = rotate(image, 20, BATCH_SIZE_DEFAULT)
    image_3 = scale(image, BATCH_SIZE_DEFAULT)
    #image_4 = random_erease(image, BATCH_SIZE_DEFAULT)

    vae_in = torch.reshape(image, (BATCH_SIZE_DEFAULT, 784))

    sec_mean, sec_std = vae_enc(vae_in)
    e = torch.zeros(sec_mean.shape).normal_()
    sec_z = sec_std * e + sec_mean
    image_4 = vae_dec(sec_z)
    image_4 = torch.reshape(image_4, (BATCH_SIZE_DEFAULT, 1, 28, 28))

    image_1 = image_1.to('cuda')
    image_2 = image_2.to('cuda')
    image_3 = image_3.to('cuda')
    image_4 = image_4.to('cuda')

    #image = image.to('cuda')
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

    filepath = 'smart_colons\\colon_' + str(0) + '.model'
    predictor_model = os.path.join(script_directory, filepath)
    colons_paths.append(predictor_model)

    input = 8192
    #input = 3840

    # c = Ensemble()
    # c.cuda()

    c = Colon(1, input)
    c.cuda()
    colons.append(c)

    vae_enc = VaeEncoder()
    vae_dec = VaeDecoder()


    c2 = Colon(1, input)
    c2.cuda()
    colons.append(c2)

    c3 = Colon(1, input)
    c3.cuda()
    colons.append(c3)

    c4 = Colon(1, input)
    c4.cuda()
    colons.append(c4)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    optimizer_enc = torch.optim.Adam(vae_enc.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer_enc)

    optimizer_dec = torch.optim.Adam(vae_dec.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer_dec)

    optimizer2 = torch.optim.Adam(c2.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer2)

    optimizer3 = torch.optim.Adam(c3.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer3)

    optimizer4 = torch.optim.Adam(c4.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer4)

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, p3, p4, mim, i_4 = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, vae_enc, vae_dec)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, p3, p4, mim, i_4 = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, vae_enc, vae_dec)
            if iteration > 600:
                print(targets[test_ids[0]])
                show_mnist(i_4[0].cpu().detach().numpy(), 28, 28)

                print(targets[test_ids[1]])
                show_mnist(i_4[1].cpu().detach().numpy(), 28, 28)

                print(targets[test_ids[2]])
                show_mnist(i_4[2].cpu().detach().numpy(), 28, 28)

                print(targets[test_ids[3]])
                show_mnist(i_4[3].cpu().detach().numpy(), 28, 28)

                print(targets[test_ids[4]])
                show_mnist(i_4[4].cpu().detach().numpy(), 28, 28)

                print(targets[test_ids[5]])
                show_mnist(i_4[5].cpu().detach().numpy(), 28, 28)

                print(targets[test_ids[6]])
                show_mnist(i_4[6].cpu().detach().numpy(), 28, 28)

                print(targets[test_ids[7]])
                show_mnist(i_4[7].cpu().detach().numpy(), 28, 28)

                print(targets[test_ids[8]])
                show_mnist(i_4[8].cpu().detach().numpy(), 28, 28)

                print(targets[test_ids[9]])
                show_mnist(i_4[9].cpu().detach().numpy(), 28, 28)

                print(targets[test_ids[10]])
                show_mnist(i_4[10].cpu().detach().numpy(), 28, 28)

            print()
            print("iteration: ", iteration)

            print_info(p1, p2, p3, p4, 150, targets, test_ids)

            test_loss = mim.item()

            if max_loss > test_loss:
                max_loss = test_loss
                print("models saved iter: " + str(iteration))
                # for i in range(number_colons):
                #     torch.save(colons[i], colons_paths[i])

            print("test loss " + str(test_loss))
            print("")


def to_tensor(X, batch_size=BATCH_SIZE_DEFAULT):
    X = np.reshape(X, (batch_size, 1, 28, 28))
    X = Variable(torch.FloatTensor(X))

    return X


def show_mnist(first_image, w ,h):
    pixels = first_image.reshape((w, h))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def print_info(p1, p2, p3, p4, number, targets, test_ids):
    print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
    for i in range(number):
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