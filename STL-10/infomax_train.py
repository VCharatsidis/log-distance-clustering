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

from utils import rotate, scale, to_gray, random_erease
from infomax_encoder import InfoEncoder

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 700
EVAL_FREQ_DEFAULT = 200
NUMBER_CLASSES = 10
FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def forward_block(X, ids, colon, optimizer, train, to_tensor_size):
    x_train = X[ids, :]

    x_tensor = to_tensor(x_train, to_tensor_size)

    images = x_tensor/255.0

    images = images[:, :, 15:75, 15:75]
    # images = torch.transpose(images, 1, 3)
    # show_mnist(images[0], images[0].shape[1], images[0].shape[2])
    # images = torch.transpose(images, 1, 3)

    images = images.to('cuda')
    preds = colon(images)

    product = preds * preds
    product = product.mean(dim=0)
    log_product = torch.log(product)
    loss = -log_product.mean(dim=0)

    if train:
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    return preds, loss


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

    colons_paths = []

    filepath = 'encoders\\encoder_' + str(0) + '.model'
    predictor_model = os.path.join(script_directory, filepath)
    colons_paths.append(predictor_model)

    input = 1024
    #input = 1152

    # c = Ensemble()
    # c.cuda()

    c = InfoEncoder(3, input)
    c.cuda()

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)


    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, mim = forward_block(X_train, ids, c, optimizer, train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:

            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)

            p1, mim = forward_block(X_test, test_ids, c, optimizer, False, BATCH_SIZE_DEFAULT)
            print("loss : ", mim.item())

            print()
            print("iteration: ", iteration)

            print(p1[0])

            print_info(p1, targets, test_ids)

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


def print_info(p1, targets, test_ids):
    #print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 10: ""}
    for i in range(p1.shape[0]):
        if i == 10:
            print("")

        val, index = torch.max(p1[i], 0)

        string = str(index.data.cpu().numpy()) + " , "

        label = targets[test_ids[i]]
        print_dict[label] += string

    for i in print_dict.keys():
        z = print_dict[i].split(',')
        print("most frequent: ", most_frequent(z))
        miss = miss_classifications(z)
        print("missclasifications: ", miss, " percentage: ", miss/len(z))
        print(i, " : ", print_dict[i])


def miss_classifications(cluster):
    mfe = most_frequent(cluster)
    missclassifications = 0
    for j in cluster:
        if j != mfe:
            missclassifications += 1

    return missclassifications


def most_frequent(List):
    return max(set(List), key=List.count)


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