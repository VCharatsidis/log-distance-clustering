
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.datasets import fetch_openml
from train import forward_block
import os
from conv import DetachedConvNet
import copy
from sklearn.cluster import KMeans
from train import to_Tensor
from copy import deepcopy


script_directory = os.path.split(os.path.abspath(__file__))[0]
detached_net_model = os.path.join(script_directory, 'detached_net.model')
conv = torch.load(detached_net_model)

colons = []
for i in range(784):
    path = 'colons\\colon_' + str(i) + '.model'
    colon = os.path.join(script_directory, path)
    colons.append(torch.load(colon))

mnist = fetch_openml('mnist_784', version=1, cache=True)
targets = mnist.target[60000:]

X_train = mnist.data[:60000]
X_test = mnist.data[60000:]

fake_optimizers = []


def loss_representations(member_ids):
    with torch.no_grad():
        res = forward_block(X_test, member_ids, conv, colons, fake_optimizers, False)
        return res


def miss_classifications(cluster):
    mfe = most_frequent(cluster)
    missclassifications = 0
    for j in cluster:
        if j != mfe:
            missclassifications += 1

    return missclassifications


def get_centroids(member_numbers):
    member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
    X = []
    for i in member_ids:
        X.append(loss_representations(i))

    # X1 = copy.deepcopy(X)
    # X1 = np.array(X1)
    # sort2 = X1.argsort(1)
    #
    # for c, i in enumerate(X):
    #     s = sort2[c].tolist()
    #     s = s[-200:]
    #
    #     s.sort()
    #     X[c] = [i[z] for z in s]
    #     print(X[c])
    #
    # X = np.array(X)
    # print(X.shape)

    predict = KMeans(n_clusters=10).fit_predict(X)

    clusters_to_ids = {}

    for i, p in enumerate(predict):
        if p in clusters_to_ids.keys():
            clusters_to_ids[p].append(targets[member_ids[i]])
        else:
            idxs = []
            idxs.append(targets[member_ids[i]])
            clusters_to_ids[p] = idxs

    avg = 0
    for c in clusters_to_ids.keys():
        mfe = most_frequent(clusters_to_ids[c])
        counter = 0
        for i in clusters_to_ids[c]:
            if i != mfe:
                counter += 1

        cluster_size = len(clusters_to_ids[c])
        print("cluster size " + str(cluster_size))
        percentage_miss = (counter * 100) / cluster_size
        avg += percentage_miss
        print("clsuter: " + str(c) + " mfe " + str(mfe) + " miss percentage " + str(percentage_miss))

    print("avg  miss: " + str(avg / 10))

    print(clusters_to_ids[0])
    print(clusters_to_ids[1])
    print(clusters_to_ids[2])
    print(clusters_to_ids[3])
    print(clusters_to_ids[4])
    print(clusters_to_ids[5])
    print(clusters_to_ids[6])
    print(clusters_to_ids[7])
    print(clusters_to_ids[8])
    print(clusters_to_ids[9])


def most_frequent(List):
    return max(set(List), key=List.count)


def log_distance_pairing(member_numbers):
    conv = DetachedConvNet(1, 1, 1)
    member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
    X = X_test[member_ids]

    sigmoided = []
    for i in X:
        z = to_Tensor(i, 1)
        sigmoided.append(conv.forward(z).detach().numpy())

    print(sigmoided[0])
    X = np.array(sigmoided)

    # X = []
    # for i in member_ids:
    #     X.append(loss_representations(i))
    #
    # X = np.array(X)


    miss = 0
    used = []

    print("done with representations")
    clusters = []
    for i in range(member_numbers):
        if i % 500 == 0:
            print("member number: "+str(i))

        min = 1000000
        min_idx = -1
        if i in used:
            continue

        cluster = []
        for j in range(member_numbers):

            if i == j:
                continue

            log_dist = np.log(1 - np.abs(X[i] - X[j]))

            sum = np.abs(log_dist.sum())

            if sum < min:
                min = sum
                min_idx = j

        cluster.append(i)
        cluster.append(min_idx)

        clusters.append(cluster)
        used.append(min_idx)
        if targets[member_ids[i]] != targets[member_ids[min_idx]]:
            miss += 1

    print("missclassifications log distance best friend clustering: " + str(miss) + " percentage miss: " + str(
        miss / member_numbers))


def log_distance_clustering(clusters, distances):

    new_clusters = []
    used = []
    for idx, c in enumerate(clusters):
        if idx in used:
            continue

        min = 10000000
        closest_cluster = -1

        for idx2, c2 in enumerate(clusters):
            cluster_sum = 0

            if idx == idx2:
                continue

            for m in c:
                member_sum = 0

                for m2 in c2:
                    if m[0] == m2[0]:
                        print(idx)
                        print(idx2)
                        print(len(c))
                        print(len(c2))
                        input()
                    key = str(m[0])+'_'+str(m2[0])
                    member_sum += distances[key]

                normalized = member_sum / len(c2)
                cluster_sum += normalized

            if cluster_sum < min:
                min = cluster_sum
                closest_cluster = idx2

        if closest_cluster not in used:
            used.append(closest_cluster)

        for m in c:
            clusters[closest_cluster].append(m)

    for c in used:
        copied_cluster = copy.deepcopy(clusters[c])
        new_clusters.append(copied_cluster)

    return new_clusters


def calculate_distances(sigmoided_data):
    distances = {}
    for idx, i in enumerate(sigmoided_data):
        for idx2, j in enumerate(sigmoided_data):
            if idx >= idx2:
                continue

            distance = np.log(1 - np.abs(i[1] - j[1]))
            sum = np.abs(distance.sum())

            key1 = str(i[0])+'_'+str(j[0])
            distances[key1] = sum

            key2 = str(j[0])+'_'+str(i[0])
            distances[key2] = sum

    return distances


def call_log_dist_clustering(member_numbers, cluster_number):
    member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
    X = X_test[member_ids]

    sigmoided = []
    idx_data = []
    for counter, i in enumerate(X):
        list = []
        z = to_Tensor(i, 1)
        convolved = conv.forward(z).numpy()[0][0]
        image_and_index = (member_ids[counter], convolved)
        idx_data.append(image_and_index)
        list.append(image_and_index)
        sigmoided.append(list)

    distances = calculate_distances(idx_data)

    counter = 0
    while len(sigmoided) > cluster_number:
        print("clustering iteration: "+str(counter))
        counter += 1
        sigmoided = log_distance_clustering(sigmoided, distances)

    for c in sigmoided:
        labels = []
        for m in c:
            labels.append(targets[m[0]])
        print(labels)

call_log_dist_clustering(200,15)
#log_distance_pairing(500)
#get_centroids(10000)
