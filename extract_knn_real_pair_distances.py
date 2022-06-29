import torch
import numpy as np
from tqdm import tqdm
import operator
from multiprocessing import Pool


FOLDER = "D:\\GANs\\Datasets\\Various\\affectnet_src.tar\\affectnet_src\\affectnet"


def calculate_real_NNK(real_features, k, data_num):
    KNN_list_in_real = {}

    # calculate KNN for each real image in real_features with Pool and imap

    for real_feature in tqdm(real_features, ncols=80):
        pairwise_distances = np.zeros(shape=(len(real_features)))

        for i, real_prime in enumerate(real_features):
            d = torch.norm((real_feature-real_prime), 2)
            pairwise_distances[i] = d

        v = np.partition(pairwise_distances, k)[k]
        KNN_list_in_real[real_feature] = v

    # remove half of larger values
    KNN_list_in_real = sorted(KNN_list_in_real.items(), key=operator.itemgetter(1))
    KNN_list_in_real = KNN_list_in_real[:int(data_num/2)]

    return KNN_list_in_real


real_feat = torch.load(FOLDER + "\\real_embeddings_vgg_pr_re.pt")[::7]

KNN = calculate_real_NNK(real_feat, 3, len(real_feat))

torch.save(KNN, FOLDER + "\\KNN_list_in_real.pt")
