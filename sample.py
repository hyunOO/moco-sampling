import random

import sklearn
from sklearn.cluster import KMeans

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Sampler


def select_kcore(encoder_q, data_loader, ratio):
    curr_index = 0
    index_list = []
    for _, (data, _) in enumerate(data_loader):
        data = data.cuda()
        data_features = encoder_q(data)        
        data_features = data_features.detach().cpu().numpy()
        kmeans_model = KMeans(n_clusters=int(data.size(0) * ratio), random_state=0).fit(data_features)

        centers = kmeans_model.cluster_centers_
        for j in range(centers.shape[0]):
            k = np.argmin(np.linalg.norm(data_features - centers[j], axis=1))
            index_list.append(curr_index + k)

        curr_index += int(data.size(0))

    return index_list


def sort_loss(moco_model, data_loader, ratio, descending=True):
    curr_index = 0
    index_list = []

    for _, (im_q, im_k) in enumerate(data_loader):
        im_q, im_k = im_q.cuda(), im_k.cuda()
        q = moco_model.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = moco_model._batch_shuffle_single_gpu(im_k)

            k = moco_model.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            k = moco_model._batch_unshuffle_single_gpu(k, idx_unshuffle)

            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_pos = torch.squeeze(l_pos)

            l_pos_idx = torch.argsort(l_pos, descending=descending) 
            l_pos_idx = l_pos_idx.tolist()
            for j in range(int(im_q.size(0) * ratio)):
                index_list.append(curr_index + int(l_pos_idx[j]))

        curr_index += int(im_q.size(0))

    return index_list


class RandomSampler(Sampler):
    def __init__(self, data_source, index_list):
        self.data_source = data_source
        self.index_list = index_list

    def __iter__(self):
        random.shuffle(self.index_list)
        return iter(self.index_list)

    def __len__(self):
        return int(len(self.index_list))

