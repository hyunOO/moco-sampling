import random
from timeit import default_timer as timer

import sklearn
from sklearn.cluster import KMeans

from kmeans_pytorch import kmeans

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Sampler


def select_kcore(encoder_q, data_loader, ratio):
    curr_index = 0
    index_list = []

    for i, (data, _) in enumerate(data_loader):
        data = data.cuda()
        device = data.device
        data_features = encoder_q(data)        
        data_features = data_features.detach().cpu().numpy()

        size = int(data.size(0) * ratio)
        kmeans_model = KMeans(n_clusters=size, random_state=0).fit(data_features)

        centers = kmeans_model.cluster_centers_
        for j in range(centers.shape[0]):
            k = np.argmin(np.linalg.norm(data_features - centers[j], axis=1))
            index_list.append(curr_index + k)

        curr_index += int(data.size(0))

    return index_list


def sort_loss(moco_model, data_loader, sample_size, descending=True):
    curr_index = 0
    index_list = []

    loss_list = []
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

            for j in range(l_pos.size(0)):
                loss_list.append(l_pos[j])

    loss_tensor = torch.tensor(loss_list) 
    loss_tensor = torch.squeeze(loss_tensor)
    index_list = torch.argsort(loss_tensor, descending=descending)
    index_list = index_list[:sample_size]  

    return index_list


def sort_loss_sequential(moco_model, dataset, sample_size):
    index_list = []
    loss_list = [0 for i in range(len(dataset))]

    while len(index_list) < sample_size:
        print('index_list: ', index_list)
        if len(index_list) == 0:
            index = random.choice([i for i in range(len(dataset))])
            index_list.append(index)
        else:
            start_time = timer()
            max_index = 0
            max_contrastive_loss = 0

            for i in range(len(dataset)):
                if i in index_list:
                    continue
                else:
                    im_q = dataset[i][0]
                    im_q = torch.unsqueeze(im_q, 0)
                    im_q = im_q.cuda()
                    contrastive_loss = loss_list[i]

                    im_k = dataset[index_list[-1]][0]
                    im_k = torch.unsqueeze(im_k, 0)
                    im_k = im_k.cuda()
                    contrastive_loss += moco_model.contrastive_loss(im_q, im_k)[0].item()
                    loss_list[i] = contrastive_loss

                    if max_contrastive_loss < contrastive_loss:
                        max_contrastive_loss = contrastive_loss
                        max_index = i
            end_time = timer()
            print(f'Elapsed time: {end_time - start_time}s')

            index_list.append(max_index)

    return index_list


class RandomSampler(Sampler):
    def __init__(self, data_source, index_list, shuffle=True):
        self.data_source = data_source
        self.index_list = index_list
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.index_list)
        return iter(self.index_list)

    def __len__(self):
        return int(len(self.index_list))

