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


def sort_loss_sequential(moco_model, data_loader, sample_size):

    def contrastive_loss(im_q, im_k, negative):
        # compute query features
        q = moco_model.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        k = moco_model.encoder_k(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)

        neg = moco_model.encoder_k(negative)  # keys: NxC
        neg = nn.functional.normalize(neg, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,kc->nk', [q, neg])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= moco_model.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_list = []
        for i in range(labels.size(0)):
            logit = torch.unsqueeze(logits[i], 0)
            label = torch.unsqueeze(labels[i], 0)
            loss = nn.CrossEntropyLoss().cuda()(logit, label)
            loss_list.append(loss.item())

        return loss_list

    index_list = []
    # loss_list = [0 for i in range(len(data_loader.dataset))]

    new_index = 0
    losses = None

    while len(index_list) < sample_size:
        print('index_list: ', index_list)
        if len(index_list) == 0:
            index = random.choice([i for i in range(len(data_loader.dataset))])
            index_list.append(index)
            new_index = index
        else:
            start_time = timer()
            # max_index = 0
            # max_contrastive_loss = 0
            contrastive_loss_list = []

            for _, (im_q, im_k) in enumerate(data_loader):
                # im_q = torch.unsqueeze(im_q, 0)
                im_q = im_q.cuda()
                im_k = im_k.cuda()
                negative = data_loader.dataset[new_index][0]
                negative = torch.unsqueeze(negative, 0)
                negative = negative.cuda()
                c_loss = contrastive_loss(im_q, im_k, negative)
                contrastive_loss_list.extend(c_loss)

            contrastive_loss_list = torch.tensor(contrastive_loss_list)
            if losses is None:
                losses = contrastive_loss_list.cpu()
            else:
                losses += contrastive_loss_list.cpu()
            print('losses shape: ', losses.shape)

            sort_index = torch.argsort(losses, descending=True)
            for index in sort_index:
                index_int = int(index)
                if index_int in index_list:
                    continue
                else:
                    index_list.append(index_int)
                    new_index = index_int
                    break

            end_time = timer()
            print(f'Elapsed time: {end_time - start_time}s')

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

