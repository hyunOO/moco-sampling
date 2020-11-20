import os
import json
import math
import pandas as pd
from tqdm import tqdm

from utils import SplitBatchNorm
from utils import knn_predict
from parse_args import args
from dataloader import CIFAR10Pair
from models.moco_model import ModelMoCo

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from torch.utils.tensorboard import SummaryWriter

from sample import select_kcore
from sample import sort_loss
from sample import RandomSampler


# train for one epoch
def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2 in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)

        loss = net(im_1, im_2)
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100


if __name__ == '__main__':
    """### Define data loaders"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    # create model
    model = ModelMoCo(
            dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t,
            arch=args.arch,bn_splits=args.bn_splits, symmetric=args.symmetric).cuda()

    # prepare data
    train_data = CIFAR10Pair(
        root='./data', train=True, transform=train_transform, download=True)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True)
    memory_data = CIFAR10(
        root='data', train=True, transform=test_transform, download=True)
    memory_loader = DataLoader(
        memory_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = CIFAR10(
        root='data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    """### Start training"""
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    # load model if resume
    epoch_start = 1
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))

    # logging
    results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    writer = SummaryWriter(args.results_dir)
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    # training loop
    for epoch in range(epoch_start, args.epochs + 1):
        if epoch > 1 and args.data_ratio < 1:

            if args.sample_method == 'kcore':
                print('Sampling method is [kcore!]')
                select_dataset = CIFAR10(
                    root='data', train=True,
                    transform=test_transform, download=True)
                select_loader = DataLoader(
                    select_dataset, batch_size=20, shuffle=False,
                    num_workers=16, pin_memory=True, drop_last=True)
                index = select_kcore(model.encoder_q, select_loader, args.data_ratio)

                change_dataset = CIFAR10Pair(
                    root='./data', train=True, transform=train_transform, download=True)
                sampler = RandomSampler(change_dataset, index)
                train_loader = DataLoader(
                    change_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=16, pin_memory=True, drop_last=True, sampler=sampler)

            elif args.sample_method == 'high_loss':
                print('Sampling method is [high_loss!]')
                select_dataset = CIFAR10Pair(
                    root='data', train=True,
                    transform=test_transform, download=True)
                select_loader = DataLoader(
                    select_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=16, pin_memory=True, drop_last=True)
                index = sort_loss(model, select_loader, args.data_ratio, descending=True)

                sampler = RandomSampler(select_dataset, index)
                train_loader = DataLoader(
                    select_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=16, pin_memory=True, drop_last=True, sampler=sampler)

            elif args.sample_method == 'low_loss':
                print('Sampling method is [low_loss!]')
                select_dataset = CIFAR10Pair(
                    root='data', train=True,
                    transform=test_transform, download=True)
                select_loader = DataLoader(
                    select_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=16, pin_memory=True, drop_last=True)
                index = sort_loss(model, select_loader, args.data_ratio, descending=False)

                sampler = RandomSampler(select_dataset, index)
                train_loader = DataLoader(
                    select_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=16, pin_memory=True, drop_last=True, sampler=sampler)
            else:
                raise NotImplementedError

        train_loss = train(model, train_loader, optimizer, epoch, args)
        results['train_loss'].append(train_loss)
        writer.add_scalar('Train loss', train_loss,epoch)

        test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch, args)
        results['test_acc@1'].append(test_acc_1)
        writer.add_scalar('Test acc', test_acc_1, epoch)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
        # save model
        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            },
            args.results_dir + '/model_last.pth'
        )

