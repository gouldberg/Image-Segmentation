
import os
import time

# import math
import random

import torch
import torch.nn as nn

from mit_semseg.dataset import TrainDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback


# ----------
# REFERENCE
# https://github.com/CSAILVision/semantic-segmentation-pytorch
# Semantic Segmentation on MIT ADE20K dataset in PyTorch


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# train one epoch
# ----------------------------------------------------------------------------------------------------------------------

def train(segmentation_module, iterator, optimizers, history, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # ----------
        # load a batch of data
        # batch_data = next(iterator)
        batch_data = next(iterator)[0]
        for k in batch_data.keys():
            batch_data[k] = batch_data[k].cuda()
        # ----------
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # forward pass
        loss, acc = segmentation_module(batch_data)
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())


def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))


# ----------------------------------------------------------------------------------------------------------------------
# group_weight
# create_optimizers
# adjust_learning_rate
# ----------------------------------------------------------------------------------------------------------------------

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets

    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)

    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)

    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------------------------------------------

from mit_semseg.config import cfg

# default config
print(cfg)


# ----------
# update config
# cfg.merge_from_file('./config/ade20k-resnet50dilated-ppm_deepsup.yaml')
# cfg.merge_from_file('./config/ade20k-resnet50-upernet.yaml')
# cfg.merge_from_file('./config/ade20k-mobilenetv2dilated-c1_deepsup.yaml')
cfg.merge_from_file('./config/ade20k-hrnetv2.yaml')

print(cfg)
# cfg.freeze()

# cfg.merge_from_file(args.cfg)
# cfg.merge_from_list(args.opts)
# cfg.freeze()


# ----------
# data root
cfg.DATASET.root_dataset = '/media/kswada/MyFiles/dataset/'


# ----------
if not os.path.isdir(cfg.DIR):
    os.makedirs(cfg.DIR)

with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
    f.write("{}".format(cfg))


# ----------------------------------------------------------------------------------------------------------------------
# Start from checkpoint
# ----------------------------------------------------------------------------------------------------------------------

if cfg.TRAIN.start_epoch > 0:
    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"


# ----------------------------------------------------------------------------------------------------------------------
# parse gpu ids
# ----------------------------------------------------------------------------------------------------------------------

gpu_id = '0'
gpus = parse_devices(gpu_id)

gpus = [x.replace('gpu', '') for x in gpus]
gpus = [int(x) for x in gpus]

num_gpus = len(gpus)

print(gpus)
print(num_gpus)


# ----------------------------------------------------------------------------------------------------------------------
# other settings
# ----------------------------------------------------------------------------------------------------------------------

cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch

cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder

cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder


# ----------
print(cfg.TRAIN.batch_size)
print(cfg.TRAIN.max_iters)
print(cfg.TRAIN.running_lr_decoder)
print(cfg.TRAIN.running_lr_decoder)


# ----------
# seed
random.seed(cfg.TRAIN.seed)

torch.manual_seed(cfg.TRAIN.seed)


# ----------------------------------------------------------------------------------------------------------------------
# model building
# ----------------------------------------------------------------------------------------------------------------------

net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_encoder)


net_decoder = ModelBuilder.build_decoder(
    arch=cfg.MODEL.arch_decoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=cfg.MODEL.weights_decoder)


crit = nn.NLLLoss(ignore_index=-1)

if cfg.MODEL.arch_decoder.endswith('deepsup'):
    segmentation_module = SegmentationModule(
        net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
else:
    segmentation_module = SegmentationModule(
        net_encoder, net_decoder, crit)


# ----------------------------------------------------------------------------------------------------------------------
# dataset and loader
# ----------------------------------------------------------------------------------------------------------------------

dataset_train = TrainDataset(
    cfg.DATASET.root_dataset,
    cfg.DATASET.list_train,
    cfg.DATASET,
    batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)


loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=len(gpus),  # we have modified data_parallel
    shuffle=False,  # we do not use this param
    collate_fn=user_scattered_collate,
    num_workers=cfg.TRAIN.workers,
    drop_last=True,
    pin_memory=True)

print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))


# create loader iterator
iterator_train = iter(loader_train)


# ----------
# batch_data = next(iterator_train)
# print(batch_data[0].keys())
# # (2, 3, 456, 456)
# print(batch_data[0]['img_data'].shape)
# # (2, 57, 80)
# print(batch_data[0]['seg_label'].shape)


# ----------------------------------------------------------------------------------------------------------------------
# load nets into gpu
# ----------------------------------------------------------------------------------------------------------------------

if len(gpus) > 1:
    segmentation_module = UserScatteredDataParallel(
        segmentation_module,
        device_ids=gpus)
    # For sync bn
    patch_replication_callback(segmentation_module)

segmentation_module.cuda()


# ----------------------------------------------------------------------------------------------------------------------
# Set up optimizers
# ----------------------------------------------------------------------------------------------------------------------

nets = (net_encoder, net_decoder, crit)

optimizers = create_optimizers(nets, cfg)


# ----------------------------------------------------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------------------------------------------------

history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
    print(f'epoch: {epoch}')
    train(segmentation_module, iterator_train, optimizers, history, epoch + 1, cfg)

    # checkpointing
    checkpoint(nets, history, cfg, epoch + 1)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# check:  group_weight
# ----------------------------------------------------------------------------------------------------------------------

net_encoder = ModelBuilder.build_encoder(
    arch='mobilenetv2dilated',
    fc_dim=320,
    weights='')


net_decoder = ModelBuilder.build_decoder(
    arch='c1_deepsup',
    fc_dim=320,
    num_class=150,
    weights='')


module = net_encoder
# module = net_decoder

group_decay = []
group_no_decay = []

for m in module.modules():
    if isinstance(m, nn.Linear):
        group_decay.append(m.weight)
        if m.bias is not None:
            group_no_decay.append(m.bias)
    elif isinstance(m, nn.modules.conv._ConvNd):
        group_decay.append(m.weight)
        if m.bias is not None:
            group_no_decay.append(m.bias)
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        if m.weight is not None:
            group_no_decay.append(m.weight)
        if m.bias is not None:
            group_no_decay.append(m.bias)


print(len(list(module.parameters())))
assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)

groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]


# ----------------------------------------------------------------------------------------------------------------------
# check:  running learning rate
# ----------------------------------------------------------------------------------------------------------------------

epochs = 20
iters_per_epoch = 5000
lr_pow = 0.9

for cur_iter in range(1, epochs * iters_per_epoch + 1):
    scale_running_lr = ((1. - float(cur_iter) / (epochs * iters_per_epoch)) ** lr_pow)
    print(f'{cur_iter}:  running lr: {scale_running_lr}')
