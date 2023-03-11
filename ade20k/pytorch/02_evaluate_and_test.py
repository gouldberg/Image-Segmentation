
import os
# import argparse
# from distutils.version import LooseVersion

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv

from mit_semseg.dataset import TestDataset
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive
from mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
import time


# ----------------------------------------------------------------------------------------------------------------------
# visualize result
# ----------------------------------------------------------------------------------------------------------------------

def visualize_result_eval(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


# ----------
colors = loadmat('./data/color150.mat')['colors']
names = {}

with open('./data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def visualize_result_test(data, pred, cfg):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))


# ----------------------------------------------------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------------------------------------------------

def evaluate(segmentation_module, loader, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # ----------
        # process data
        batch_data = batch_data[0]
        # ----------
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # visualization
        if cfg.VAL.visualize:
            visualize_result_eval(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'results_eval')
            )

        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))


# ----------------------------------------------------------------------------------------------------------------------
# test
# ----------------------------------------------------------------------------------------------------------------------

def test(segmentation_module, loader, gpu):
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # ----------
        # process data
        batch_data = batch_data[0]
        # ----------
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        visualize_result_test(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg
        )

        pbar.update(1)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# Evaluate a trained model on the validation set
#  - validation set is ./data/validation.odgt (image path and width,height) --> total 2000 images
# ----------------------------------------------------------------------------------------------------------------------

# del cfg

base_path = '/home/kswada/kw/segmentation/ade20k/pytorch'

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
cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)

assert os.path.exists(cfg.MODEL.weights_encoder) and \
       os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

print(os.path.join(cfg.DIR, "results_eval"))

if not os.path.isdir(os.path.join(cfg.DIR, "results_eval")):
    os.makedirs(os.path.join(cfg.DIR, "results_eval"))


# ----------
gpu = 0
torch.cuda.set_device(gpu)


# ----------
# Network Builders
net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_encoder)

net_decoder = ModelBuilder.build_decoder(
    arch=cfg.MODEL.arch_decoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=cfg.MODEL.weights_decoder,
    use_softmax=True)

crit = nn.NLLLoss(ignore_index=-1)

segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)


# ----------
# Dataset and Loader
print(cfg.DATASET.list_val)

dataset_val = ValDataset(
    cfg.DATASET.root_dataset,
    cfg.DATASET.list_val,
    cfg.DATASET)

loader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=cfg.VAL.batch_size,
    shuffle=False,
    collate_fn=user_scattered_collate,
    num_workers=5,
    drop_last=True)


# ----------
segmentation_module.cuda()

cfg.VAL.visualize = True

evaluate(segmentation_module, loader_val, cfg, gpu)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# test
# ----------------------------------------------------------------------------------------------------------------------

del cfg

base_path = '/home/kswada/kw/segmentation/ade20k/pytorch'

from mit_semseg.config import cfg

# default config
print(cfg)


# ----------
# update config
# cfg.merge_from_file('./config/ade20k-resnet50dilated-ppm_deepsup.yaml')
# cfg.merge_from_file('./config/ade20k-resnet50-upernet.yaml')
cfg.merge_from_file('./config/ade20k-mobilenetv2dilated-c1_deepsup.yaml')
print(cfg)
# cfg.freeze()

# cfg.merge_from_file(args.cfg)
# cfg.merge_from_list(args.opts)
# cfg.freeze()


# ----------
# data root
cfg.DATASET.root_dataset = '/media/kswada/MyFiles/dataset/'


# ----------
cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()


# ----------
# absolute paths of model weights

cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

assert os.path.exists(cfg.MODEL.weights_encoder) and \
       os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"


# ----------
# generate testing image list
dat_path = '/media/kswada/MyFiles/dataset/ADEChallengeData2016'
imgs = find_recursive(dat_path)

cfg.list_test = [{'fpath_img': x} for x in imgs]

# ----------
cfg.TEST.result = os.path.join(base_path, 'results')

if not os.path.exists(cfg.TEST.result):
    os.makedirs(cfg.TEST.result)


# ----------
# network builders

gpu = 0
torch.cuda.set_device(gpu)

net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder,
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_encoder)


net_decoder = ModelBuilder.build_decoder(
    arch=cfg.MODEL.arch_decoder,
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=cfg.MODEL.weights_decoder,
    use_softmax=True)


crit = nn.NLLLoss(ignore_index=-1)

segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)


# ----------
# dataset and loader

dataset_test = TestDataset(
    cfg.list_test,
    cfg.DATASET)

loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=cfg.TEST.batch_size,
    shuffle=False,
    collate_fn=user_scattered_collate,
    num_workers=5,
    drop_last=True)


# -----------
# test

segmentation_module.cuda()

test(segmentation_module, loader_test, gpu)
