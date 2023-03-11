
import os
import numpy as np
import cv2
import pickle
import random

import glob

import matplotlib.pyplot as plt


base_path = '/home/kswada/kw/segmentation/bdd/unet'

# Think Autonomous and Here is useful trial for drive lane detection !!!
# https://towardsdatascience.com/lane-detection-with-deep-learning-part-1-9e096f3320b7


#######################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------
# load images and labels: from original data
# ---------------------------------------------------------------------------------------------------------------------

img_path = '/media/kswada/MyFiles/dataset/BDD/bdd100k_images_100k/bdd100k/images/100k/train'
label_path = '/media/kswada/MyFiles/dataset/BDD/bdd100k_drivable_labels_trainval/bdd100k/labels/drivable/colormaps/train'


# ----------
img_files = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
label_files = sorted(glob.glob(os.path.join(label_path, '*.png')))

print(img_files[0])
print(label_files[0])


# 70000 images
print(len(img_files))

# 70000 images
print(len(label_files))


# ----------
for _ in range(10):
    index = random.randint(0, len(label_files))

    images = cv2.imread(img_files[index])
    labels = cv2.imread(label_files[index])

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(images)
    ax1.set_title('Image', fontsize=30)
    ax2.imshow(labels)
    ax2.set_title('Label', fontsize=30)
    plt.show()


# original image has high resolution
# (720, 1280, 3)
print(images.shape)
print(labels.shape)


#######################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------
# load images and labels
#  - selected 3000 images and their associated labels from BDD 100K dataset
#  - each image has been resized to 160 * 80 for memory reasons
# ---------------------------------------------------------------------------------------------------------------------

with open(os.path.join(base_path, './data/images_3000_160.p'), 'rb') as f:
    images = pickle.load(f)

with open(os.path.join(base_path, './data/labels_3000_160.p'), 'rb') as f:
    labels = pickle.load(f)


# ----------
print(len(images))
# (80, 160, 3)
print(images[0].shape)
print(images[0])

print(len(labels))
# (80, 160, 3)
print(labels[0].shape)
print(labels[0])


# As we are only working with 2 classes, and they all are colored either red or blue,
# it's quite easy to work with.
# If you ever get more classes, let's say 7, some pixels will not be 0 or 255,
# they will have some intensity value and it will get more complicated.
# Preprocessing is therefore necessary.


# ---------------------------------------------------------------------------------------------------------------------
# show images and labels, selecting randomly
# ---------------------------------------------------------------------------------------------------------------------

for _ in range(10):

    index = random.randint(0, len(labels))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(images[index].squeeze())
    ax1.set_title('Image', fontsize=30)
    ax2.imshow(labels[index].squeeze())
    ax2.set_title('Label', fontsize=30)
    plt.show()


# ----------
# the values is: Blue, Green, Red
print(labels[index][60,70,:])


# ---------------------------------------------------------------------------------------------------------------------
# convert background to green color
#  --> this ensures one hot encoding for 3 classes
#      (green: background, red and blue: drivable lanes)
# ---------------------------------------------------------------------------------------------------------------------

# IT TAKES TIME !!!
# new_labels = []
#
# for lab in labels:
#     for x in range(lab.shape[0]):
#         for y in range(lab.shape[1]):
#             if (np.all(lab[x][y] == [0,0,0])):
#                 lab[x][y] = [0,1,0]
#     new_labels.append(lab)
#
# plt.imshow(new_labels[0])
#
#
# # ----------
# with open(os.path.join(base_path, 'data\\labels_3000_160_background_green.p'), 'wb') as f:
#     pickle.dump(new_labels, f)


# ----------
with open(os.path.join(base_path, './data/labels_3000_160_background_green.p'), 'rb') as f:
    new_labels = pickle.load(f)


plt.imshow(new_labels[0])

