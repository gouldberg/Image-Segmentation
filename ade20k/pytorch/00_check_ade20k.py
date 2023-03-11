
import os
import glob

import numpy as np
import pandas as pd
import cv2

import random
from collections import Counter

import matplotlib.pyplot as plt


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# image and annotation
# ----------------------------------------------------------------------------------------------------------------------

dat_path = '/media/kswada/MyFiles/dataset/ADEChallengeData2016'

# image is jpg
img_files_train = sorted(glob.glob(os.path.join(dat_path, 'images', 'training', '*.jpg')))
img_files_val = sorted(glob.glob(os.path.join(dat_path, 'images', 'validation', '*.jpg')))

# annotation is png
anot_files_train = sorted(glob.glob(os.path.join(dat_path, 'annotations', 'training', '*.png')))
anot_files_val = sorted(glob.glob(os.path.join(dat_path, 'annotations', 'validation', '*.png')))

# 20210
print(len(img_files_train))
# 2000
print(len(img_files_val))
# 20210
print(len(anot_files_train))
# 2000
print(len(anot_files_val))


# ----------
print(img_files_train[0])
print(anot_files_train[0])


# ----------
dat_cat = dict(np.loadtxt(os.path.join(dat_path, 'sceneCategories.txt'), dtype="unicode"))

objinfo = dict(pd.read_csv(os.path.join(dat_path, 'objectInfo150.txt'), sep='\t')[['Idx', 'Name']].to_numpy())
objinfo[0] = 'background'

# 150 classes + background
print(objinfo)


# ----------
for _ in range(10):
    index = random.randint(0, len(img_files_train))
    fname = img_files_train[index].split('/')[-1].split('.')[0]
    img = cv2.imread(img_files_train[index])
    annot = cv2.imread(anot_files_train[index])
    cat = dat_cat[fname]

    obj_idx = list(Counter(annot.flatten()).keys())
    obj = [objinfo[idx] for idx in obj_idx]
    print(f'{fname} - {cat}')
    print(f'{img.shape} - {annot.shape}')
    print(f'{obj}')

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title(f'Image: {cat}', fontsize=15)
    ax2.imshow(annot)
    ax2.set_title(f'annotation', fontsize=15)
    plt.show()
