import os
import numpy as np
import cv2
import pickle
import random

import glob

import matplotlib.pyplot as plt

base_path = '/home/kswada/kw/segmentation/bdd/unet'


from keras.preprocessing import image
from tensorflow.keras.utils import load_img


import tensorflow as tf
print(tf.__version__)

# this should be required for GPU utilization
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)



#######################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------
# data augmentation
# ---------------------------------------------------------------------------------------------------------------------

# image data file (original)
img_path = '/media/kswada/MyFiles/dataset/BDD/bdd100k_images_100k/bdd100k/images/100k/train'
img_files = sorted(glob.glob(os.path.join(img_path, '*.jpg')))


# ----------
# parameters for ImageDataGenerator
params = {
    'rotation_range': 20,
    'width_shift_range': 0.4,
    'channel_shift_range': 0.2,
    'height_shift_range': 0.3,
    'horizontal_flip': True,
    'brightness_range': [0.3, 1.0]
}

datagen = image.ImageDataGenerator(**params)


# ----------
# select 1 image and convert

index = random.randint(0, len(img_files))

img = load_img(img_files[index])
img = np.array(img)

plt.imshow(img)
plt.show()


# ----------
# convert to [H, W, C] --> [1, H, W, C]
x = img[np.newaxis]
gen = datagen.flow(x, batch_size=1)

plt.figure(figsize=(10, 8))
for i in range(9):
    # [numBatches, H, W, C]
    batches = next(gen)
    gen_img = batches[0].astype(np.uint8)
    plt.subplot(3, 3, i + 1)
    plt.imshow(gen_img)
    plt.axis('off')

plt.show()


