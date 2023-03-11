sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

import os
import numpy as np
import cv2
import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)

base_path = 'C:\\Users\\kouse\\Desktop\\ImageProcessing\\ThinkAutonomous\\image_segmentation\\drivable_area_detection'


# ------------------------------------------------------------------------------------------
# functions:  inference and visualization
# ------------------------------------------------------------------------------------------

def rgb_channel(img, thresholding=False, thresh=230):
    """Threshold the re-drawn images"""
    image = np.copy(img)
    if thresholding:
        ret, image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    return R, G, B


def run(model, input_image, thresholding=True, thresh=230):
    h, w, d = input_image.shape
    network_image = input_image.copy()
    network_image = cv2.resize(network_image, (160, 80), interpolation=cv2.INTER_AREA)
    network_image = network_image[None, :, :, :]

    prediction = model.predict(network_image)[0] * 255
    R, G, B = rgb_channel(prediction, thresholding=thresholding, thresh=thresh)

    blank = np.zeros_like(R).astype(np.uint8)
    lane_image = np.dstack((R, blank, B))
    lane_image = cv2.resize(lane_image, (w, h))
    result = cv2.addWeighted(input_image, 1, lane_image.astype(np.uint8), 1, 0)

    return result, lane_image


# ------------------------------------------------------------------------------------------
# load model
# ------------------------------------------------------------------------------------------

model1 = EncoderDecoderSkipConnections.build()
model1.load_weights(os.path.join(base_path, 'model\\model_EncoderDecoderSkipConnections_epoch50.h5'))

model2 = EncoderDecoderSkipConnections.build()
model2.load_weights(os.path.join(base_path, 'model\\model_EncoderDecoderSkipConnections_epoch50_2.h5'))


# ------------------------------------------------------------------------------------------
# inference and visualization
# ------------------------------------------------------------------------------------------

# index = random.randint(0, len(images))

index = 1791
print(images[index].shape)


# model1 is much better
img_test, lane_img_test = run(model=model1, input_image=images[index], thresholding=True, thresh=100)
# img_test, lane_img_test = run(model=model2, input_image=images[index], thresholding=True, thresh=100)


# ----------
f, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 10))
ax1[0].imshow(images[index].squeeze())
ax1[0].set_title('Image', fontsize=20)
ax1[1].imshow(labels[index].squeeze())
ax1[1].set_title('Label', fontsize=20)
ax2[0].imshow(lane_img_test)
ax2[0].set_title("Drawn Prediction", fontsize=20)
ax2[1].imshow(img_test)
ax2[1].set_title("Ovarlaid with Prediction", fontsize=20)


# ------------------------------------------------------------------------------------------
# inference and visualization, step by step
# ------------------------------------------------------------------------------------------

model = model1


# ----------
index = random.randint(0, len(images))

input_image = images[index]


# ----------
h, w, d = input_image.shape
network_image = input_image.copy()
network_image = cv2.resize(network_image, (160, 80), interpolation=cv2.INTER_AREA)
network_image = network_image[None, :, :, :]

prediction = model.predict(network_image)[0] * 255

# thresholding, thresh = 100 (low value)
ret, prediction2 = cv2.threshold(prediction, 100, 255, cv2.THRESH_BINARY)

R = prediction2[:, :, 0]
G = prediction2[:, :, 1]
B = prediction2[:, :, 2]

blank = np.zeros_like(R).astype(np.uint8)
lane_image = np.dstack((R, blank, B))
lane_image = cv2.resize(lane_image, (w, h))

result = cv2.addWeighted(input_image, 1, lane_image.astype(np.uint8), 1, 0)


# ----------
f, (ax1, ax2) = plt.subplots(2, 2, figsize=(16, 8))
ax1[0].imshow(input_image.squeeze())
ax1[0].set_title('Image', fontsize=20)
ax1[1].imshow(labels[index].squeeze())
ax1[1].set_title('Label', fontsize=20)
ax2[0].imshow(lane_image)
ax2[0].set_title("Drawn Prediction", fontsize=20)
ax2[1].imshow(result)
ax2[1].set_title("Ovarlaid with Prediction", fontsize=20)


# ------------------------------------------------------------------------------------------
# inference and visualization for NEVER SEEN IMAGE
# ------------------------------------------------------------------------------------------

model = model1


# ----------
# image data file (original)
img_path = 'C:\\Users\\kouse\\Desktop\\imageData\\BDD\\bdd100k_images_100k\\train'
img_files = glob.glob(os.path.join(img_path, '*.jpg'))


# ----------
index = random.randint(0, len(img_files))

input_image = mpimg.imread(img_files[index])
print(input_image.shape)

# ----------
img_test, lane_img_test = run(model=model, input_image=input_image, thresholding=True, thresh=100)

# ----------
f, (ax1, ax2) = plt.subplots(2, 2, figsize=(16, 8))
ax1[0].imshow(input_image.squeeze())
ax1[0].set_title('Image', fontsize=20)
ax2[0].imshow(lane_img_test)
ax2[0].set_title("Drawn Prediction", fontsize=20)
ax2[1].imshow(img_test)
ax2[1].set_title("Ovarlaid with Prediction", fontsize=20)

