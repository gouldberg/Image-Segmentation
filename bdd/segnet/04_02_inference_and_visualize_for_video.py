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


from moviepy.editor import VideoFileClip


# ------------------------------------------------------------------------------------------
# functions:  inference and visualization
#   - THIS IS FOR VIDEO videoFileClip
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


# in order to use VideoFileClip, the argument is only frame (= input_image)

def run(input_image):
    h, w, d = input_image.shape
    network_image = input_image.copy()
    network_image = cv2.resize(network_image, (160, 80), interpolation=cv2.INTER_AREA)
    network_image = network_image[None, :, :, :]

    prediction = model.predict(network_image)[0] * 255

    # Thresholding = True and thresh = 125 (or lower)
    R, G, B = rgb_channel(prediction, thresholding=True, thresh=100)

    blank = np.zeros_like(R).astype(np.uint8)
    lane_image = np.dstack((R, blank, B))
    lane_image = cv2.resize(lane_image, (w, h))
    result = cv2.addWeighted(input_image, 1, lane_image.astype(np.uint8), 1, 0)

    return result


# ------------------------------------------------------------------------------------------
# load model
# ------------------------------------------------------------------------------------------

model = EncoderDecoderSkipConnections.build()


# model.load_weights(os.path.join(base_path, 'model\\model_EncoderDecoderSkipConnections_epoch50.h5'))
model.load_weights(os.path.join(base_path, 'model\\model_EncoderDecoderSkipConnections_epoch50_bilinear.h5'))
# model.load_weights(os.path.join(base_path, 'model\\model_EncoderDecoderSkipConnections_epoch50_bilinear_dataug2.h5'))



# ------------------------------------------------------------------------------------------
# inference on video
#  - FCN models generally are fast. They can run on CPU at 5 FPS
# ------------------------------------------------------------------------------------------

# video_file = os.path.join(base_path, 'videos\\project.avi')
# video_file = os.path.join(base_path, 'videos\\project_2.avi')
# video_file = os.path.join(base_path, 'videos\\project_video.mp4')
video_file = os.path.join(base_path, 'videos\\paris_challenge.mov')
# video_file = os.path.join(base_path, 'videos\\costa_rica_challenge.mp4')

clip = VideoFileClip(video_file)


# ----------
white_clip = clip.fl_image(run)


# ----------
%time white_clip.write_videofile(os.path.join(base_path, 'output', os.path.basename(os.path.splitext(video_file)[0]) + '_test_bilinear.mp4'),audio=False)

