
import os
import numpy as np
import cv2

import tensorflow as tf
print(tf.__version__)

# this should be required for GPU utilization
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Input, add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

base_path = 'C:\\Users\\kouse\\Desktop\\ImageProcessing\\ThinkAutonomous\\image_segmentation\\drivable_area_detection'


# -----------------------------------------------------------------------------
# model: EncoderDecoderSkipConnections   based on SegNet
#   - Credits for the architecture: https://towardsdatascience.com/lane-detection-with-deep-learning-part-2-3ba559b5c5af
# -----------------------------------------------------------------------------

class EncoderDecoderSkipConnections:

    @staticmethod
    def build(input_shape=(160, 80, 3), pool_size=(2, 2), dropout_rate=0.5):

        # --------------------
        # ENCODER
        input_x = Input(shape=(80, 160, 3))
        x1 = BatchNormalization(input_shape=input_shape)(input_x)

        # CONV 1
        x = Conv2D(8, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x1)
        # CONV 2 + SKIP CONNECTION
        x = Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        # CONV 3
        x = Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        # CONV4 + SKIP CONNECTION
        x2 = Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x2)
        # CONV5
        x = Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        # CONV6
        x = Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        # CONV7
        x = Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling2D(pool_size=pool_size)(x)

        # --------------------
        # DECODER
        # UPSAMPLING 7
        # bilinear is better than nearest
        # x = UpSampling2D(size=pool_size, interpolation='nearest')(x)
        x = UpSampling2D(size=pool_size, interpolation='bilinear')(x)
        x = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        # UPSAMPLING 6
        x = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        # UPSAMPLING 5
        # x = UpSampling2D(size=pool_size, interpolation='nearest')(x)
        x = UpSampling2D(size=pool_size, interpolation='bilinear')(x)
        x = Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x)

        # UPSAMPLING 4  (add x2)
        x = add([x2, x])
        x = Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = Dropout(dropout_rate)(x)

        # UPSAMPLING 3
        x = Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        # UPSAMPLING 2
        # x = UpSampling2D(size=pool_size, interpolation='nearest')(x)
        x = UpSampling2D(size=pool_size, interpolation='bilinear')(x)
        x = Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)

        # UPSAMPLING 1  (add x1)
        x = Conv2DTranspose(3, (3, 3), padding='valid', strides=(1, 1), activation='relu')(x)
        x = add([x1, x])
        x = Conv2D(3, (1, 1), padding='valid', strides=(1, 1), activation='softmax')(x)

        return Model(input_x, x)


# ----------
model = EncoderDecoderSkipConnections.build()

model.summary()


# -----------------------------------------------------------------------------
# reference model :  based on SegNet
#  - https://github.com/mvirgo/MLND-Capstone/blob/master/fully_conv_NN.py
# -----------------------------------------------------------------------------

# (i.e. contains zero fully connected layers) neural network for detecting lanes.
# This version assumes the inputs to be road images in the shape of 80 x 160 x 3 (RGB)
# with the labels as 80 x 160 x 1 (just the G channel with a re-drawn lane).
# Note that in order to view a returned image, the predictions is
# later stacked with zero'ed R and B layers and added back to the initial road image.

from keras.models import Sequential

def create_model(input_shape, pool_size):

    model = Sequential()

    # Normalizes incoming inputs. First layer needs the input shape to work
    model.add(BatchNormalization(input_shape=input_shape))

    # Below layers were re-named for easier reading of model summary; this not necessary
    # Conv Layer 1
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv1'))

    # Conv Layer 2
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv2'))

    # Pooling 1
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 3
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv3'))
    model.add(Dropout(0.2))

    # Conv Layer 4
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv4'))
    model.add(Dropout(0.2))

    # Conv Layer 5
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv5'))
    model.add(Dropout(0.2))

    # Pooling 2
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 6
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv6'))
    model.add(Dropout(0.2))

    # Conv Layer 7
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv7'))
    model.add(Dropout(0.2))

    # Pooling 3
    model.add(MaxPooling2D(pool_size=pool_size))

    # Upsample 1
    model.add(UpSampling2D(size=pool_size))

    # Deconv 1
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv1'))
    model.add(Dropout(0.2))

    # Deconv 2
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv2'))
    model.add(Dropout(0.2))

    # Upsample 2
    model.add(UpSampling2D(size=pool_size))

    # Deconv 3
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv3'))
    model.add(Dropout(0.2))

    # Deconv 4
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv4'))
    model.add(Dropout(0.2))

    # Deconv 5
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv5'))
    model.add(Dropout(0.2))

    # Upsample 3
    model.add(UpSampling2D(size=pool_size))

    # Deconv 6
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv6'))

    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Final'))

    return model
