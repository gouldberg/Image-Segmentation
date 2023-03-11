sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

import os
import numpy as np
import cv2

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Input, add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import regularizers

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

base_path = 'C:\\Users\\kouse\\Desktop\\ImageProcessing\\ThinkAutonomous\\image_segmentation\\drivable_area_detection'


# -----------------------------------------------------------------------------
# prepare train and test data
# -----------------------------------------------------------------------------

images = np.array(images)
labels = np.array(new_labels)


# ----------
# Shuffle
images, labels = shuffle(images, labels)


# ----------
# Test size may go from 10% to 30%
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.15)
n_train = len(X_train)
n_val = len(X_val)

print("Number of element in training set")
print(n_train)
print("Number of element in validation set")
print(n_val)


# -----------------------------------------------------------------------------
# prepare train and test data
# -----------------------------------------------------------------------------

# Using a generator to help the model use less data
# chennel_shift_range:  Channel shifts help with shadows slightly
params = {
    # 'rotation_range': 20,
    # 'width_shift_range': 0.4,
    'channel_shift_range': 0.2,
    # 'height_shift_range': 0.3,
    # 'horizontal_flip': True,
    # 'brightness_range': [0.3, 1.0]
}

datagen = image.ImageDataGenerator(**params)

datagen.fit(X_train)


# ---------
# model
input_shape = X_train.shape[1:]
pool_size = (2, 2)
dropout_rate = 0.5

model = EncoderDecoderSkipConnections.build(
    input_shape=input_shape, pool_size=pool_size, dropout_rate=dropout_rate)

model.summary()


# ----------
# optimizer, loss function, and model compile
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate, amsgrad=False, name="Adam")

# Compiling and training the model
# loss function is categorical_crossentropy: here 3 classes (background, drivable area, adjacent area)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy')


# ----------
# training
batch_size = 32
# batch_size = 64
epochs = 50
steps_per_epoch = len(X_train) / batch_size

history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs, verbose=1, validation_data=(X_val, y_val))

# ----------
print(history.history.keys())


# summarize history for loss
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

fig.savefig(os.path.join(base_path, 'model\\hist_model_EncoderDecoderSkipConnections_epoch50_bilinear_dataug2.png'))


# ----------
# Save model architecture and weights
model.save(os.path.join(base_path, 'model\\model_EncoderDecoderSkipConnections_epoch50_bilinear_dataug2.h5'))




