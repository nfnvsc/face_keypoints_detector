import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD, Adam


def make_model():
    """
        AlexNet adapted network
    """
    model = Sequential()
    model.add(Conv2D(48, (7,7), (4,4), activation="relu", input_shape=(208, 172, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3), strides=(2,2)))
    model.add(Conv2D(128, (5,5), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3), strides=(2,2)))
    model.add(Conv2D(192, (3,3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (3,3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(68*2, activation='sigmoid'))

    sgd = SGD(lr=0.01,momentum = 0.9,nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    print(model.summary())
    return model

def make_or_restore_model(checkpoint_dir):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)

    print('Creating a new model')
    return make_model()