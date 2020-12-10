import tensorflow as tf
from config import config
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD, Adam


def alexnet():
    """
        AlexNet adapted network
    """
    model = Sequential()
    model.add(Input(shape=(*config.TARGET_SIZE, 3)))
    model.add(Conv2D(32, (3,3), (1,1), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(64, (1,1), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(128, (3,3), padding="same", activation="relu")) #1.9e-04 before this layer
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(config.TARGET_OUTPUT, activation='sigmoid'))

    #opt = SGD(lr=0.01,momentum = 0.9,nesterov=True)
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)
    print(model.summary())
    return model
