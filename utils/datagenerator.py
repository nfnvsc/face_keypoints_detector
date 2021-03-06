"""
Adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

import os
import time
import numpy as np
import tensorflow.keras as keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10,shuffle=True, file_extension=".jpg", data_dir="data/", data_set_x=None, data_set_y=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.file_extension = file_extension
        self.data_dir = data_dir
        self.data_set_x = data_set_x
        self.data_set_y = data_set_y
        if self.data_set_x and self.data_set_y:
            self.list_IDs = [i for i in range(len(data_set_x))]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        if self.data_set_x and self.data_set_y:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty((self.batch_size, self.n_classes))
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = self.data_set_x[ID]/255.0
                y[i,] = self.data_set_y[ID]
        else:
            X, y = self.__data_generation(list_IDs_temp)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            name, idx = ID.split(":")

            dir = os.path.join(self.data_dir, name)

            tmp = np.load(dir, allow_pickle=True)["arr_" + idx]

            X[i,], y[i] = tmp

        return X, y
