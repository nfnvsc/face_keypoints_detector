import pickle
import matplotlib
import tensorflow as tf
import numpy as np
import math

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from utils.datagenerator import DataGenerator
from models.model import make_or_restore_model
from config import config

#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu_devices[0], True)
matplotlib.use('TKAgg') #docker

def decay(epoch, steps=100):
    initial_lrate = 0.001
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

callbacks = [
    # This callback saves a SavedModel every 1000 batches.
    # We include the training loss in the folder name.
    ModelCheckpoint(
        filepath=config.CHECKPOINT_DIR + '/ckpt-loss={loss:.6f}',
        save_freq=10000),
    TensorBoard(log_dir=config.CHECKPOINT_DIR)
    #LearningRateScheduler(decay, verbose=True)
]

#Model parameters
epochs = 150
batch_size = 16
validation_split = 0.8

params = {'dim': config.TARGET_SIZE,
        'batch_size': batch_size,
        'n_classes': config.TARGET_OUTPUT,
        'n_channels': 3,
        'shuffle': True,
        'file_extension': '.npz'}

with open(config.DATASET_DIR_PICKLE, "rb") as in_file:
    x_data = pickle.load(in_file)
with open(config.LABELS_DIR_PICKLE, "rb") as in_file:
    y_data = pickle.load(in_file)

print(np.array(x_data).shape)
print(np.array(y_data).shape)

x_train = np.array(x_data[:int(len(x_data)*0.8)])
y_train = np.array(y_data[:int(len(y_data)*0.8)])

x_validation = np.array(x_data[int(len(x_data)*0.8):])
y_validation = np.array(y_data[int(len(y_data)*0.8):])


#train_generator = DataGenerator(data_set_x=x_train, data_set_y=y_train, **params, data_dir=config.DATASET_DIR)
#validation_generator = DataGenerator(data_set_x=x_validation, data_set_y=y_validation, **params, data_dir=config.DATASET_DIR)

model = make_or_restore_model(config.CHECKPOINT_DIR)

model.fit(x_train, 
            y_train,
            validation_data=(x_validation, y_validation),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks)

"""
model.fit(train_generator,
            validation_data=validation_generator,
            #use_multiprocessing=True,
            workers=2,
            max_queue_size=30,
            epochs=epochs,
            callbacks=callbacks)
"""

"""
model.fit(x_train, y_train, 
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=len(x_train)/batch_size,
        shuffle=True,
        callbacks=callbacks)
"""