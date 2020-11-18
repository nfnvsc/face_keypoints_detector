import pickle
import matplotlib
import tensorflow as tf
import config
import numpy as np
from utils.datagenerator import DataGenerator
from model import make_or_restore_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

matplotlib.use('TKAgg') #docker

callbacks = [
    # This callback saves a SavedModel every 1000 batches.
    # We include the training loss in the folder name.
    ModelCheckpoint(
        filepath=config.CHECKPOINT_DIR + '/ckpt-loss={loss:.6f}',
        save_freq=10000)
    #TensorBoard(log_dir=config.LOG_DIR)
]

#Model parameters
epochs = 50
validation_split = 0.8
params = {'dim': config.TARGET_SIZE,
        'batch_size': 4,
        'n_classes': config.TARGET_OUTPUT,
        'n_channels': 3,
        'shuffle': False,
        'file_extension': '.npz'}

with open(config.DATASET_DIR_PICKLE, "rb") as in_file:
    x_train = pickle.load(in_file)
with open(config.LABELS_DIR_PICKLE, "rb") as in_file:
    y_train = pickle.load(in_file)

#with open(config.LABELS_DIR, "rb") as in_file:
#    labels = pickle.load(in_file)
#
#train_ids = labels[:int(len(labels)*0.8)]
#validation_ids = labels[int(len(labels)*0.8):]

training_generator = DataGenerator(data_set_x=x_train, data_set_y=y_train, **params, data_dir=config.DATASET_DIR)

#training_generator = DataGenerator(train_ids, **params, data_dir=config.DATASET_DIR)
#validation_generator = DataGenerator(validation_ids, **params, data_dir=config.DATASET_DIR)

model = make_or_restore_model(config.CHECKPOINT_DIR)
"""
model.fit(x_train, y_train,
            epochs=epochs,
            callbacks=callbacks)
"""
model.fit(training_generator,
            #validation_data=validation_generator,
            use_multiprocessing=True,
            workers=4,
            max_queue_size=30,
            epochs=epochs,
            callbacks=callbacks)