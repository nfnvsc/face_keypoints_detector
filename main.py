import pickle
import matplotlib
import tensorflow as tf
from datagenerator import DataGenerator
from model import make_or_restore_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tfrecords import read_dataset

matplotlib.use('TKAgg') #docker
#tf.compat.v1.disable_eager_execution()
#Directories
DATASET_DIR = "/datasets/"
LABELS_DIR = "/datasets/labels.pickle"
CHECKPOINT_DIR = "/ckpt"
LOG_DIR = "/logs"

callbacks = [
    # This callback saves a SavedModel every 1000 batches.
    # We include the training loss in the folder name.
    ModelCheckpoint(
        filepath=CHECKPOINT_DIR + '/ckpt-loss={loss:.6f}',
        save_freq=10000)
    #TensorBoard(log_dir=LOG_DIR)
]


#Data Generator parameters
params = {'dim': (208, 172),
        'batch_size': 8,
        'n_classes': 68*2,
        'n_channels': 3,
        'shuffle': False,
        'file_extension': '.npz'}

#General Parameters
validation_split = 0.8
EPOCHS = 50

with open(LABELS_DIR, "rb") as in_file:
    labels = pickle.load(in_file)

train_ids = labels[:int(len(labels)*0.8)]
validation_ids = labels[int(len(labels)*0.8):]

training_generator = DataGenerator(train_ids, **params, data_dir=DATASET_DIR)
validation_generator = DataGenerator(validation_ids, **params, data_dir=DATASET_DIR)

model = make_or_restore_model(CHECKPOINT_DIR)
model.fit(training_generator,
            #validation_data=validation_generator,
            use_multiprocessing=True,
            workers=8,
            max_queue_size=30,
            epochs=EPOCHS,
            callbacks=callbacks)