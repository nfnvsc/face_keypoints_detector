import sys
sys.path.append("..")
from config import config

import os

from tensorflow import keras

from models.alexnet_model import alexnet
from models.inception_model import inception


def make_model():
    if config.MODEL == "inception":
        return inception()
    elif config.MODEL == "alexnet":
        return alexnet()

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


