import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import progressbar

from tensorflow.keras.datasets import cifar10
from utils import _apply_transformations

tf.compat.v1.enable_eager_execution()

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_array_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecords(data_dir, filename, target_size):
    widgets = [progressbar.ETA(), " | ", progressbar.Percentage(), " ", progressbar.Bar()]
    bar = progressbar.ProgressBar(widgets=widgets,maxval=len(os.listdir(data_dir)))
    bar.start()
    with tf.io.TFRecordWriter(filename) as writer:
        for b, dir in enumerate(os.listdir(data_dir)):
            try:
                array = np.load(os.path.join(data_dir, dir))
                for i in range(array["colorImages"].shape[-1]):
                    if i == 20:
                        break
                    image, label = _apply_transformations(array, target_size, i)
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'image': _bytes_feature(image.tobytes()),
                            'label': _float_array_feature(label),
                        }))
                    writer.write(example.SerializeToString())
            except Exception as err:
                print(err)
            
            if b == 1:
                break
            bar.update(b)


def _parse_image_function(example_proto):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([208,172,3], tf.string),
        'label': tf.io.FixedLenFeature([68*2], tf.float32),
    }
    return tf.io.parse_single_example(example_proto, image_feature_description)


def read_dataset(epochs, batch_size, channel, channel_name):

    dataset = tf.data.TFRecordDataset(channel)
    dataset = dataset.map(_parse_image_function)#, num_parallel_calls=10)
    dataset = dataset.prefetch(10)
    dataset = dataset.repeat(epochs)
    dataset = dataset.shuffle(buffer_size=10 * batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

if __name__ == "__main__":
    data_dir = "/disk2/datasets/faces_dataset"
    filename = "/disk2/test/train.tfrecords"
    target_size = (208, 172)
    write_tfrecords(data_dir, filename, target_size)
    print("finished")