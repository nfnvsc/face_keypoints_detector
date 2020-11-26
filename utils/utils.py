import sys
sys.path.append("..")
from config import config

import os
import pickle
import queue
import numpy as np
import progressbar
import matplotlib.pyplot as plt
import face_recognition
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
from skimage.transform import resize
from multiprocessing import Process, cpu_count, Queue, Pool

#data_dir = "/disk2/datasets/faces_dataset/"
labels_dir = "/disk2/datasets/faces_dataset/labels.pickle"

SPLITTER = ":"
widgets = [progressbar.ETA(), " | ", progressbar.Percentage(), " ", progressbar.Bar()]

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([
    #iaa.Affine(rotate=(45,45), fit_output=True),
    #iaa.Crop(px=(1, 16), keep_size=False),
    sometimes(iaa.Clouds()),
    ##iaa.WithChannels((0,1,2), iaa.MultiplyHue((0.2,1.3))),
    sometimes(iaa.WithChannels((0,1,2), iaa.imgcorruptlike.GaussianNoise(severity=(1,3)))),
    #iaa.WithChannels((0,1,2), iaa.MultiplyHue((0.5,1.5))),
    #iaa.WithChannels((0,1,2), iaa.MultiplyHue((0.5,1.5))),
    #iaa.MultiplyHue((0.5, 1.5)),
    #iaa.BlendAlphaVerticalLinearGradient(
    #    iaa.AveragePooling(11),
    #    start_at=(0.0, 1.0), end_at=(0.0, 1.0)),
    #iaa.SomeOf

    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
    #iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
    #iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7

    #sometimes(iaa.Sharpen(alpha=(0.5, 1.0), lightness=(0.75, 1.5))), # sharpen images
    #iaa.Rotate((0,5), fit_output=True),
    #iaa.Resize((0.15, 0.3)),
    #iaa.PerspectiveTransform(scale=(0.1, 0.1), fit_output=True)
])

def test_augmentation(data_dir, labels_dir, target_size):
    for i, dir in enumerate(os.listdir(labels_dir)):
        with open(os.path.join(labels_dir, dir), "r") as label_file:
            for line in label_file.readlines():
                print(line)
                tmp = line.split(" ")
                
                file_name = tmp[0] + ".chip.jpg"
                image = np.array(Image.open(os.path.join(data_dir, file_name)))
                img = seq(image=image)#, keypoints=pts)
                img = Image.fromarray(img)
                img.show()

                #features = np.array(tmp[1:-1], dtype=np.float16).reshape(-1, 2)
                #features = np.divide(features, img_shape).flatten()
                #features = np.maximum(features, 0)
                
                break

        break


def create_labels(data_dir, output_dir):
    labels = []
    bar = progressbar.ProgressBar(widgets=widgets,maxval=len(os.listdir(data_dir)))
    bar.start()
    for x, dir in enumerate(os.listdir(data_dir)):
        try:
            array = np.load(os.path.join(data_dir, dir), allow_pickle=True)
            for i, _ in enumerate(array):
                labels.append(dir + SPLITTER + str(i))
        except Exception as err:
            print(err)
            #print(dir)
        
        bar.update(x)

    with open(output_dir, "wb") as open_file:
        pickle.dump(labels, open_file)

def plot_image_points(img, points):
    #if max(img) <= 1:
    plt.imshow(img)
    s = img.shape
    plt.plot(points[:, 0]*s[0], points[:, 1]*s[1], "bo", markersize=1)
    #else:
    #    plt.imshow(img)
    #    plt.plot(points[:, 0], points[:, 1], "bo", markersize=1)
    #plt.savefig("/photo/test.png")
    plt.show()

def _resize_image(img, target_size):
    return resize(img, target_size)

def _apply_transformations(data, target_size, i):
    tmp = data["colorImages"][:, :, :, i]
    target_img = _resize_image(tmp, target_size)
    target_img = np.array(target_img, dtype=np.float16)
    #target_img /= 255.0
    
    s = tmp.shape[0:2]
    tmp = data["landmarks2D"][:, :, i]
    target_labels = np.divide(tmp, s).flatten()
    target_labels = np.array(target_labels, dtype=np.float16)

    return target_img, target_labels


def create_dataset_multiprocessing(dataset_dir, output_dir, target_size):
    target_queue = []
    for dir in os.listdir(dataset_dir):
        file_dir = os.path.join(dataset_dir, dir)
        target_queue.append((file_dir, output_dir, target_size))
    
    with Pool(processes=cpu_count()) as pool:
        pool.map(create_dataset, target_queue)

    

def create_dataset(args):
    file_dir = args[0]
    output_dir = args[1]
    target_size = args[2]

    tmp = np.load(file_dir)

    name = file_dir.split("/")[-1] 
    
    new_data = []
    for i in range(tmp["colorImages"].shape[-1]):
        sample = []
        img, labels = _apply_transformations(tmp, target_size, i)

        sample.append(np.array(img, dtype=np.float16))
        sample.append(np.array(labels, dtype=np.float16))

        new_data.append(sample)

    target_dir = os.path.join(output_dir, name)

    with open(target_dir, "wb") as out_file:
        np.savez(out_file, *new_data)
    

def test_memory(data_dir):
    import time
    data = []
    total = 0
    for dir in os.listdir(data_dir):
        try:
            array = np.load(os.path.join(data_dir, dir))
            for idx in array:
                tmp = array[idx][0]
                target_img = _resize_image(tmp, (64,64))
                target_img = np.array(target_img, dtype=np.float16)
                data.append(target_img)
                total += 1
                print(total)
        except Exception:
            pass
        break
    
    print(data[0].shape)
    print(data[3].shape)
    tmp = data[3]
    start = time.time()
    tmp *= 255.0
    print(f"Took: {(time.time()-start)*1000}ms")
    
    """
    start = time.time()
    tmp = data[0]
    print(f"Took: {(time.time()-start)*1000}ms")

    start = time.time()
    tmp = data[4370]
    print(f"Took: {(time.time()-start)*1000}ms")
    """

def load_data_to_memory(data_dir, labels_dir, target_size):
    images = []
    labels = []

    for i, dir in enumerate(os.listdir(labels_dir)):
        with open(os.path.join(labels_dir, dir), "r") as label_file:
            for line in label_file.readlines():
                for _ in range(3):
                    tmp = line.split(" ")
                    
                    file_name = tmp[0] + ".chip.jpg"
                    image = np.array(Image.open(os.path.join(data_dir, file_name)), dtype=np.uint8)
                    img_shape = image.shape[:2]
                    image = seq(image=image)
                    image = np.array(_resize_image(image, target_size)*255, dtype=np.uint8)
                    
                    #age, sex, race = file_name.split("_")[:3]
                    features = np.array(tmp[1:-1], dtype=np.float16).reshape(-1, 2)
                    features = np.divide(features, img_shape).flatten()
                    #features = np.concatenate(([int(age)/config.MAX_AGE, int(sex), int(race)/config.MAX_RACE], features))
                    features = np.maximum(features, 0)
                    
                    images.append(image)
                    labels.append(features)

    with open(f"/disk2/datasets/faces_croped/images_{target_size}.pickle", "wb") as out_file:
        pickle.dump(images, out_file)
    with open("/disk2/datasets/faces_croped/labels.pickle", "wb") as out_file:
        pickle.dump(labels, out_file)




if __name__ == "__main__":
    #test_augmentation("/disk2/datasets/faces_croped/images", "/disk2/datasets/faces_croped/labels", config.TARGET_SIZE)

    #create_dataset_multiprocessing(data_dir, target_dir, target_size)

    """
    with open(f"/disk2/datasets/faces_croped/images_{config.TARGET_SIZE}.pickle", "rb") as out_file:
        test = pickle.load(out_file)

        plt.imshow(test[15200]/255)
        plt.show()
    """
    load_data_to_memory("/disk2/datasets/faces_croped/images", "/disk2/datasets/faces_croped/labels", config.TARGET_SIZE)
    #test_memory("/disk2/datasets/faces_dataset")
    """
    import time
    dir = "/disk2/test/Jason_Biggs_0_0.npz"

    start = time.time()

    tmp = np.load(dir, allow_pickle=True)
    test = tmp[0]
    print(f"Took {(time.time()-start)*1000}ms\n")

    dir = "/disk2/test/Jason_Biggs_0_1.npz"

    start = time.time()

    tmp = np.load(dir, allow_pickle=True)
    test = tmp["arr_120"]
    print(f"Took {(time.time()-start)*1000}ms\n")
    """

    """
    data_dir = "/disk2/datasets/faces_dataset/Jason_Biggs_0.npz"
    target_dir = "/disk2/test"
    target_size = (208,172)
    create_dataset((data_dir, target_dir, target_size))
    """

    """
    data_dir = "/disk2/datasets/faces_dataset"
    target_dir = "/disk2/test"
    target_size = (208,172)

    create_labels(target_dir, target_dir + "/labels.pickle")
    #create_dataset_multiprocessing(data_dir, target_dir, target_size)
    """