import os
import pickle
import queue
import numpy as np
import progressbar
import matplotlib.pyplot as plt
from skimage.transform import resize
from multiprocessing import Process, cpu_count, Queue, Pool

#data_dir = "/disk2/datasets/faces_dataset/"
labels_dir = "/disk2/datasets/faces_dataset/labels.pickle"

SPLITTER = ":"

def create_labels(data_dir, output_dir):
    labels = []
    widgets = [progressbar.ETA(), " | ", progressbar.Percentage(), " ", progressbar.Bar()]
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


def test(dataset_dir, output_dir, target_size):

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




if __name__ == "__main__":
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

    data_dir = "/disk2/datasets/faces_dataset"
    target_dir = "/disk2/test"
    target_size = (208,172)

    create_labels(target_dir, target_dir + "/labels.pickle")
    #test(data_dir, target_dir, target_size)