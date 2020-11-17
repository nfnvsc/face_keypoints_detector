from PIL import Image
from model import make_or_restore_model
from utils import _resize_image, plot_image_points
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
#matplotlib.use('TKAgg') #docker

CHECKPOINT_DIR = "/disk2/ckpt"
cap = cv2.VideoCapture(0)
model = make_or_restore_model(CHECKPOINT_DIR)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    X = np.empty((1, 208, 172, 3))
    X[0] = _resize_image(frame, (208,172))

    labels = model.predict(X)
    labels = np.reshape(labels, (68,2))

    for point in labels:
        print(point)
        print(frame.shape)
        pt = point * frame.shape[0:2]
        frame = cv2.circle(frame, (int(pt[0]), int(pt[1])), radius=1, color=(0,0,255))

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


