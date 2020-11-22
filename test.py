from PIL import Image
from model import make_or_restore_model
from utils.utils import _resize_image, plot_image_points
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import face_recognition
import config
#matplotlib.use('TKAgg') #docker

CHECKPOINT_DIR = "/disk2/ckpt"

model = make_or_restore_model(CHECKPOINT_DIR)

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) 
    face_locations = face_recognition.face_locations(small_frame)

    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        print(top, right, bottom, left)

        face_croped = frame[top:top + bottom-top, left:left + right-left]
        original_shape = face_croped.shape
        face_croped = _resize_image(face_croped, config.TARGET_SIZE)
        X = np.empty((1, *config.TARGET_SIZE, 3))
        X[0] = face_croped

        labels = model.predict(X)[0]
        #age, sex, race = labels[len(labels)-4:-1]
        #print(f"Age: {age*config.MAX_AGE}")
        #print(f"Sex: {sex}")
        #print(f"Race: {race*config.MAX_RACE}")
        #labels = labels[:len(labels)-3]
        labels = np.reshape(labels, (68,2))

        for point in labels:
            pt = point * original_shape[0:2]
            frame = cv2.circle(frame, (int(pt[0]) + left, int(pt[1]) + top) , radius=1, color=(0,255,0))

        cv2.imshow("test", face_croped)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


