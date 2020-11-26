docker run --gpus all -it --rm \
   --env="DISPLAY" \
   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   -w /usr/src/face_detection \
   -v ~/Documents/face_keypoints_detector:/usr/src/face_detection \
   -v ~/Documents/face_keypoints_detector:/photo \
   -v /disk2/ckpts/face_keypoints_detector:/ckpt \
   -v /disk2/datasets/faces_croped:/datasets \
   --device="/dev/video0:/dev/video0" \
   face_detection python3 ./main.py 
