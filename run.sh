docker run --gpus all -it --rm --env="DISPLAY" \
   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   -w /usr/src/face_detection \
   -v ~/Documents/face_detection:/usr/src/face_detection \
   -v ~/Documents/face_detection:/photo \
   -v /disk2/ckpt:/ckpt \
   -v ~/Documents/datasets/test:/datasets \
   --device="/dev/video0:/dev/video0" \
   face_detection python3 ./test.py
