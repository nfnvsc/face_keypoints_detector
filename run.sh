docker run --gpus all -it --rm --env="DISPLAY" \
   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   -w /usr/src/face_detection \
   -v ~/Documents/face_detection:/usr/src/face_detection \ #source code
   -v ~/Documents/face_detection:/photo \ #not used
   -v /disk2/ckpt:/ckpt \ #checkpoints dir
   -v ~/Documents/datasets/test:/datasets \ #dataset dir
   --device="/dev/video0:/dev/video0" \ #enable camera
   face_detection python3 ./main.py #use test.py rather than main.py to live camera detection
