FROM tensorflow/tensorflow:latest-gpu

WORKDIR /usr/src/face_detection

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

RUN apt install libgl1-mesa-glx -y
#ignore user input when needed
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-tk