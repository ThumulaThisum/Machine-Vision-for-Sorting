# Machine-Vision-for-Sorting

This repository contains three object detection models that I have trained to sort mechanical fasteners such as bolts, screws, nuts, washers, and bearings. 


Before loading the models I recommend creating a virtual environment in command prompt to install the required packages and activate the virtual environemnt as the kernel in IDEs like VScode and anaconda. 


1. Yolo - yolov5 folder contains a requirements.txt file that has the needed libraries that you need to install to load the yolo model.

2. SDD - pip install tensorflow opencv-python protobuf==3.20.*

3. FasterRCNN -

3.1 Clone the TensorFlow Models repository and proceed to one of the installation options.
  git clone https://github.com/tensorflow/models.git

3.2 Python Package Installation
  cd models/research

Download the protobuf version from https://protobuf.dev/downloads/ (or by searching Google Protocol Buffers). After downloading the protoc.exe run the following command.

  (file location of protoc.exe) object_detection/protos/*.proto --python_out=.

3.3 Copy object_detection/packages/tf2/setup.py file and paste it to models/research folder 

3.4 Install the object detection API 
  python -m pip install --use-feature=2020-resolver .

