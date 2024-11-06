# Machine-Vision-for-Sorting

This repository contains three object detection models trained for sorting mechanical fasteners such as bolts, screws, nuts, washers, and bearings. 
## Setting Up the Environment

Before loading the models, it's recommended to create a virtual environment for installing the required packages. You can set up and activate this virtual environment as the kernel in IDEs like VSCode and Anaconda.

### YOLO (YOLOv5)

The `yolov5` folder contains a `requirements.txt` file that lists the necessary libraries for running the YOLO model. Install them with the following command:

```bash
pip install -r yolov5/requirements.txt
```

### Single Shot Detector (SSD)

To use the SSD model, install the following packages:

```bash
pip install tensorflow opencv-python protobuf==3.20.*
```

### Faster R-CNN

1. **Clone the TensorFlow Models Repository:**

   ```bash
   git clone https://github.com/tensorflow/models.git
   ```

2. **Python Package Installation:**

   Navigate to the `models/research` directory:

   ```bash
   cd models/research
   ```

3. **Install Protocol Buffers:**

   Download the protobuf compiler from [Protocol Buffers Downloads](https://protobuf.dev/downloads/). After downloading, run the following command to compile the protocol buffers:

   ```bash
   <file location of protoc.exe> object_detection/protos/*.proto --python_out=.
   ```

4. **Setup Configuration:**

   Copy the `object_detection/packages/tf2/setup.py` file to the `models/research` folder:

   ```bash
   cp object_detection/packages/tf2/setup.py .
   ```
   You can either run the code or copy and paste it manually.
   
6. **Install the Object Detection API:**

   Finally, install the TensorFlow Object Detection API:

   ```bash
   python -m pip install --use-feature=2020-resolver .
   ```

## Usage

Once the environment and packages are set up, you can load and use each model for fastener detection. Each model folder contains sample code and configuration files to help you get started quickly.
