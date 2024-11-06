#packages 
import pyniryo 
from pyniryo2 import *
from PIL import Image, ImageOps
from FasterRCNN_detector import FasterRCNN_Detector
import cv2
import time 
import numpy as np
import os 

#connecting the robot
client = NiryoRobot("192.168.0.101")
client.arm.calibrate_auto()
client.arm.move_joints([1.573, 0.081, -0.896, 0.080, -1.011, -0.109])

#calibration parameters 
mtx, dst = client.vision.get_camera_intrinsics()

data_path = r"C:\Users\ttmpa\Desktop\od_models\FasterR_CNN\labelmap.pbtxt"
model_path = r"C:\Users\ttmpa\Desktop\od_models\FasterR_CNN\saved_model"
detector = FasterRCNN_Detector(data_path, model_path)

while True: 
    #Getting the image from the robot 
    img_compressed = client.vision.get_img_compressed()
    img_raw = pyniryo.uncompress_image(img_compressed)
    img_undistort = pyniryo.undistort_image(img_raw, mtx, dst)
    img = img_undistort.copy()

    #getting predicitions 
    start_time = time.time()
    preds, info = detector.predictions(img)
    end_time =time.time()
    inference_time = end_time - start_time

    #showing images 
    text = f"The inference time is {inference_time:.4} seconds"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2 )
    cv2.imshow('Vision validation', preds)
    if cv2.waitKey(0) == 27:
        break