import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util

class FasterRCNN_Detector():
    def __init__(self, labelmap_path, model_dir_path):
        #load labelmap.pbtxt
        label_map = label_map_util.load_labelmap(labelmap_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=5, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        #load fasterr_cnn model
        tf.keras.backend.clear_session()
        self.interpreter = tf.saved_model.load(model_dir_path)

    def predictions(self, image):
        #load image 
        im_height, im_width, _ = image.shape
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        input_tensor = np.expand_dims(image, 0) # (1, W, H, 3)
        detections = self.interpreter(input_tensor)
        
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()

        detections_info = []
        for idx in range(len(boxes)):
            if scores[idx] >= 0.5:
                y_min = int(boxes[idx][0] * im_height)
                x_min = int(boxes[idx][1] * im_width)
                y_max = int(boxes[idx][2] * im_height)
                x_max = int(boxes[idx][3] * im_width)
                score = scores[idx]
                object_name = self.category_index[int(classes[idx])]['name']
                detections_info.append([object_name, f'{score*100:.2f}%'])
                
                cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (10, 255, 0), 2)
                label = '%s: %d%%' % (object_name, (scores[idx]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(y_min, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (x_min, label_ymin-labelSize[1]-10), (x_min+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(image, label, (x_min, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        return image, detections_info


