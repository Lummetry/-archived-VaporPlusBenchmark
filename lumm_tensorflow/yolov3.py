"""
Copyright 2019 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.


* NOTICE:  All information contained herein is, and remains
* the property of Knowledge Investment Group SRL.  
* The intellectual and technical concepts contained
* herein are proprietary to Knowledge Investment Group SRL
* and may be covered by Romanian and Foreign Patents,
* patents in process, and are protected by trade secret or copyright law.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Knowledge Investment Group SRL.


@copyright: Lummetry.AI
@author: Lummetry.AI
@project: 
@description:
"""

"""
This implementation is not optimized!!!!!
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'third-party', 'tensorflow', 'keras_yolo3'))
import numpy as np

from libraries import Logger
from yolo3_one_file_to_detect_them_all import make_yolov3_model, WeightReader, preprocess_input, decode_netout, correct_yolo_boxes, do_nms

def get_keras_yolo3():
  PATH = os.path.join(log.get_models_folder(), 'keras_yolo3', 'yolov3.weights')
  # make the yolov3 model to predict 80 classes on COCO
  yolov3 = make_yolov3_model()

  # load the weights trained on COCO into the model
  weight_reader = WeightReader(PATH)
  weight_reader.load_weights(yolov3)
  return yolov3

def predict(model, image):
  net_h, net_w = 416, 416
  obj_thresh, nms_thresh = 0.5, 0.45
  anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
  labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

  

  new_image = preprocess_input(image, net_h, net_w)

  # run the prediction
  yolos = model.predict(new_image)
  boxes = []

  for i in range(len(yolos)):
      # decode the output of the network
      boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

  image_h, image_w, _ = image.shape
  # correct the sizes of the bounding boxes
  correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

  # suppress non-maximal boxes
  do_nms(boxes, nms_thresh)
  return boxes

  
if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  model = get_keras_yolo3()
  predict(model, np.random.randint(0, 255, (300, 300, 3)))

