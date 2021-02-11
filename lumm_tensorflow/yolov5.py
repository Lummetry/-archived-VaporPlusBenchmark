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

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'tensorflow', 'Yolov5_tf'))

from libraries import Logger
from yolo3_one_file_to_detect_them_all import make_yolov3_model, WeightReader

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  yolov3 = make_yolov3_model()

  # load the weights trained on COCO into the model
  weight_reader = WeightReader(weights_path)
  weight_reader.load_weights(yolov3)