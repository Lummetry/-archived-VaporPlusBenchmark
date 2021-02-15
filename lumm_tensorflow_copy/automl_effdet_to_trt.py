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
import cv2
import numpy as np
import tensorflow as tf

from libraries import Logger
from lumm_tensorflow.utils import graph_to_trt

"""
This conversion doesn't work: ValueError: NodeDef mentions attr 'explicit_paddings' not in Op<name=MaxPool; signature=input:T -> output:T; attr=T:type,default=DT_FLOAT,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16, DT_QINT8]; attr=ksize:list(int),min=4; attr=strides:list(int),min=4; attr=padding:string,allowed=["SAME", "VALID"]; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW", "NCHW_VECT_C"]>; NodeDef: {{node resample_p6/max_pooling2d/MaxPool}}. (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).
"""

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  PATH_PB = os.path.join(log.get_models_folder(), 'automl_effdet/savedmodeldir')
  for i in range(1):
    model_name = 'efficientdet-d{}'.format(i)
    # path_pb = os.path.join(PATH_PB, model_name, '{}_frozen.pb'.format(model_name))
    path_pb = '{}_frozen.pb'.format(model_name)
    graph_to_trt(
      log=log,
      graph_name=path_pb,
      folder=None,
      input_name=['image_arrays:0'],
      output_name=['detections:0']
      )
  
  
  
  
  
  
  
  
  
  
  
  