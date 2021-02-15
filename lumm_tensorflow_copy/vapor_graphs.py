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
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf_runoptions = tf.RunOptions(report_tensor_allocations_upon_oom=True)

from data_gen import data_generator
from time import time
from libraries import Logger

from vapor_inference.graphs import YoloInferenceGraph, EfficientDet0InferenceGraph

TF_YOLO = 'TF_YOLO'
EFF_DET0 = 'EFF_DET0'

GRAPHS = {
  TF_YOLO: (608, 608), 
  EFF_DET0: (574, 1020)
  }

def get_vapor_graph(log, graph_name):
  path_config = 'vapor_inference/inference.txt'
  if graph_name == TF_YOLO:
    graph = YoloInferenceGraph(log=log, config_path=path_config)
  elif graph_name == EFF_DET0:
    graph = EfficientDet0InferenceGraph(log=log, config_path=path_config)
  return graph
  
  
  
  
  
  
  
  
  
  
  
  
  