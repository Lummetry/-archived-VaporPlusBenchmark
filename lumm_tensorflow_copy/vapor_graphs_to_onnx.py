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
from lumm_tensorflow.utils import session_graph_to_onnx
from lumm_tensorflow.vapor_graphs import get_vapor_graph, GRAPHS


if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  N_WARMUP = 1
  N_ITERS = 10
  BATCH_SIZE = 1
  
  log.p('Tensorflow version: {}'.format(tf.__version__))
  
  path_images = os.path.join(log.get_data_subfolder('General'))
  lst_imgs = [os.path.join(path_images, x) for x in os.listdir(path_images)]
  lst_imgs = [cv2.imread(x) for x in lst_imgs]
  np_imgs = np.array(lst_imgs)
  
  log.p('Benchmarking will be made on tensor: {}'.format(np_imgs.shape))
  
  dct_times = {}
  for graph_name in GRAPHS:
    vapor_graph = get_vapor_graph(log, graph_name)
    session_graph_to_onnx(
      log=log, 
      graph=vapor_graph.graph, 
      input_names=vapor_graph.get_input_names(), 
      output_names=vapor_graph.get_output_names(), 
      onnx_name='tf_' + graph_name,
      opset=12
      )
    tf.keras.backend.clear_session()
  #endfor

  

  
  
  
  
  
  
  
  
  
  
  
  
  