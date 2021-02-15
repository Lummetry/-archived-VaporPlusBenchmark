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
from lumm_tensorflow.vapor_graphs import get_vapor_graph, GRAPHS

#EfficientDet hangs at conversion

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  dct_times = {}
  for graph_name in GRAPHS:
    vapor_graph = get_vapor_graph(log, graph_name)
    path = os.path.join(log.get_models_folder(), vapor_graph.config_graph['GRAPH'])
    graph_to_trt(
      log=log,
      graph_name=path,
      folder=None,
      input_name=vapor_graph.get_input_names(),
      output_name=vapor_graph.get_output_names()
      )
  #endfor

  

  
  
  
  
  
  
  
  
  
  
  
  
  