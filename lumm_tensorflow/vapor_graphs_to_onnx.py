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
Please install tf2onnx with: pip install -U tf2onnx
"""


"""
Prolems on conversion with EFFDET0:
EFFDET0 conversion errs: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
"""

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

  

  
  
  
  
  
  
  
  
  
  
  
  
  