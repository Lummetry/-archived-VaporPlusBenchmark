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
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'tensorflow', 'automl'))
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'tensorflow', 'automl', 'efficientdet'))
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'tensorflow', 'automl', 'efficientdet', 'keras'))

from libraries import Logger
from lumm_tensorflow.utils import load_onnx_model, graph_to_onnx

"""
The conversion is not working:
  File "/home/eta/anaconda3/envs/tf23/lib/python3.7/site-packages/numpy/lib/function_base.py", line 2151, in _get_ufunc_and_otypes
    outputs = func(*inputs)
  File "/home/eta/anaconda3/envs/tf23/lib/python3.7/site-packages/tf2onnx/tf_utils.py", line 62, in <lambda>
    decode = np.vectorize(lambda x: x.decode('UTF-8'))
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte

[BENCHMARK][2021-02-14 14:46:55] Done saving onnx model

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
    graph_to_onnx(
      log=log, 
      graph_name=path_pb, 
      onnx_name=model_name,
      inputs='image_files:0',
      outputs='detections:0',
      folder=None,
      opset=12
      )
  
  graph = log.load_tf_graph(pb_file=path_pb)
  lst = log.get_tensors_in_tf_graph(graph)
  print(lst[0], lst[-1])
    
