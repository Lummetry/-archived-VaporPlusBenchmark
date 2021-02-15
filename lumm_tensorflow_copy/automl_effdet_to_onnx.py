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
The conversion is not working
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
      inputs=['image_arrays:0'],
      outputs=['detections:0'],
      folder=None,
      opset=12
      )
    
    
