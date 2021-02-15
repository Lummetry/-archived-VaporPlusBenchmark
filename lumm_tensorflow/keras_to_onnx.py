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
Please make sure to install: keras2onnx

pip install keras2onnx

"""

import numpy as np

from libraries import Logger
from lumm_tensorflow.utils import keras_to_onnx, load_onnx_model
from lumm_tensorflow.keras import MODELS, get_keras_model

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
    
  for model_name, resize in MODELS.items():
    np_batch = np.random.randint(0, 255, (2, resize[0], resize[1], 3)).astype(np.float32)
    log.p('Converting keras {} to onnx model'.format(model_name))
    model = get_keras_model(model_name)
    keras_to_onnx(
      log=log,
      model=model,
      onnx_name=model_name
      )
    
    model_onnx, ort_sess, input_names, output_names = load_onnx_model(
      log=log,
      model_name=model_name + '.onnx',
      )
    
    log.p(input_names, output_names)
    preds = ort_sess.run(
      output_names=output_names, 
      input_feed={input_names[0]: np_batch}
      )
    
    