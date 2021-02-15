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
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from libraries import Logger
from lumm_tensorflow.keras import MODELS, get_keras_model

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  lst = list(MODELS.keys())
  for model_name in lst:
    tf.keras.backend.clear_session()
    model = get_keras_model(model_name)
    log.save_keras_model(model, label=model_name)
    tf.keras.backend.clear_session()
    log.model_h5_to_graph(model_name, model_name=model_name + '.pb')
