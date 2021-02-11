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
Generate pb:
1. cd "D:/Projects/VaporPlusBenchmark/third_party/tensorflow/automl/efficientdet"
2. run:
python model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 --ckpt_path="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/checkpoints/efficientdet-d0" --saved_model_dir="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/savedmodeldir/efficientdet-d0"
python model_inspect.py --runmode=saved_model --model_name=efficientdet-d1 --ckpt_path="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/checkpoints/efficientdet-d1" --saved_model_dir="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/savedmodeldir/efficientdet-d1"
python model_inspect.py --runmode=saved_model --model_name=efficientdet-d2 --ckpt_path="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/checkpoints/efficientdet-d2" --saved_model_dir="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/savedmodeldir/efficientdet-d2"
python model_inspect.py --runmode=saved_model --model_name=efficientdet-d3 --ckpt_path="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/checkpoints/efficientdet-d3" --saved_model_dir="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/savedmodeldir/efficientdet-d3"
python model_inspect.py --runmode=saved_model --model_name=efficientdet-d4 --ckpt_path="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/checkpoints/efficientdet-d4" --saved_model_dir="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/savedmodeldir/efficientdet-d4"
python model_inspect.py --runmode=saved_model --model_name=efficientdet-d5 --ckpt_path="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/checkpoints/efficientdet-d5" --saved_model_dir="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/savedmodeldir/efficientdet-d5"
python model_inspect.py --runmode=saved_model --model_name=efficientdet-d6 --ckpt_path="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/checkpoints/efficientdet-d6" --saved_model_dir="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/savedmodeldir/efficientdet-d6"
python model_inspect.py --runmode=saved_model --model_name=efficientdet-d7 --ckpt_path="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/checkpoints/efficientdet-d7" --saved_model_dir="C:/Users/ETA/Lummetry.AI Dropbox/DATA/_vapor_data/_benchmarks/_models/automl_effdet/savedmodeldir/efficientdet-d7"
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'tensorflow', 'automl'))
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'tensorflow', 'automl', 'efficientdet'))
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'tensorflow', 'automl', 'efficientdet', 'keras'))

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

import hparams_config
import inference
from keras import efficientdet_keras
from libraries import Logger

#this method was build using automl/efficientdet/keras/infer.py

def get_effdet_keras(log, model_name):
  DEBUG = False
  MODEL_NAME = model_name
  MODEL_CKPT = os.path.join(log.get_models_folder(), 'automl_effdet/checkpoints', MODEL_NAME)
  log.p('Creating keras model for {}'.format(MODEL_NAME))

  MODEL_CKPT = os.path.join(log.get_models_folder(), 'automl_effdet/checkpoints', MODEL_NAME)
  
  # Create model config.
  config = hparams_config.get_efficientdet_config(model_name)
  config.is_training_bn = False
  config.image_size = '1920x1280'
  config.nms_configs.score_thresh = 0.4
  config.nms_configs.max_output_size = 100
  # config.override(FLAGS.hparams)

  # Use 'mixed_float16' if running on GPUs.
  policy = tf.keras.mixed_precision.Policy('float32')
  tf.keras.mixed_precision.set_global_policy(policy)
  tf.config.run_functions_eagerly(DEBUG)

  # Create and run the model.
  model = efficientdet_keras.EfficientDetModel(config=config)
  model.build((None, None, None, 3))
  model.load_weights(tf.train.latest_checkpoint(MODEL_CKPT))
  model.summary()
  
  #could not save it as .h5 model
  # log.save_keras_model(model, label='efficientdet-d0')
  return model


def get_effdet_pb(log, model_name):
  PATH_PB = os.path.join(log.get_models_folder(), 'automl_effdet/savedmodeldir')
  path_pb = os.path.join(PATH_PB, model_name, '{}_frozen.pb'.format(model_name))
  log.p('Loading EfficientDet from: {}'.format(path_pb))
  graph = log.load_tf_graph(path_pb)

  sess = tf.Session(graph=graph)
  tf_input = graph.get_tensor_by_name('image_arrays:0')
  tf_output = graph.get_tensor_by_name('detections:0')
  return graph, sess, tf_input, tf_output

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  for i in range(7):
    tf.keras.backend.clear_session()  
    model = get_effdet_keras(log, 'efficientdet-d{}'.format(i))
  
  for i in range(7):
    get_effdet_pb(log, 'efficientdet-d{}'.format(i))
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  