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

import constants as ct
import tensorflow as tf

from libraries import Logger
from benchmark_methods import benchmark_keras_model
from data import read_images, save_benchmark_results
from lumm_tensorflow.automl_effdet import get_effdet_keras

"""
This benchmark was executed on tf24 environment. This environment was created with:
  conda create -n tf24 anaconda python=3.7 opencv
  
  The requirements from the automl: https://github.com/google/automl/blob/master/efficientdet/requirements.txt
  remove: git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
  change: tensorflow to tensorflow-gpu
  save into a local file (ex: on D:\)
  
  run: pip install -r requirements.txt 
"""

def benchmark_automl_effdet_keras(log, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Benchmarking AutoMLEffdetKeras on image tensor: {}'.format(np_imgs_bgr.shape))
  dct_times = {}
  for i in range(8):
    model_name = 'efficientdet-d{}'.format(i)
    try:
      #you need to clear session in order to load every single model. at list for tf2.4.0
      tf.keras.backend.clear_session() 
      model = get_effdet_keras(log, 'efficientdet-d{}'.format(i))
      log.log_keras_model(model)
      preds, lst_time = benchmark_keras_model(
        log=log,
        model=model, 
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=batch_size, 
        n_warmup=n_warmup, 
        n_iters=n_iters,
        as_rgb=True
        )
      dct_times[model_name] = lst_time
    except Exception as e:
      log.p('Exception on {}: {}'.format(model_name, str(e)))
      dct_times[model_name] = [None] * np_imgs_bgr.shape[0]
  #endfor
  
  save_benchmark_results(
    log=log, 
    dct_times=dct_times, 
    batch_size=batch_size,
    fn=ct.AUTOML_EFFDET
    )
  return

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  BS = 1
  N_WP = 1
  N_IT = 1
  
  np_imgs_bgr = read_images(log=log, folder=ct.DATA_FOLDER_GENERAL)
  benchmark_automl_effdet_keras(
    log=log,
    np_imgs_bgr=np_imgs_bgr, 
    batch_size=BS, 
    n_warmup=N_WP, 
    n_iters=N_IT
    )
  
  
  
  
  
  
  
  