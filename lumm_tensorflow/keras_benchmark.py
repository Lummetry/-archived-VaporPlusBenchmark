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
from data import get_nr_batches, read_images, save_benchmark_results
from lumm_tensorflow.keras import get_keras_model, MODELS

def benchmark_keras_models(log, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Benchmarking KerasModels {} on image tensor: {}'.format(','.join(MODELS.keys()), np_imgs_bgr.shape))
  dct_times = {}
  for model_name, resize in MODELS.items():
    try:
      model = get_keras_model(model_name)
      log.log_keras_model(model)
      log.p('Benchmarking {}'.format(model_name))
      preds, lst_time = benchmark_keras_model(
        log=log,
        model=model, 
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=batch_size, 
        n_warmup=n_warmup, 
        n_iters=n_iters,
        as_rgb=True,
        resize=resize
        )
      dct_times[model_name] = lst_time
      del model
      log.clear_gpu_memory()
    except Exception as e:
      log.p('Exception on {}: {}'.format(model_name, str(e)))
      dct_times[model_name] = [None] * get_nr_batches(np_imgs_bgr, batch_size)
  #endfor
  
  save_benchmark_results(
    log=log, 
    dct_times=dct_times, 
    batch_size=batch_size,
    fn=ct.KERAS
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
  benchmark_keras_models(
    log=log,
    np_imgs_bgr=np_imgs_bgr, 
    batch_size=BS, 
    n_warmup=N_WP, 
    n_iters=N_IT
    )
  