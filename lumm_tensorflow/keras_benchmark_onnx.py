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

from libraries import Logger
from lumm_tensorflow.keras import MODELS
from lumm_tensorflow.utils import load_onnx_model
from benchmark_methods import benchmark_onnx_model
from data import read_images, save_benchmark_results

def benchmark_keras_models_onnx(log, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Benchmarking KerasModelsONNX {} on image tensor: {}'.format(','.join(MODELS.keys()), np_imgs_bgr.shape))
  dct_times = {}
  for model_name, resize in MODELS.items():
    try:
      model, ort_sess, inputs, outputs = load_onnx_model(log, model_name)
      log.p('Benchmarking {}'.format(model_name))
      preds, lst_time = benchmark_onnx_model(
        log=log, 
        ort_sess=ort_sess, 
        input_name=inputs[0],
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=batch_size, 
        n_warmup=n_warmup, 
        n_iters=n_iters,
        as_rgb=True,
        resize=resize
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
    fn=ct.KERAS_ONNX
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
  benchmark_keras_models_onnx(
    log=log,
    np_imgs_bgr=np_imgs_bgr, 
    batch_size=BS, 
    n_warmup=N_WP, 
    n_iters=N_IT
    )