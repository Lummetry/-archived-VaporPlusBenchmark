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
import torch as th
import constants as ct

from libraries import Logger
from torch2trt import TRTModule
from data import get_nr_batches, read_images, save_benchmark_results
from benchmark_methods import benchmark_pytorch_model
from lumm_pytorch.pytorch_applications import MODELS

def benchmark_pytorch_models_trt(log, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Benchmarking PytorchTRT {} on image tensor: {}'.format(','.join(MODELS.keys()), np_imgs_bgr.shape))
  dct_times = {}
  for model_name, resize in MODELS.items():
    try:
      path_model = os.path.join(log.get_models_folder(), 'th_{}_trt.pth'.format(model_name))
      model_trt = TRTModule()
      model_trt.load_state_dict(th.load(path_model))
      
      lst_preds, lst_time = benchmark_pytorch_model(
        log=log,
        model=model_trt, 
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=batch_size, 
        n_warmup=n_warmup, 
        n_iters=n_iters,
        as_rgb=True,
        resize=resize,
        to_nchw=True
        )
      dct_times[model_name] = lst_time
      del model_trt
      log.clear_gpu_memory()
    except Exception as e:
      log.p('Exception on {}: {}'.format(model_name, str(e)))
      dct_times[model_name] = [None] * get_nr_batches(np_imgs_bgr, batch_size)
  #endfor
    
  save_benchmark_results(
    log=log, 
    dct_times=dct_times, 
    batch_size=batch_size,
    fn=ct.PYTORCH_TRT
    )
  return
  
if __name__ == '__main__':  
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt', 
    TF_KERAS=False
    )
  log.set_nice_prints(df_precision=5)
  
  BS = 1
  N_WP = 1
  N_IT = 1
  
  np_imgs_bgr = read_images(log=log, folder=ct.DATA_FOLDER_GENERAL)
  benchmark_pytorch_models_trt(
    log=log,
    np_imgs_bgr=np_imgs_bgr, 
    batch_size=BS, 
    n_warmup=N_WP, 
    n_iters=N_IT
    )
  
  
  
  