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
Remember to install timm and omegaconf in your environment
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'pytorch', 'efficientdet'))
import torch as th
import constants as ct

from libraries import Logger
from data import get_nr_batches, read_images, save_benchmark_results
from benchmark_methods import benchmark_pytorch_effdet_model
from lumm_pytorch.pytorch_effdet import THH_EFFDET_MODEL_LIST, get_th_effdet_model

DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

def benchmark_pytorch_effdet_models(log, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Benchmarking PytorchEFFDET on image tensor: {}'.format(np_imgs_bgr.shape))
  
  dct_times = {}
  for dct_info in THH_EFFDET_MODEL_LIST:
    try:
      model_name = dct_info['model_name']
      model = get_th_effdet_model(model_name)
      log.p('Benchmarking {}'.format(model_name))
      lst_preds, lst_time = benchmark_pytorch_effdet_model(
        log=log,
        model=model, 
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=batch_size, 
        n_warmup=n_warmup, 
        n_iters=n_iters,
        as_rgb=True,
        to_nchw=True
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
    fn=ct.PYTORCH_EFFDET
    )
  
  return

if __name__ == '__main__':
  log = Logger(lib_name='VPBMRK', config_file='config.txt', max_lines=1000)
  log.set_nice_prints(df_precision=5)

  BS = 2
  N_WP = 1
  N_IT = 1
  
  np_imgs_bgr = read_images(log=log, folder=ct.DATA_FOLDER_GENERAL)
  benchmark_pytorch_effdet_models(
    log=log,
    np_imgs_bgr=np_imgs_bgr, 
    batch_size=BS, 
    n_warmup=N_WP, 
    n_iters=N_IT
    )

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  