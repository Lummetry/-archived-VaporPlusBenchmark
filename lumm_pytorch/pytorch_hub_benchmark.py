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
from data import read_images, save_benchmark_results
from lumm_pytorch.pytorch_hub import MODELS, get_pytorchhub_model
from benchmark_methods import benchmark_torch_hub_model

def benchmark_pytorch_hub_models(log, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Benchmarking PytorchHub {} on image tensor: {}'.format(','.join([x[1] for x in MODELS]), np_imgs_bgr.shape))
  dct_times = {}
  for repo_or_dir, model_name in MODELS:
    try:
      model = get_pytorchhub_model(log, repo_or_dir, model_name)
      log.p('Benchmarking {}'.format(model_name))
      lst_preds, lst_time = benchmark_torch_hub_model(
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
    except Exception as e:
      log.p('Exception on {}: {}'.format(model_name, str(e)))
      dct_times[model_name] = [None] * np_imgs_bgr.shape[0]
  #endfor
    
  save_benchmark_results(
    log=log, 
    dct_times=dct_times, 
    batch_size=batch_size,
    fn=ct.PYTORCH_HUB
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
  benchmark_pytorch_hub_models(
    log=log,
    np_imgs_bgr=np_imgs_bgr, 
    batch_size=BS, 
    n_warmup=N_WP, 
    n_iters=N_IT
    )
