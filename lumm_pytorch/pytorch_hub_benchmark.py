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
import cv2
import torch as th
import numpy as np
import pandas as pd
import torchvision.models as models

from time import time
from libraries import Logger
from data_gen import data_generator
from lumm_pytorch.pytorch_hub import MODELS, get_pytorchhub_model

DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

def benchmark_thhub_model(log, model, np_imgs_bgr, batch_size, n_warmup, n_iters,
                       as_rgb=False, resize=None):
  def _predict(np_batch):
    if resize:
      np_batch = np.array([cv2.resize(x, resize) for x in np_batch])
    if as_rgb:
      np_batch = np_batch[:,:,:,::-1]
    np_batch = np.transpose(np_batch, (0, 3, 1, 2)).astype('float32')
    th_x = th.from_numpy(np_batch).to(DEVICE)
    with th.no_grad():
      th_preds = model(th_x)
      p1 = th_preds[0].cpu().numpy()
      p2 = [x.cpu().numpy() for x in th_preds[1]]
      preds = (p1, p2)
    return preds
  
  #warmup
  for i in range(n_warmup):
    log.p(' Warmup {}'.format(i))
    gen = data_generator(np_imgs=np_imgs_bgr, batch_size=batch_size)
    for np_batch in gen:
      _predict(np_batch)

  
  #iters
  for i in range(n_iters):
    log.p(' Iter {}'.format(i))
    gen = data_generator(np_imgs=np_imgs_bgr, batch_size=batch_size)
    lst_time = []
    for np_batch in gen:
      start = time()
      preds = _predict(np_batch)
      stop = time()
      lst_time.append(stop - start)
  #endfor
  return preds, lst_time

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  BATCH_SIZE = 1
  N_WARMUP = 1
  N_ITERS = 10
  
  path_images = os.path.join(log.get_data_subfolder('General'))
  lst_imgs = [os.path.join(path_images, x) for x in os.listdir(path_images)]
  lst_imgs = [cv2.imread(x) for x in lst_imgs]
  np_imgs = np.array(lst_imgs)
  
  dct_times = {}
  for repo_or_dir, model_name in MODELS:
    model = get_pytorchhub_model(log, repo_or_dir, model_name)
    log.p('Benchmarking {}'.format(model_name))
    preds, lst_time = benchmark_thhub_model(
      log=log,
      model=model, 
      np_imgs_bgr=np_imgs, 
      batch_size=BATCH_SIZE, 
      n_warmup=N_WARMUP, 
      n_iters=N_ITERS,
      as_rgb=True,
      )
    dct_times[model_name] = lst_time
  #endfor
    
  df = pd.DataFrame(dct_times)
  log.p('\n\n{}'.format(df))
  log.save_dataframe(
    df=df,
    fn='{}_{}.csv'.format('pytorch_hub', log.now_str()),
    folder='output'
    )
  