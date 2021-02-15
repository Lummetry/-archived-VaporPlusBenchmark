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

from time import time
from torch2trt import TRTModule
from data_gen import data_generator
from libraries import Logger
from lumm_pytorch import utils
from lumm_pytorch.pytorch_applications import to_onnx_model, load_onnx_model, \
  get_th_model, MODELS


DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


def benchmark_th_model(model, np_imgs_bgr, batch_size, n_warmup, n_iters,
                       as_rgb=False, resize=None):
  def _predict(np_batch):
    if resize:
      np_batch = np.array([cv2.resize(x, resize) for x in np_batch])
    if as_rgb:
      np_batch = np_batch[:,:,:,::-1]
    np_batch = np.transpose(np_batch, (0, 3, 1, 2)).astype('float32')
    with th.no_grad():
      th_x = th.from_numpy(np_batch).to(DEVICE)
      preds = model(th_x).cpu().numpy()
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
  log.set_nice_prints(df_precision=5)
  
  BATCH_SIZE = 1
  N_WARMUP = 10
  N_ITERS = 10
  
  path_images = os.path.join(log.get_data_subfolder('General'))
  lst_imgs = [os.path.join(path_images, x) for x in os.listdir(path_images)]
  lst_imgs = [cv2.imread(x) for x in lst_imgs]
  np_imgs = np.array(lst_imgs)
  
  dct_times = {}
  for model_name, resize in MODELS.items():
    model_trt = TRTModule()
    model_trt.load_state_dict(th.load('th_{}_trt.pth'.format(model_name)))
    
    preds, lst_time = benchmark_th_model(
      model=model_trt, 
      np_imgs_bgr=np_imgs, 
      batch_size=BATCH_SIZE, 
      n_warmup=N_WARMUP, 
      n_iters=N_ITERS,
      as_rgb=True,
      resize=MODELS[model_name]
      )
    dct_times[model_name] = lst_time
  #endfor
    
  df = pd.DataFrame(dct_times)
  log.p('\n\n{}'.format(df))
  log.p('Mean values: \n{}'.format(df.mean()))
  platform, system = log.get_platform()
  log.save_dataframe(
    df=df,
    fn='{}_{}_{}.csv'.format(platform, 'pytorch_applications_onnx', log.now_str()),
    folder='output'
    )
  
  
  
  
  
  
  
  
  