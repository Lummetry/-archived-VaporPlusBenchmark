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
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from data_gen import data_generator
from lumm_tensorflow.automl_effdet import get_effdet_keras
from time import time
from libraries import Logger


"""
This benchmark was executed on tf24 environment. This environment was created with:
  conda create -n tf24 anaconda python=3.7 opencv
  
  The requirements from the automl: https://github.com/google/automl/blob/master/efficientdet/requirements.txt
  remove: git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
  change: tensorflow to tensorflow-gpu
  save into a local file (ex: on D:)
  
  run: pip install -r requirements.txt 
"""

def benchmark_effdet_model(log, model, np_imgs_bgr, batch_size, n_warmup, n_iters,
                       as_rgb=False, resize=None):
  def _predict(np_batch):
    if resize:
      np_batch = np.array([cv2.resize(x, resize) for x in np_batch])
    if as_rgb:
      np_batch = np_batch[:,:,:,::-1]
    preds = model.predict(np_batch)
    return preds
  
  #warmup
  for i in range(n_warmup):
    gen = data_generator(np_imgs=np_imgs_bgr, batch_size=batch_size)
    log.p(' Warmup {}'.format(i))
    for np_batch in gen:
      _predict(np_batch)
  
  #iters
  for i in range(n_iters):
    gen = data_generator(np_imgs=np_imgs_bgr, batch_size=batch_size)
    log.p(' Iter {}'.format(i))
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
  
  N_WARMUP = 1
  N_ITERS = 10
  BATCH_SIZE = 1
  
  log.p('Tensorflow version: {}'.format(tf.__version__))
  
  path_images = os.path.join(log.get_data_subfolder('General'))
  lst_imgs = [os.path.join(path_images, x) for x in os.listdir(path_images)]
  lst_imgs = [cv2.imread(x) for x in lst_imgs]
  np_imgs = np.array(lst_imgs)
  
  log.p('Benchmarking will be made on tensor: {}'.format(np_imgs.shape))
  
  dct_times = {}
  for i in range(8):
    #you need to clear session in order to load every single model. at list for tf2.4.1
    tf.keras.backend.clear_session()  
    model_name = 'efficientdet-d{}'.format(i)
    model = get_effdet_keras(log, 'efficientdet-d{}'.format(i))
    log.log_keras_model(model)
    preds, lst_time = benchmark_effdet_model(
      log=log,
      model=model, 
      np_imgs_bgr=np_imgs, 
      batch_size=BATCH_SIZE, 
      n_warmup=N_WARMUP, 
      n_iters=N_ITERS,
      as_rgb=True
      )
    dct_times[model_name] = lst_time
  #endfor
  
  df = pd.DataFrame(dct_times)
  log.p('\n\n{}'.format(df))
  platform, system = log.get_platform()
  log.save_dataframe(
    df=df,
    fn='{}_{}_{}.csv'.format(platform, 'effdet_keras', log.now_str()),
    folder='output'
    )
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  