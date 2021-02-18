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

def get_nr_batches(np_imgs, batch_size):
  nr_batches = int(math.ceil(len(np_imgs) / batch_size))
  return nr_batches

def data_generator(np_imgs, batch_size):
  nr_batches = get_nr_batches(np_imgs, batch_size)
  for i in range(nr_batches):
    start = i * batch_size
    stop = (i + 1) * batch_size if i < nr_batches - 1 else len(np_imgs)
    np_batch = np_imgs[start:stop]
    #just to make sure that VaporEffDet is working properly
    if len(np_batch) < batch_size:
      bs, h, w, c = np_batch.shape
      np_batch = np.vstack([np_batch, np.repeat(np.expand_dims(np_batch[0], axis=0), batch_size-bs, axis=0)])
    yield np_batch
  return

def read_images(log, folder):
  path_images = os.path.join(log.get_data_subfolder(folder))
  assert os.path.exists(path_images), 'Path does not exist: {}'.format(path_images)
  lst_paths = [os.path.join(path_images, x) for x in os.listdir(path_images)]
  lst_imgs = [cv2.imread(x) for x in lst_paths]
  np_imgs = np.array(lst_imgs)
  return lst_paths, np_imgs

def get_path_batch_size(log, batch_size):
  platform, system = log.get_platform()
  machine_name = log.get_machine_name()
  path = os.path.join(log.get_output_folder(), platform, machine_name, str(batch_size))
  return path

def get_path_results(log, batch_size, fn):
  path = get_path_batch_size(log, batch_size)
  path = os.path.join(path, fn)
  return path

def save_results(log, dct, batch_size, fn):
  path = get_path_results(log, batch_size, fn)
  path = os.path.join(path, '{}.csv'.format(log.now_str()))
  os.makedirs(os.path.split(path)[0], exist_ok=True)
  df = pd.DataFrame(dct)
  log.p('\n\n{}'.format(df))
  log.save_dataframe(
    df=df,
    fn=path,
    full_path=True
    )
  return df

def save_benchmark_results(log, dct_times, batch_size, fn):
  df = save_results(log=log, dct=dct_times, batch_size=batch_size, fn=fn)
  return df

def save_quality_results(log, dct_detections, batch_size, fn):
  df = save_results(log=log, dct=dct_detections, batch_size=batch_size, fn=fn)
  return df

