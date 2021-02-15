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
import math
import numpy as np
import pandas as pd
from PIL import Image

def get_nr_batches(np_imgs, batch_size):
  nr_batches = int(math.ceil(np_imgs.shape[0] / batch_size))
  return nr_batches

def data_generator(np_imgs, batch_size):
  nr_batches = get_nr_batches(np_imgs, batch_size)
  for i in range(nr_batches):
    start = i * batch_size
    stop = (i + 1) * batch_size if i < nr_batches - 1 else np_imgs.shape[0]
    np_batch = np_imgs[start:stop]
    yield np_batch
  return

def read_images(log, folder):
  path_images = os.path.join(log.get_data_subfolder(folder))
  assert os.path.exists(path_images), 'Path does not exist: {}'.format(path_images)
  lst_imgs = [os.path.join(path_images, x) for x in os.listdir(path_images)]
  lst_imgs = [np.array(Image.open(x)) for x in lst_imgs]
  np_imgs = np.array(lst_imgs)
  return np_imgs

def get_path_results(log, batch_size, fn):
  platform, system = log.get_platform()
  machine_name = log.get_machine_name()
  path = os.path.join(log.get_output_folder(), platform, machine_name, str(batch_size), fn)
  return path

def save_benchmark_results(log, dct_times, batch_size, fn):
  path = get_path_results(log, batch_size, fn)
  os.makedirs(path, exist_ok=True)
  path = os.path.join(path, '{}_{}.csv'.format(log.now_str(), fn))
  df = pd.DataFrame(dct_times)
  log.p('\n\n{}'.format(df))
  log.save_dataframe(
    df=df,
    fn=path,
    full_path=True
    )
  return df