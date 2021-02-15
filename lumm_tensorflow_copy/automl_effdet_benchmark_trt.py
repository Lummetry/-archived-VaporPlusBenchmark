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
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf_runoptions = tf.RunOptions(report_tensor_allocations_upon_oom=True)

from data_gen import data_generator
from lumm_tensorflow.automl_effdet import get_effdet_pb
from time import time
from libraries import Logger
from lumm_tensorflow.utils import benchmark_tf_graph


if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  BATCH_SIZE = 1
  N_WARMUP = 10
  N_ITERS = 10
  
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
    trt_name = os.path.join(model_name, 'tensorrt_fp32', 'saved_model.pb')
    graph, sess, tf_input, tf_output = get_effdet_pb(log, trt_name, is_saved_model=True)
    log.p('Benchmarking {}'.format(model_name))
    preds, lst_time = benchmark_tf_graph(
      log=log,
      sess=sess, 
      tf_inp=tf_input,
      tf_out=tf_output,
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
    fn='{}_{}_{}.csv'.format(platform, 'effdet_pb', log.now_str()),
    folder='output'
    )
  
 