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
import numpy as np
import pandas as pd
import tensorflow as tf

from time import time
from libraries import Logger
from data_gen import data_generator
from lumm_tensorflow.vapor_graphs import GRAPHS, get_vapor_graph
from lumm_tensorflow.utils import get_trt_graph, benchmark_tf_graph

GRAPHS = {
  'EFF_DET0': (574, 1020)
  }

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
  
  dct_times = {}
  for graph_name, resize in GRAPHS.items():
    vg = get_vapor_graph(log, graph_name)
    model, sess, tf_inp, tf_out = get_trt_graph(log, vg.config_graph['GRAPH'])
    log.p('Benchmarking {}'.format(graph_name))
    preds, lst_time = benchmark_tf_graph(
      log=log, 
      sess=sess, 
      tf_inp=tf_inp,
      tf_out=tf_out,
      np_imgs_bgr=np_imgs, 
      batch_size=BATCH_SIZE, 
      n_warmup=N_WARMUP, 
      n_iters=N_ITERS,
      as_rgb=True,
      resize=resize
      )
    dct_times[graph_name] = lst_time
  #endfor
  
  df = pd.DataFrame(dct_times)
  log.p('\n\n{}'.format(df))
  platform, system = log.get_platform()
  log.save_dataframe(
    df=df,
    fn='{}_{}_{}.csv'.format(platform, 'vapor_graphs_trt', log.now_str()),
    folder='output'
    )