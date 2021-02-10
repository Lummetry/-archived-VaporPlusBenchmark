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
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf_runoptions = tf.RunOptions(report_tensor_allocations_upon_oom=True)

from time import time
from libraries import Logger
from data_gen import data_generator


MOBILENET_V2  = 'mobilenetv2'
INCEPTION_V3  = 'inceptionv3'
RESNET50      = 'resnet50'

RESIZE = {
  MOBILENET_V2: (224, 224),
  INCEPTION_V3: (299, 299),
  RESNET50    : (299, 299)
  }

def get_pb(log, model_name):
  tf_graph, s_input, s_output = log.load_graph_from_models(
    model_name='{}.pb'.format(model_name),
    get_input_output=True
    )
  sess = tf.Session(graph=tf_graph)
  return tf_graph, s_input, s_output, sess

def benchmark_tf_graph(log, sess, tf_inp, tf_out, 
                       np_imgs_bgr, batch_size, n_warmup, n_iters,
                       as_rgb=False, resize=None):
  def _predict(np_batch):
    if resize:
      np_batch = np.array([cv2.resize(x, resize) for x in np_batch])
    if as_rgb:
      np_batch = np_batch[:,:,:,::-1]
    out_scores = sess.run(
      tf_out,
      feed_dict={tf_inp: np_batch},
      options=tf_runoptions
      )
    return out_scores
  
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
      out_scores = _predict(np_batch)
      stop = time()
      lst_time.append(stop - start)
  #endfor
  return out_scores, lst_time


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
  for model_name in [MOBILENET_V2, INCEPTION_V3, RESNET50]:
    graph, sess, tf_input, tf_output = get_pb(log, model_name)
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
  log.save_dataframe(
    df=df,
    fn='{}_{}.csv'.format('tf_graph', log.now_str()),
    folder='output'
    )
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  