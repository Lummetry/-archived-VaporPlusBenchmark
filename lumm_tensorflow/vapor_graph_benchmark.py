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
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf_runoptions = tf.RunOptions(report_tensor_allocations_upon_oom=True)

from libraries import Logger
from benchmark_methods import benchmark_vapor_graph
from data import read_images, save_benchmark_results
from lumm_tensorflow.vapor_graphs import GRAPHS, get_vapor_graph

"""
This script tests vapor_graphs inference times (postprocess step included)
"""

def benchmark_vapor_graphs(log, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Benchmarking VaporGraphs {} on image tensor: {}'.format(','.join(GRAPHS.keys()), np_imgs_bgr.shape))
  dct_times = {}
  for model_name, _ in GRAPHS.items():
    try:
      graph = get_vapor_graph(log, model_name)
      log.p('Benchmarking {}'.format(model_name))
      lst_preds, lst_time = benchmark_vapor_graph(
        log=log,
        n_warmup=n_warmup, 
        n_iters=n_iters,
        graph=graph, 
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=batch_size, 
        as_rgb=True
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
    fn=ct.VAPOR_GRAPHS
    )
  return

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  log.set_nice_prints(df_precision=5)

  log.p('Tensorflow version: {}'.format(tf.__version__))
  
  BS = 1
  N_WP = 1
  N_IT = 1
  
  np_imgs_bgr = read_images(log=log, folder=ct.DATA_FOLDER_GENERAL)
  benchmark_vapor_graphs(
    log=log,
    np_imgs_bgr=np_imgs_bgr, 
    batch_size=BS, 
    n_warmup=N_WP, 
    n_iters=N_IT
    )
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  