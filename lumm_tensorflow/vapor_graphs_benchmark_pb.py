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
import tensorflow as tf

from libraries import Logger
from benchmark_methods import benchmark_tf_graph
from data import get_nr_batches, read_images, save_benchmark_results
from lumm_tensorflow.vapor_graphs import GRAPHS, get_vapor_graph

"""
This script tests only the graphs from vapor_graphs (postprocess step not included!!!)
"""

def benchmark_vapor_graphs_pb(log, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Benchmarking VaporGraphsPB {} on image tensor: {}'.format(','.join(GRAPHS.keys()), np_imgs_bgr.shape))
  dct_times = {}
  for model_name, resize in GRAPHS.items():
    try:
      graph = get_vapor_graph(log, model_name, batch_size)
      sess = graph.sess
      tf_inp = graph.get_input_tensors()
      tf_out = graph.get_output_tensors()
      log.p('Benchmarking {}'.format(model_name))
      lst_preds, lst_time = benchmark_tf_graph(
        log=log,    
        n_warmup=n_warmup, 
        n_iters=n_iters,
        sess=sess,
        tf_inp=tf_inp,
        tf_out=tf_out,
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=batch_size, 
        as_rgb=True,
        resize=resize
        )
      dct_times[model_name] = lst_time
      del graph
      del sess
      log.clear_gpu_memory()
    except Exception as e:
      log.p('Exception on {}: {}'.format(model_name, str(e)))
      dct_times[model_name] = [None] * get_nr_batches(np_imgs_bgr, batch_size)
  #endfor
  
  save_benchmark_results(
    log=log, 
    dct_times=dct_times, 
    batch_size=batch_size,
    fn=ct.VAPOR_GRAPHS_PB
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
  benchmark_vapor_graphs_pb(
    log=log,
    np_imgs_bgr=np_imgs_bgr, 
    batch_size=BS, 
    n_warmup=N_WP, 
    n_iters=N_IT
    )
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  