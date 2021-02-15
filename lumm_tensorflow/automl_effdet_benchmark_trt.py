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
import constants as ct

from libraries import Logger
from lumm_tensorflow.automl_effdet import get_effdet_pb
from data import get_nr_batches, read_images, save_benchmark_results
from benchmark_methods import benchmark_tf_graph

def benchmark_automl_effdet_trt(log, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Benchmarking AutoMLEffdetTRT on image tensor: {}'.format(np_imgs_bgr.shape))
  dct_times = {}
  for i in range(8):    
    model_name = 'efficientdet-d{}'.format(i)
    try:
      trt_name = os.path.join(model_name, 'tensorrt_fp32', 'saved_model.pb')
      graph, sess, tf_input, tf_output = get_effdet_pb(log, trt_name, is_saved_model=True)
      log.p('Benchmarking {}'.format(model_name))
      preds, lst_time = benchmark_tf_graph(
        log=log,
        sess=sess, 
        tf_inp=tf_input,
        tf_out=tf_output,
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=batch_size, 
        n_warmup=n_warmup, 
        n_iters=n_iters,
        as_rgb=True
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
    fn=ct.AUTOML_EFFDET_TRT
    )
  return


if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  BS = 1
  N_WP = 1
  N_IT = 1
  
  np_imgs_bgr = read_images(log=log, folder=ct.DATA_FOLDER_GENERAL)
  benchmark_automl_effdet_trt(
    log=log,
    np_imgs_bgr=np_imgs_bgr, 
    batch_size=BS, 
    n_warmup=N_WP, 
    n_iters=N_IT
    )
 