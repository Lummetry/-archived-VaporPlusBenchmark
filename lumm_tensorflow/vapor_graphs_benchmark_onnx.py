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

from libraries import Logger
from lumm_tensorflow.vapor_graphs import GRAPHS
from lumm_tensorflow.utils import load_onnx_model
from benchmark_methods import benchmark_onnx_model
from data import get_nr_batches, read_images, save_benchmark_results

def benchmark_vapor_graphs_onnx(log, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Benchmarking VaporGraphsONNX {} on image tensor: {}'.format(','.join(GRAPHS.keys()), np_imgs_bgr.shape))
  dct_times = {}
  for graph_name, resize in GRAPHS.items():
    try:
      model, ort_sess, inputs, outputs = load_onnx_model(log, 'tf_' + graph_name)
      log.p('Benchmarking {}'.format(graph_name))
      preds, lst_time = benchmark_onnx_model(
        log=log, 
        ort_sess=ort_sess, 
        input_name=inputs[0],
        n_warmup=n_warmup, 
        n_iters=n_iters,
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=batch_size, 
        as_rgb=True,
        resize=resize
        )
      dct_times[graph_name] = lst_time
      del model
      del ort_sess
      log.clear_gpu_memory()
    except Exception as e:
      log.p('Exception on {}: {}'.format(graph_name, str(e)))
      dct_times[graph_name] = [None] * get_nr_batches(np_imgs_bgr, batch_size)
  #endfor
  
  save_benchmark_results(
    log=log, 
    dct_times=dct_times, 
    batch_size=batch_size,
    fn=ct.VAPOR_GRAPHS_ONNX
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
  benchmark_vapor_graphs_onnx(
    log=log,
    np_imgs_bgr=np_imgs_bgr, 
    batch_size=BS, 
    n_warmup=N_WP, 
    n_iters=N_IT
    )
