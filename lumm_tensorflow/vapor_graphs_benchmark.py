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
import constants as ct
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf_runoptions = tf.RunOptions(report_tensor_allocations_upon_oom=True)

from libraries import Logger
from libraries.vis import VisEngine
from benchmark_methods import benchmark_vapor_graph
from data import get_nr_batches, get_path_results, read_images, save_benchmark_results, save_quality_results
from lumm_tensorflow.vapor_graphs import GRAPHS, get_vapor_graph

"""
This script tests vapor_graphs inference times (postprocess step included)
"""

BENCHMARK_NAME = ct.VAPOR_GRAPHS
FOLDER_GENERAL = ct.DATA_FOLDER_GENERAL
FOLDER_DETECTION = ct.DATA_FOLDER_DETECTION

def qualitative_vapor_graphs(log, vis_eng, lst_paths, np_imgs_bgr, filter_type={}, debug=False):
  log.p('Qualitative benchmarking VaporGraphs {} on image tensor: {}'.format(','.join(GRAPHS.keys()), np_imgs_bgr.shape))
  for model_name, _ in GRAPHS.items():
    try:
      graph = get_vapor_graph(log, model_name, 1)
      log.p('Quality checking {}'.format(model_name))
      lst_preds, lst_time = benchmark_vapor_graph(
        log=log,
        n_warmup=0, 
        n_iters=1,
        graph=graph, 
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=1, 
        as_rgb=True
        )
      del graph
      log.clear_gpu_memory()
      
      if filter_type:
        for dct_pred in lst_preds:
          lst_preds = dct_pred['INFERENCES'][0]
          lst = []
          for k,v in filter_type:
            for x in lst_preds:
              if x['TYPE'] == k and x['PROB_PRC'] >= v:
                lst.append(x)
        lst_preds = lst
      #endfilter
      
      dct_detections = {'IMGS': [], 'NR': []}
      for i,dct_pred in enumerate(lst_preds):
        img_name = os.path.split(lst_paths[i])[1]
        lst_preds = dct_pred['INFERENCES'][0]
        if filter_type:
          lst = []
          for k,v in filter_type:
            for x in lst_preds:
              if x['TYPE'] == k and x['PROB_PRC'] >= v:
                lst.append(x)
          lst_preds = lst
        #endif
        img = np_imgs_bgr[i]
        dct_detections['IMGS'].append(img_name)
        dct_detections['NR'].append(len(lst_preds))
        img = vis_eng.draw_inferences(img, lst_preds, lst_color=[(255, 255, 255)] * len(lst_preds))
        path_dst = get_path_results(log, batch_size=1, fn=os.path.join(
          BENCHMARK_NAME, 
          FOLDER_DETECTION, 
          model_name))
        os.makedirs(path_dst, exist_ok=True)
        path_dst = os.path.join(path_dst, img_name)
        cv2.imwrite(path_dst, img)
      #endfor
      save_quality_results(
        log=log, 
        dct_detections=dct_detections, 
        batch_size=1,
        fn=os.path.join(BENCHMARK_NAME, FOLDER_DETECTION, model_name)
        )
    except Exception as e:
      log.p('Exception on {}: {}'.format(model_name, str(e)))
      if debug:
        raise e
  #endfor
  return

def benchmark_vapor_graphs(log, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Benchmarking VaporGraphs {} on image tensor: {}'.format(','.join(GRAPHS.keys()), np_imgs_bgr.shape))
  dct_times = {}
  for model_name, _ in GRAPHS.items():
    try:
      graph = get_vapor_graph(log, model_name, batch_size)
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
      del graph
      log.clear_gpu_memory()
    except Exception as e:
      log.p('Exception on {}: {}'.format(model_name, str(e)))
      dct_times[model_name] = [None] * get_nr_batches(np_imgs_bgr, batch_size)
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
  
  vis = VisEngine(log=log).get_eng()
  
  BS = 1
  N_WP = 1
  N_IT = 1
  
  lst_paths, np_imgs_bgr = read_images(log=log, folder=FOLDER_GENERAL)
  benchmark_vapor_graphs(
    log=log,
    np_imgs_bgr=np_imgs_bgr, 
    batch_size=BS, 
    n_warmup=N_WP, 
    n_iters=N_IT
    )
  
  lst_paths, np_imgs_bgr = read_images(log=log, folder=FOLDER_DETECTION)
  qualitative_vapor_graphs(
    log=log,
    vis_eng=vis,
    lst_paths=lst_paths,
    np_imgs_bgr=np_imgs_bgr,
    debug=True
    )
  
  
  
  
  
  
  
  
  
  
  
  
  
  