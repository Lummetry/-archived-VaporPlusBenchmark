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
import shutil
import numpy as np
import constants as ct

from libraries import Logger
from benchmark_methods import benchmark_tf_graph
from lumm_tensorflow.keras import MODELS
from lumm_tensorflow.utils import get_pb
from data import read_images, get_path_results

def qualitative_benchmark_keras_models_trt(log, lst_paths, np_imgs_bgr, batch_size, n_warmup, n_iters):
  log.p('Qualitative benchmarking KerasModelsPB {} on image tensor: {}'.format(','.join(MODELS.keys()), np_imgs_bgr.shape))
  dct_classes = log.load_json('imagenet_classes_json.txt', folder='data')
  for model_name, dct_opt in MODELS.items():
    try:
      graph, sess, tf_inp, tf_out = get_pb(log, model_name)
      log.p('Benchmarking {}'.format(model_name))
      preds, lst_time = benchmark_tf_graph(
        log=log, 
        sess=sess, 
        tf_inp=tf_inp,
        tf_out=tf_out,
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=batch_size, 
        n_warmup=n_warmup, 
        n_iters=n_iters,
        as_rgb=True,
        resize=dct_opt['RESIZE'],
        preprocess_input_fn=dct_opt['PREPROCESS'],
        )
      np_preds = np.array(preds).squeeze()
      for i,pred in enumerate(np_preds):
        path_src = lst_paths[i]
        path_dst = get_path_results(log, batch_size=1, fn=os.path.join(ct.KERAS_PB, ct.DATA_FOLDER_CLASSIFICATION, model_name))
        os.makedirs(path_dst, exist_ok=True)
        idx = np.argmax(pred, axis=-1)
        lbl = dct_classes[idx][1]
        shutil.copyfile(path_src, os.path.join(path_dst, lbl + '.png'))
      #endfor
      del graph
      del sess
      log.clear_gpu_memory()
    except Exception as e:
      log.p('Exception on {}: {}'.format(model_name, str(e)))
  #endfor
  return

#gasire landamrk
#extracgere imagine
#aplicare filtru de detectie de margine sau filtru de laplacian
#verificare distanta de fiecare data cand vii cu o imagine noua

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  BS = 1
  N_WP = 0
  N_IT = 1
  
  lst_preds = []
  lst_paths, lst_imgs = read_images(log=log, folder=ct.DATA_FOLDER_CLASSIFICATION)
  qualitative_benchmark_keras_models_trt(
    log=log,
    lst_paths=lst_paths,
    np_imgs_bgr=lst_imgs,
    batch_size=BS,
    n_warmup=N_WP,
    n_iters=N_IT
    )


  
  
  
  
  