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
import tensorflow.compat.v1 as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
tf.disable_eager_execution()
import constants as ct

from libraries import Logger
from data import read_images


from lumm_pytorch import benchmark_pytorch_models_trt
from lumm_tensorflow import benchmark_automl_effdet_trt

BENCHMARKS_TF23PY36 = {
    ct.PYTORCH_TRT:       benchmark_pytorch_models_trt,
    # ct.AUTOML_EFFDET_TRT: benchmark_automl_effdet_trt
  }

BATCH_SIZES = list([1] + list(range(2, 31, 2)))
# BATCH_SIZES = [1]

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  N_WP = 1
  N_IT = 2
  
  np_imgs_bgr = read_images(log=log, folder=ct.DATA_FOLDER_GENERAL)
  for bs in BATCH_SIZES:
    for key, benchmark_method in BENCHMARKS_TF23PY36.items():
      benchmark_method(
        log=log,
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=bs, 
        n_warmup=N_WP, 
        n_iters=N_IT
        )