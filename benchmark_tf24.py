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
from data import read_images

# from lumm_tensorflow import benchmark_automl_effdet_keras, 
# from lumm_pytorch import benchmark_pytorch_models_trt

from lumm_tensorflow import benchmark_automl_effdet_keras

BENCHMARKS_TF23 = {
    ct.AUTOML_EFFDET:  benchmark_automl_effdet_keras,
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
  for key, benchmark_method in BENCHMARKS_TF23.items():
    for bs in BATCH_SIZES:
      log.p('Started {} benchmark on batch_size {}'.format(key, bs))
      benchmark_method(
        log=log,
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=bs, 
        n_warmup=N_WP, 
        n_iters=N_IT
        )