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

from lumm_pytorch import benchmark_pytorch_effdet_models, \
  benchmark_pytorch_models, benchmark_pytorch_models_onnx,  \
  benchmark_pytorch_hub_models, benchmark_pytorch_hub_models_onnx
  
from lumm_tensorflow import benchmark_automl_effdet_pb, benchmark_automl_effdet_trt, \
  benchmark_keras_models, benchmark_keras_models_onnx, benchmark_keras_models_pb, benchmark_keras_models_trt, \
  benchmark_vapor_graphs, benchmark_vapor_graphs_pb, benchmark_vapor_graphs_onnx, benchmark_vapor_graphs_trt

BENCHMARKS_TF23 = {
    ct.PYTORCH:           benchmark_pytorch_models,
    ct.PYTORCH_ONNX:      benchmark_pytorch_models_onnx,
    
    ct.PYTORCH_EFFDET:    benchmark_pytorch_effdet_models,
    
    ct.PYTORCH_HUB:       benchmark_pytorch_hub_models,
    ct.PYTORCH_HUB_ONNX:  benchmark_pytorch_hub_models_onnx,
    
    ct.AUTOML_EFFDET_PB:  benchmark_automl_effdet_pb,
    ct.AUTOML_EFFDET_TRT: benchmark_automl_effdet_trt,
    
    ct.KERAS:             benchmark_keras_models,
    ct.KERAS_PB:          benchmark_keras_models_pb,
    ct.KERAS_TRT:         benchmark_keras_models_trt,
    ct.KERAS_ONNX:        benchmark_keras_models_onnx,
    
    ct.VAPOR_GRAPHS:      benchmark_vapor_graphs,
    ct.VAPOR_GRAPHS_PB:   benchmark_vapor_graphs_pb,
    ct.VAPOR_GRAPHS_TRT:  benchmark_vapor_graphs_trt,
    ct.VAPOR_GRAPHS_ONNX: benchmark_vapor_graphs_onnx
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
    for key, benchmark_method in BENCHMARKS_TF23.items():
      benchmark_method(
        log=log,
        np_imgs_bgr=np_imgs_bgr, 
        batch_size=bs, 
        n_warmup=N_WP, 
        n_iters=N_IT
        )