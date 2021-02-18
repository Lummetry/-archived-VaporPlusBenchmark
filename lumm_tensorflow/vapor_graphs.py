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
tf.disable_eager_execution()
tf_runoptions = tf.RunOptions(report_tensor_allocations_upon_oom=True)

from vapor_inference.graphs import YoloInferenceGraph, EfficientDet0InferenceGraph

TF_YOLO   = 'TF_YOLO'
EFF_DET0  = 'EFF_DET0'

GRAPHS = {
  TF_YOLO:  (608, 608), 
  EFF_DET0: (574, 1020)
  }

BATCH_SIZES = list([1] + list(range(2, 31, 2)))

def get_vapor_graph(log, graph_name, batch_size=1):
  path_config = 'vapor_inference/inference.txt'
  if graph_name == TF_YOLO:
    graph = YoloInferenceGraph(log=log, config_path=path_config)
  elif graph_name == EFF_DET0:
    cfg = log.load_json('vapor_inference/inference.txt')
    cfg_effdet = cfg['EFF_DET0']
    path = cfg_effdet['MODELS_PATH']
    path_pb = path.format(batch_size)
    cfg_effdet['GRAPH'] = path_pb
    cfg_effdet['BATCH_SIZE'] = batch_size
    graph = EfficientDet0InferenceGraph(log=log, config_graph=cfg_effdet)
  return graph

if __name__ == '__main__':
  from libraries import Logger
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  # for batch_size in BATCH_SIZES:
  #   get_vapor_graph(log=log, graph_name=EFF_DET0, batch_size=batch_size)
  
  graph = get_vapor_graph(
    log=log, 
    graph_name=EFF_DET0, 
    batch_size=18
    )
  
  import constants as ct
  from data import read_images
  np_imgs_bgr = read_images(log=log, folder=ct.DATA_FOLDER_GENERAL)
  
  graph.predict(np_imgs_bgr)
  
  graph = get_vapor_graph(
    log=log,
    graph_name=TF_YOLO,
    )
  graph.predict(np_imgs_bgr)
  
  
  
  
  
  
  
  
  
  
  
  
  
  