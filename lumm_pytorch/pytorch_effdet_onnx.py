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

"""
Remember to install timm and omegaconf in your environment
"""

import os
import cv2
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'pytorch', 'efficientdet'))
import pandas as pd

import torch as th
import numpy as np

from time import time
from lumm_pytorch import utils
from data_gen import data_generator
from effdet import create_model
from libraries import Logger
from effdet.efficientdet import HeadNet
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict

DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

def get_net(model_name):
  config = get_efficientdet_config(model_name)
  net = EfficientDet(config)
  model = DetBenchPredict(net)
  model.eval().to(DEVICE)
  return model

def to_onnx_model(log, model_name):
  log.p('Converting {} to onnx'.format(model_name))  
  input_shape = (1, 3, 1080, 1920) #random shape, will be override
  
  log.p('Getting model')
  model = get_net(model_name)
  log.p('Done', show_time=True)
  
  log.p('Converting...')
  onnx_model_name = model_name + '_dynamic' + '.onnx'
  utils.create_onnx_model(
    log=log,
    model=model,
    input_shape=input_shape,
    file_name=onnx_model_name,
    use_dynamic_axes=True
    )    
  
  log.p('Done', show_time=True)
  return

def load_onnx_model(log, model_name):
  log.p('Loading onnx model {}...'.format(model_name))
  model, ort_sess = utils.load_onnx_model(
    log=log,
    model_name=model_name, 
    full_path=False
    )
  log.p('Done', show_time=True)
  return model, ort_sess


if __name__ == '__main__':
  log = Logger(lib_name='VPBMRK', config_file='config.txt', max_lines=1000)
  log.set_nice_prints(df_precision=5)
  
  """
  EXPORT DOESN'T WORK
  """
  
  model_name = 'tf_efficientdet_d0'
  to_onnx_model(log, model_name)
    

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  