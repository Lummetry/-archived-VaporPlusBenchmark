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
import sys
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'pytorch', 'yolov3'))

import torch as th
import torch.nn as nn

import models
from utils.activations import Hardswish, SiLU

from lumm_pytorch import utils

THH_YOLOV3 = ('ultralytics/yolov3', 'yolov3')
THH_YOLOV3SPP = ('ultralytics/yolov3', 'yolov3_spp')
THH_YOLOV3TINY = ('ultralytics/yolov3', 'yolov3_tiny')

THH_YOLOV5 = ('ultralytics/yolov5', 'yolov5')
THH_YOLOV5S = ('ultralytics/yolov5', 'yolov5s')
THH_YOLOV5M = ('ultralytics/yolov5', 'yolov5m')
THH_YOLOV5L = ('ultralytics/yolov5', 'yolov5l')
THH_YOLOV5X = ('ultralytics/yolov5', 'yolov5x')

MODELS = [THH_YOLOV3, THH_YOLOV3SPP, THH_YOLOV3TINY, 
       THH_YOLOV5, THH_YOLOV5S, THH_YOLOV5M, THH_YOLOV5L, THH_YOLOV5X]

DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


def get_pytorchhub_model(log, repo_or_dir, model_name):
  model = th.hub.load(
    repo_or_dir=repo_or_dir, 
    model=model_name, 
    pretrained=True,
    force_reload=True
    ).to(DEVICE)
  return model

def to_onnx_model(log, repo_or_dir, model_name):
  """"
  This method doesn't work, please use the export method provided by the original github project: https://github.com/ultralytics/yolov3/blob/master/models/export.py
  """
  assert (repo_or_dir, model_name) in MODELS, 'Model {} not configured'.format(model_name)
  
  log.p('Converting {} to onnx'.format(model_name))  
  input_shape = (1, 3, 1080, 1920) #random shape, will be override
  
  log.p('Getting model')
  model = get_pytorchhub_model(log, repo_or_dir, model_name)
  log.p('Done', show_time=True)
  
  th_x = th.zeros(*input_shape)
  # Update model
  for k, m in model.named_modules():
    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    if isinstance(m, models.common.Conv):  # assign export-friendly activations
      if isinstance(m.act, nn.Hardswish):
        m.act = Hardswish()
      elif isinstance(m.act, nn.SiLU):
        m.act = SiLU()
    # model[-1].export = True  # set Detect() layer export=True
    y = model(th_x)  # dry run
  
  model.eval()
  
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