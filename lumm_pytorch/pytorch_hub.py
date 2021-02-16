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

from lumm_pytorch import utils

THH_YOLOV3 = ('ultralytics/yolov3', 'yolov3')
THH_YOLOV3SPP = ('ultralytics/yolov3', 'yolov3_spp')
THH_YOLOV3TINY = ('ultralytics/yolov3', 'yolov3_tiny')

# THH_YOLOV5 = ('ultralytics/yolov5', 'yolov5')
THH_YOLOV5S = ('ultralytics/yolov5', 'yolov5s')
THH_YOLOV5M = ('ultralytics/yolov5', 'yolov5m')
THH_YOLOV5L = ('ultralytics/yolov5', 'yolov5l')
THH_YOLOV5X = ('ultralytics/yolov5', 'yolov5x')

MODELS = [THH_YOLOV3, THH_YOLOV3SPP, THH_YOLOV3TINY, 
          THH_YOLOV5S, THH_YOLOV5M, THH_YOLOV5L, THH_YOLOV5X]

DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

def get_pytorchhub_model(log, repo_or_dir, model_name):
  model = th.hub.load(
    repo_or_dir=repo_or_dir, 
    model=model_name, 
    pretrained=True,
    # force_reload=True
    ).to(DEVICE)
  return model


def load_onnx_model(log, model_name):
  log.p('Loading onnx model {}...'.format(model_name))
  model, ort_sess = utils.load_onnx_model(
    log=log,
    model_name=model_name, 
    full_path=False
    )
  log.p('Done', show_time=True)
  return model, ort_sess

