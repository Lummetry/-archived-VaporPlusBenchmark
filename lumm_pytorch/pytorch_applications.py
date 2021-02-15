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
  
import torch as th
import torchvision.models as models

from lumm_pytorch import utils

MOBILENET_V2 = 'mobilenetv2'
INCEPTION_V3 = 'inceptionv3'
RESNET50    = 'resnet50'

MODELS = {
  MOBILENET_V2: (224, 224),
  INCEPTION_V3: (299, 299),
  RESNET50: (299, 299)
  }


DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


def get_th_model(model_name):
  if model_name == MOBILENET_V2:
    model = models.mobilenet_v2(pretrained=True)
  elif model_name == INCEPTION_V3:
    model = models.inception_v3(pretrained=True)
  elif model_name == RESNET50:
    model = models.resnet50(pretrained=True)
  model.eval().to(DEVICE)
  return model

def to_onnx_model(log, model_name):
  assert model_name in MODELS, 'Model {} not configured'.format(model_name)
  
  log.p('Converting {} to onnx'.format(model_name))  
  resize = MODELS[model_name]
  height, width = resize
  input_shape = (1, 3, height, width) #random number for batch_size. the batch_size will be made dynamic
  
  log.p('Getting model')
  model = get_th_model(model_name)
  log.p('Done', show_time=True)
  
  model.eval()
  
  log.p('Converting...')
  onnx_model_name = model_name + '_fixed_{}'.format('x'.join(str(x) for x in resize)) + '.onnx'
  utils.create_onnx_model(
    log=log,
    model=model,
    input_shape=input_shape,
    file_name=onnx_model_name,
    use_dynamic_batch_size=True
    )    
  
  log.p('Done', show_time=True)
  return

def load_onnx_model(log, model_name):
  log.p('Loading onnx model...')
  resize = MODELS[model_name]
  height, width = resize
  onnx_model_name = model_name + '_fixed_{}'.format('x'.join(str(x) for x in resize)) + '.onnx'
  model, ort_sess = utils.load_onnx_model(
    log=log,
    model_name=onnx_model_name, 
    full_path=False
    )
  log.p('Done', show_time=True)
  return model, ort_sess



  
  
  
  
  
  
  
  
  