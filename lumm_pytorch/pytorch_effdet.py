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
Remember to install timm and timm, omegaconf in your environment
conda install -c conda-forge timm

"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'pytorch', 'efficientdet'))
import torch as th

from effdet import create_model

DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

def _entry(model_name, paper_model_name, paper_arxiv_id, batch_size, model_desc=None):
  return dict(
    model_name=model_name,
    model_description=model_desc,
    paper_model_name=paper_model_name,
    paper_arxiv_id=paper_arxiv_id,
    batch_size=batch_size
    )

THH_EFFDET_MODEL_LIST = [
    ## Weights trained by myself or others in PyTorch
    _entry('tf_efficientdet_lite0', 'EfficientDet-Lite0', '1911.09070', batch_size=128,
           model_desc='Trained in PyTorch with https://github.com/rwightman/efficientdet-pytorch'),
    _entry('efficientdet_d0', 'EfficientDet-D0', '1911.09070', batch_size=112,
           model_desc='Trained in PyTorch with https://github.com/rwightman/efficientdet-pytorch'),
    _entry('efficientdet_d1', 'EfficientDet-D1', '1911.09070', batch_size=72,
           model_desc='Trained in PyTorch with https://github.com/rwightman/efficientdet-pytorch'),

    ## Weights ported by myself from other frameworks
    _entry('tf_efficientdet_d0', 'EfficientDet-D0', '1911.09070', batch_size=112,
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d1', 'EfficientDet-D1', '1911.09070', batch_size=72,
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d2', 'EfficientDet-D2', '1911.09070', batch_size=48,
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d3', 'EfficientDet-D3', '1911.09070', batch_size=32,
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d4', 'EfficientDet-D4', '1911.09070', batch_size=16,
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d5', 'EfficientDet-D5', '1911.09070', batch_size=12,
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d6', 'EfficientDet-D6', '1911.09070', batch_size=8,
           model_desc='Ported from official Google AI Tensorflow weights'),
    _entry('tf_efficientdet_d7', 'EfficientDet-D7', '1911.09070', batch_size=4,
           model_desc='Ported from official Google AI Tensorflow weights')
]

def get_th_effdet_model(model_name):
  model = create_model(
      model_name,
      bench_task='predict',
      pretrained=True,
  ).eval().to(DEVICE)
  return model

  
  
  
  

  
  
  
  
  
  
  
  
  