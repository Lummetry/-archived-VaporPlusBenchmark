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
Cloned: https://github.com/NVIDIA-AI-IOT/torch2trt
cd .../VaporPlusBenchmark/third_party/pytorch/torch2trt
python setup.py install

Test by importing: 
import tensorrt as trt
from torch2trt import tensorrt_converter

"""

import os
import torch as th

from torch2trt import TRTModule
from torch2trt import torch2trt
from libraries import Logger
from lumm_pytorch.pytorch_applications import get_th_model, MODELS

if __name__ == '__main__':  
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt', 
    TF_KERAS=False
    )
  
  for model_name, size in MODELS.items():
    log.p('Converting {} into TRT model'.format(model_name))
    
    #create model
    model = get_th_model(model_name)
  
    # create example data
    x = th.ones((1, 3, size[0], size[1])).cuda()
    
    # convert to TensorRT feeding sample data as input
    model_trt = torch2trt(model, [x])
    
    path_model = os.path.join(log.get_models_folder(), 'th_{}_trt.pth'.format(model_name))
    th.save(model_trt.state_dict(), path_model)
    
    #load_model
    model_trt = TRTModule()
    model_trt.load_state_dict(th.load(path_model))
  #endfor
  
  
  
  