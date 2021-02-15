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
Not working. RuntimeError: Sizes of tensors must match except in dimension 2. Got 135 and 136 (The offending index is 0)
"""

import torch as th

from torch2trt import torch2trt, TRTModule
from libraries import Logger
from lumm_pytorch.pytorch_hub import MODELS, get_pytorchhub_model

DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  dct_times = {}
  for repo_or_dir, model_name in MODELS:
    try:
      model = get_pytorchhub_model(log, repo_or_dir, model_name)
          
      # create example data
      x = th.ones((1, 3, 1080, 1920)).cuda()
      
      # convert to TensorRT feeding sample data as input
      model_trt = torch2trt(model, [x])
    
      th.save(model_trt.state_dict(), 'th_{}_trt.pth'.format(model_name))
      
      model_trt = TRTModule()
      
      th.load_state_dict(th.load('th_{}_trt.pth'.format(model_name)))
    except:
      pass
    