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
from data_gen import data_generator
from effdet import create_model
from libraries import Logger

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
  ).to(DEVICE)
  model.eval()
  return model

def benchmark_th_effdet_model(model, np_imgs_bgr, batch_size, n_warmup, n_iters,
                              as_rgb=False, resize=None):
  def _predict(np_batch, resize):
    if resize:
      np_batch = np.array([cv2.resize(x, resize) for x in np_batch])
    if as_rgb:
      np_batch = np_batch[:,:,:,::-1]
    np_batch = np.transpose(np_batch, (0, 3, 1, 2)).astype('float32')
    with th.no_grad():
      th_x = th.from_numpy(np_batch).to(DEVICE)
      output = model(th_x).cpu().numpy()
      results = []
      for index, sample in enumerate(output):
        image_id = i
        sample[:, 2] -= sample[:, 0]
        sample[:, 3] -= sample[:, 1]
        for det in sample:
          score = float(det[4])
          if score < .001:  # stop when below this threshold, scores in descending order
              break
          coco_det = dict(
              image_id=image_id,
              bbox=det[0:4].tolist(),
              score=score,
              category_id=int(det[5]))
          results.append(coco_det)
      #endfor
    #end with
    return

  input_size = model.config.image_size
  param_count = sum([m.numel() for m in model.parameters()])
  # print('Model %s created, param count: %d' % (model_name, param_count))
  
  #warmup
  for i in range(n_warmup):
    log.p(' Warmup {}'.format(i))
    gen = data_generator(np_imgs=np_imgs_bgr, batch_size=batch_size)
    for np_batch in gen:
      _predict(np_batch, tuple(input_size))
  
  #iters
  for i in range(n_iters):
    log.p(' Iter {}'.format(i))
    gen = data_generator(np_imgs=np_imgs_bgr, batch_size=batch_size)
    lst_time = []
    for np_batch in gen:
      start = time()
      preds = _predict(np_batch, tuple(input_size))
      stop = time()
      lst_time.append(stop - start)
  #endfor
  return preds, lst_time

if __name__ == '__main__':
  log = Logger(lib_name='VPBMRK', config_file='config.txt', max_lines=1000)
  log.set_nice_prints(df_precision=5)
  
  BATCH_SIZE = 1
  NR_WARMUP = 1
  NR_ITERS = 10
  
  path_images = os.path.join(log.get_data_subfolder('General'))
  lst_imgs = [os.path.join(path_images, x) for x in os.listdir(path_images)]
  lst_imgs = [cv2.imread(x) for x in lst_imgs]
  np_imgs = np.array(lst_imgs)
  
  dct_times = {}
  for dct_info in THH_EFFDET_MODEL_LIST:
    model_name = dct_info['model_name']
    model = get_th_effdet_model(model_name)
    log.p('Benchmarking {}'.format(model_name))
    preds, lst_time = benchmark_th_effdet_model(
      model=model, 
      np_imgs_bgr=np_imgs, 
      batch_size=BATCH_SIZE, 
      n_warmup=NR_WARMUP, 
      n_iters=NR_ITERS,
      as_rgb=True
      )
    dct_times[model_name] = lst_time
  #endfor
  
  df = pd.DataFrame(dct_times)
  log.p('\n\n{}'.format(df))
  platform, system = log.get_platform()
  log.save_dataframe(
    df=df,
    fn='{}_{}_{}.csv'.format(platform, 'pytorch_effdet', log.now_str()),
    folder='output'
    )
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  