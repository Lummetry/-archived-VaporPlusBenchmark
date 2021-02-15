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
import numpy as np
import pandas as pd
import constants as ct
import matplotlib.pyplot as plt

from libraries import Logger
from data import get_path_results

def get_cmap(n, name='hsv'):
  '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
  RGB color; the keyword argument name must be a standard mpl colormap name.'''
  return plt.cm.get_cmap(name, n)

def load_results(log, batch_size, benchmark_name):
  path = get_path_results(
    log=log, 
    batch_size=batch_size, 
    fn=benchmark_name
    )
  # path = path.replace('Linux', '_Linux')
  last_results = sorted(os.listdir(path))[0]
  df = pd.read_csv(os.path.join(path, last_results))
  
  # df.columns[df.isna().any()].tolist()
  dct_mean = df.mean().to_dict()
  for k, _ in dct_mean.items():
    dct_mean[benchmark_name + '_' + k] = dct_mean.pop(k)
  return dct_mean, df
  

def plot_benchmark_results(batch_size, lst_benchmark_name, lst_filter):
  from itertools import cycle
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  cycol = cycle('bgrcmk')
  
  for nr, benchmark_name in enumerate(lst_benchmark_name):
    dct_mean, _ = load_results(
      log=log,
      batch_size=BS,
      benchmark_name=benchmark_name
      )
    if lst_filter:
      dct = {}
      for k,v in dct_mean.items():
        found = False
        for f in lst_filter:
          if f in k:
            found = True
            break
        if found:
          dct[k] = v
      dct_mean = dct
    #endfilter
    
    x = [BS] * len(dct_mean)
    y = list(dct_mean.values())
    names = list(dct_mean.keys())
    for i in range(len(x)):
      ax.scatter(x[i], y[i], c=next(cycol))
      ax.annotate(names[i], (x[i], y[i]))
    #endfor
  #endfor
  ax.set_xlabel('Batch size')
  ax.set_ylabel('Mean inference time')
  ax.set_title('Mean inference time per image')
  plt.show()
  return

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  BS = 1
  plot_benchmark_results(
    batch_size=BS, 
    lst_benchmark_name=[
      ct.KERAS,
      ct.KERAS_ONNX,
      ct.KERAS_PB,
      ct.KERAS_TRT,
      ct.PYTORCH,
      ct.PYTORCH_ONNX,
      ct.PYTORCH_TRT
      ],
    lst_filter=[]
    )

  plot_benchmark_results(
    batch_size=BS, 
    lst_benchmark_name=[
      ct.VAPOR_GRAPHS_PB,
      ct.VAPOR_GRAPHS,
      ct.VAPOR_GRAPHS_ONNX,
      ct.VAPOR_GRAPHS_TRT
      ],
    lst_filter=[]
    )