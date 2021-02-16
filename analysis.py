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

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

def load_results(log, batch_size, benchmark_name, prepend_benchmark_name=False):
  path = get_path_results(
    log=log, 
    batch_size=batch_size, 
    fn=benchmark_name
    )
  # path = path.replace('Linux', '_Linux')
  last_results = sorted(os.listdir(path), reverse=True)[0]
  df = pd.read_csv(os.path.join(path, last_results))
  
  # df.columns[df.isna().any()].tolist()
  dct_mean = df.mean().to_dict()
  if prepend_benchmark_name:
    dct = {}
    for k, _ in dct_mean.items():
      dct[benchmark_name + '_' + k] = dct_mean[k]
    dct_mean = dct
  return dct_mean, df
  

def plot_benchmark_results(lst_batch_size, lst_benchmark_name, lst_filter=None,
                           display_names=False):
  from itertools import cycle
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  cycol = cycle('bgrcmk')
  
  lst_colors = []
  for nr, benchmark_name in enumerate(lst_benchmark_name):
    color = next(cycol)
    lst_colors.append(color)
    for bs in lst_batch_size:      
      dct_mean, _ = load_results(
        log=log,
        batch_size=bs,
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
      x = [bs] * len(dct_mean)
      y = list(dct_mean.values())
      names = list(dct_mean.keys())
      plt.plot(x, y, linestyle='-', marker='o', c=color)
      for i in range(len(x)):
        # ax.scatter(x[i], y[i], c=color)
        if display_names:
          ax.annotate(names[i], (x[i], y[i]))
      # endfor
  #endfor
  
  ax.legend(lst_benchmark_name, loc="upper right", title='Models', labelcolor=lst_colors)
  ax.set_xticks(lst_batch_size)
  ax.set_xlabel('Batch size')
  ax.set_ylabel('Mean inference time per image')
  ax.set_title('Inference time')
  plt.show()
  return

def ppplot(lst_batch_size, lst_benchmark_name, lst_filter=None,
           display_names=False, prepend_benchmark_name=False):
  from itertools import cycle
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  cycol = cycle('bgrcmykw')
  
  all_dicts = {}
  lst_colors = []
  lst_legend = []
  for nr, benchmark_name in enumerate(lst_benchmark_name):
    dct_vals = {}
    for bs in lst_batch_size:      
      dct_mean, _ = load_results(
        log=log,
        batch_size=bs,
        benchmark_name=benchmark_name,
        prepend_benchmark_name=prepend_benchmark_name
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
      
      for k,v in dct_mean.items():
        if k not in dct_vals:
          dct_vals[k] = []
        dct_vals[k].append(v)
    #endfor
    all_dicts.update(dct_vals)
    
    x = lst_batch_size
    for k,v in dct_vals.items():
      color = next(cycol)
      lst_colors.append(color)
      lst_legend.append(k)
      plt.plot(x, v, linestyle='-', marker='o', c=color)
    #endfor
  #endfor
  
  ax.legend(lst_legend, loc="upper right", title='Models')
  ax.set_xticks(lst_batch_size)
  ax.set_xlabel('Batch size')
  ax.set_ylabel('Mean inference time per image')
  ax.set_title('Inference time')
  plt.show()
  
  dct = {'BS': lst_batch_size}
  dct.update(all_dicts)
  df = pd.DataFrame(dct)
  log.p('\n\n{}'.format(df))
  return df

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  BS = list([1] + list(range(2, 31, 2)))
  # dct = ppplot(
  #   lst_batch_size=BS, 
  #   lst_benchmark_name=[
  #     ct.KERAS,
  #     ct.KERAS_ONNX,
  #     ct.KERAS_PB,
  #     ct.KERAS_TRT,
  #     ct.PYTORCH,
  #     ct.PYTORCH_ONNX,
  #     ct.PYTORCH_TRT
  #     ],
  #   lst_filter=['resnet50']
  #   )

  ppplot(
    lst_batch_size=BS, 
    lst_benchmark_name=[
      ct.VAPOR_GRAPHS_PB,
      ct.VAPOR_GRAPHS,
      ct.VAPOR_GRAPHS_ONNX,
      ct.VAPOR_GRAPHS_TRT
      ],
    lst_filter=[]
    )
  
  dct = ppplot(
    lst_batch_size=BS, 
    lst_benchmark_name=[
      ct.AUTOML_EFFDET_PB,
      ct.AUTOML_EFFDET_TRT,
      ct.PYTORCH_EFFDET
      ],
    lst_filter=['d7'],
    prepend_benchmark_name=True
    )
  
  dct = ppplot(
    lst_batch_size=BS, 
    lst_benchmark_name=[
      ct.PYTORCH_HUB,
      ct.PYTORCH_HUB_ONNX,
      
      ],
    lst_filter=['5x'],
    prepend_benchmark_name=True
    )