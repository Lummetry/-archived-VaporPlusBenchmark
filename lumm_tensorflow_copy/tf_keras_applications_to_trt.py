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
https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
The user guide specifies that: Note: Converting frozen graphs is no longer supported in TensorFlow 2.0. 
"""

import os
import cv2
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np

from tqdm import tqdm
from libraries import Logger
from lumm_tensorflow.tf_keras_applications import get_tf_keras_application_model, MODELS
from lumm_tensorflow.utils import get_trt_graph

if __name__ == '__main__':
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  img = cv2.imread(os.path.join(log.get_data_folder(), 'images/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png'))
  img = log.center_image2(img, 224, 224)[:,:,::-1]
  np_imgs = np.expand_dims(img, axis=0)
  
  for graph_name in MODELS.keys():
    model = get_tf_keras_application_model(graph_name.replace('.pb', ''))
    outs_keras = model.predict(np_imgs)
  
    trt_graph, sess, tf_inp, tf_out = get_trt_graph(log, graph_name + '.pb')
    
    for i in tqdm(range(1)):
      outs_rt = sess.run(
        fetches=tf_out,
        feed_dict={tf_inp: np_imgs}
        )
      # print(outs_rt)
    #endfor
  #endfor
    
    
    