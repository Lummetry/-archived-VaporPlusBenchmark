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
import cv2
import torch as th
import torchvision.models as models

from lumm_pytorch import utils
from lumm_pytorch.pytorch_applications import get_th_model, MODELS


if __name__ == '__main__':
  path = 'D:/test'
  movie_name = '1_coada de persoane.mp4'
  path_movie = os.path.join(path, movie_name)
  cap = cv2.VideoCapture(path_movie)
  i = 0
  while(True):
      # Capture frame-by-frame
      ret, frame = cap.read()
      i+=1 
  
      cv2.imwrite(os.path.join(path, '{}.png'.format(i)), frame)
      # Display the resulting frame
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  
  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()
  
  
  
  
  
  