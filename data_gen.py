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

import math

def data_generator(np_imgs, batch_size):
  nr_batches = int(math.ceil(np_imgs.shape[0] / batch_size))
  for i in range(nr_batches):
    start = i * batch_size
    stop = (i + 1) * batch_size if i < nr_batches - 1 else np_imgs.shape[0]
    np_batch = np_imgs[start:stop]
    yield np_batch
  return