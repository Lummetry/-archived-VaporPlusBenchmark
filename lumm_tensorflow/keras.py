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
import tensorflow as tf

MOBILENET_V2 = 'mobilenetv2'
INCEPTION_V3 = 'inceptionv3'
RESNET50    = 'resnet50'

MODELS = {
  MOBILENET_V2: (224, 224),
  INCEPTION_V3: (299, 299),
  RESNET50: (224, 224)
  }

def get_keras_model(model_name):
  if model_name == MOBILENET_V2:
    model = tf.keras.applications.MobileNetV2()
  elif model_name == INCEPTION_V3:
    model = tf.keras.applications.InceptionV3()
  elif model_name == RESNET50:
    model = tf.keras.applications.ResNet50()
  return model
