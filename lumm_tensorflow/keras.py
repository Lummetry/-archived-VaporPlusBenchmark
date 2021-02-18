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
from tensorflow.keras.applications import resnet, mobilenet_v2, inception_v3

MOBILENET_V2 = 'mobilenetv2'
INCEPTION_V3 = 'inceptionv3'
RESNET50    = 'resnet50'

MODELS = {
  MOBILENET_V2: {'RESIZE': (224, 224), 'PREPROCESS':  mobilenet_v2.preprocess_input},
  INCEPTION_V3: {'RESIZE': (299, 299), 'PREPROCESS': inception_v3.preprocess_input},
  RESNET50: {'RESIZE': (224, 224), 'PREPROCESS': resnet.preprocess_input}
  }

def get_keras_model(model_name):
  if model_name == MOBILENET_V2:
    model = tf.keras.applications.MobileNetV2()
  elif model_name == INCEPTION_V3:
    model = tf.keras.applications.InceptionV3()
  elif model_name == RESNET50:
    model = tf.keras.applications.ResNet50()
  return model
