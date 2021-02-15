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
import torch as th

def export_onnx_model(model, input_shape, onnx_path, input_names=None, 
                      output_names=None, dynamic_axes=None, opset_version=None):
  device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
  th_inputs = th.zeros(*input_shape).to(device)
  th.onnx.export(
    model=model, 
    args=th_inputs, 
    f=onnx_path, 
    input_names=input_names, 
    output_names=output_names, 
    dynamic_axes=dynamic_axes,
    opset_version=opset_version,
    verbose=True
    )
  return


def load_onnx_model(log, model_name, full_path=False):
  import onnx
  import onnxruntime as ort
  
  if not full_path:
    model_full_path = os.path.join(log.get_models_folder(), model_name)
  else:
    model_full_path = model_name
  
  if not os.path.isfile(model_full_path):
    log.p('Provided path does not exists {}'.format(model_full_path))
    return
  
  model = onnx.load(model_full_path)
  onnx.checker.check_model(model)
  onnx.helper.printable_graph(model.graph)
  
  ort_session = ort.InferenceSession(model_full_path)
  return model, ort_session

def create_onnx_model(log, model, input_shape, file_name, 
                      input_names=None, output_names=None,
                      use_dynamic_axes=False, use_dynamic_batch_size=False,
                      opset_version=11
                      ):
  if not input_names:
    input_names = ['input']
  if not output_names:
    output_names = ['output']
    
  onnx_path = file_name
  if not onnx_path.endswith('.onnx'):
    onnx_path+= '.onnx'
  
  if use_dynamic_axes:
    dynamic_axes= {
      'input' : {0: 'batch_size', 2: 'height', 3: 'width'}, 
      'output': {0: 'batch_size', 2: 'height', 3: 'width'}
      } #adding names for better debugging
  elif use_dynamic_batch_size:
    dynamic_axes={
      'input' : {0 : 'batch_size'},    
      'output' : {0 : 'batch_size'}
      }
  else:
    dynamic_axes = None
  
  save_path = os.path.join(log.get_models_folder(), file_name)
  log.p('Saving model to: {}'.format(save_path[-50:]))
  export_onnx_model(
    model=model, 
    input_shape=input_shape, 
    onnx_path=save_path, 
    input_names=input_names, 
    output_names=output_names, 
    dynamic_axes=dynamic_axes,
    opset_version=opset_version
    )
  return