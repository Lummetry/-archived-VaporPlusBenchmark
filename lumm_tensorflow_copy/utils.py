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
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf_runoptions = tf.RunOptions(report_tensor_allocations_upon_oom=True)

from time import time
from data_gen import data_generator


def get_pb(log, model_name):
  tf_graph, s_input, s_output = log.load_graph_from_models(
    model_name='{}.pb'.format(model_name),
    get_input_output=True
    )
  sess = tf.Session(graph=tf_graph)
  tf_inp = tf_graph.get_tensor_by_name(s_input)
  tf_out = tf_graph.get_tensor_by_name(s_output)
  return tf_graph, sess, tf_inp, tf_out

####ONNX
def graph_to_onnx(log, 
                  graph_name, 
                  onnx_name, 
                  inputs, 
                  outputs, 
                  opset=11, 
                  folder='models'
                  ):
  import subprocess
  assert folder in [None, 'data', 'output', 'models']
    
  if not onnx_name.endswith('.onnx'):
    onnx_name+= '.onnx'  
  
  lfld = log.get_target_folder(target=folder)
  if folder is not None:
    full_path_graph = os.path.join(lfld, graph_name)
    full_path_onnx = os.path.join(lfld, onnx_name)
  else:
    full_path_graph = graph_name
    full_path_onnx = onnx_name
    
  assert os.path.isfile(full_path_graph), 'Path does not exist: {}'.format(full_path_graph)
  
  log.p('Saving graph to onnx model to: {}'.format(full_path_onnx))
  proc = subprocess.run(
    'python -m tf2onnx.convert --graphdef {} ' \
    '--output {} --opset {} ' \
    '--inputs {} --outputs {}'.format(
      full_path_graph, 
      full_path_onnx, 
      opset,
      inputs,
      outputs
      ).split(),
    capture_output=True)
  log.p(proc.returncode)
  log.p(proc.stdout.decode('ascii'))
  log.p(proc.stderr.decode('ascii'))
  
  log.p('Done saving onnx model')
  return

def session_graph_to_onnx(log, 
                          graph, 
                          input_names, 
                          output_names, 
                          onnx_name, 
                          opset=None,
                          folder='models'):
  import tf2onnx 
  assert folder in [None, 'data', 'output', 'models']
    
  if not onnx_name.endswith('.onnx'):
    onnx_name+= '.onnx'  
  
  lfld = log.get_target_folder(target=folder)
  
  if folder is not None:
    full_path_onnx = os.path.join(lfld, onnx_name)
  else:
    full_path_onnx = onnx_name
    
  onnx_graph = tf2onnx.tfonnx.process_tf_graph(
    tf_graph=graph, 
    input_names=input_names, 
    output_names=output_names,
    opset=opset
    )
  model_proto = onnx_graph.make_model('onnx_model')
  with open(full_path_onnx, 'wb') as f:
    f.write(model_proto.SerializeToString())
  
  log.p('Done saving onnx model')
  return

def keras_to_onnx(log, model, onnx_name, folder='models'):
  import keras2onnx as k2o
  assert folder in [None, 'data', 'output', 'models']
  
  onnx_model = k2o.convert_keras(
    model=model, 
    name=onnx_name
    )
  
  if not onnx_name.endswith('.onnx'):
    onnx_name+= '.onnx'
  
  lfld = log.get_target_folder(target=folder)

  if folder is not None:
    path = os.path.join(lfld, onnx_name)
  else:
    path = onnx_name
    
  log.p('Saving keras model to onnx model at: {}'.format(path))
  k2o.save_model(onnx_model, path)
  log.p('Done saving onnx model')
  return

def load_onnx_model(log, model_name, full_path=False):
  import onnx
  import onnxruntime as ort
  
  if not model_name.endswith('.onnx'):
    model_name+= '.onnx'
  
  if not full_path:
    model_full_path = os.path.join(log.get_models_folder(), model_name)
  else:
    model_full_path = model_name
  
  if not os.path.isfile(model_full_path):
    log.p('Provided path does not exists {}'.format(model_full_path))
    return
  
  log.p('Loading onnx model from: {}'.format(model_full_path))
  model = onnx.load(model_full_path)
  onnx.checker.check_model(model)
  onnx.helper.printable_graph(model.graph)
  
  ort_session = ort.InferenceSession(model_full_path)
  
  output_names = [node.name for node in model.graph.output]
  
  input_all = [node.name for node in model.graph.input]
  input_initializer =  [node.name for node in model.graph.initializer]
  input_names = list(set(input_all)  - set(input_initializer))
  
  return model, ort_session, input_names, output_names


###########TRT
def graph_to_trt(log, graph_name, 
                 input_name=None, 
                 output_name=None, 
                 folder='models',
                 save=True):
  from tensorflow.python.compiler.tensorrt import trt_convert as trt
  assert folder in [None, 'models']
  if folder:
    path_graph = log.get_models_file(graph_name)
    _, input_name, output_name = log.load_graph_from_models(graph_name, get_input_output=True)
    input_name = input_name.split(':')[0]
    output_name = output_name.split(':')[0]
  else:
    path_graph = graph_name
    assert input_name and output_name, 'Please provide input and output graph names'
  
  assert os.path.isfile(path_graph), 'Graph not found in {}'.format(path_graph)
    
  with tf.Session() as sess:
    # First deserialize your frozen graph:
    with tf.gfile.GFile(path_graph, 'rb') as f:
      frozen_graph = tf.GraphDef()
      frozen_graph.ParseFromString(f.read())
    #endwith
    
    # Now you can create a TensorRT inference graph from your frozen graph:
    converter = trt.TrtGraphConverter(
      input_graph_def=frozen_graph,
      nodes_blacklist=output_name
      ) #output nodes
    trt_graph = converter.convert()
    
    if save:
      graph_name = graph_name + '.trt'
      # path_save = os.path.join(log.get_models_folder(), graph_name)
      # converter.save(path_save)
      
      input_tensor_names = input_name
      output_tensor_names = output_name
      log.save_graph_to_file(
        sess=sess, 
        graph=trt_graph,
        pb_file=graph_name,
        input_tensor_names=input_tensor_names,
        output_tensor_names=output_tensor_names
        )
  return trt_graph

def load_trt_graph(log, graph_file, folder='models'):
  import tensorflow.compat.v1 as tf
  from tensorflow.python.compiler.tensorrt import trt_convert as trt
  assert folder in [None, 'models']
  
  if not graph_file.endswith('.trt'):
    graph_file+= '.trt'
  if folder is None:
    path = graph_file
  else:
    path = os.path.join(log.get_models_folder(), graph_file)
    
  if os.path.exists(path):
    cfg = log.load_models_json(graph_file + '.txt')
    input_name = cfg['INPUT_0']
    output_name = cfg['OUTPUT_0']
    trt_graph = log.load_tf_graph(path, return_elements=[output_name])
    return trt_graph, input_name, output_name
  else:
    log.p('TRT graph not found on location: {}'.format(path))
  return

def get_trt_graph(log, graph_name):
  if not graph_name.endswith('.trt'):
    graph_name+= '.trt'
    
  if log.get_models_file(graph_name) is None:
    graph_to_trt(log, graph_name=graph_name.replace('.trt', ''))
  
  trt_graph, inp_name, out_name = load_trt_graph(log, graph_name)
  
  sess = tf.Session(graph=trt_graph)
  tf_inp = trt_graph.get_tensor_by_name(inp_name)
  tf_out = trt_graph.get_tensor_by_name(out_name)
  return trt_graph, sess, tf_inp, tf_out


def benchmark_onnx_model(log, 
                         ort_sess, 
                         input_name,
                         np_imgs_bgr, 
                         batch_size, 
                         n_warmup, 
                         n_iters,
                         as_rgb=False, 
                         channels_first=False,
                         resize=None
                         ):
  def _predict(np_batch):
    if resize:
      np_batch = np.array([cv2.resize(x, resize) for x in np_batch])
    if as_rgb:
      np_batch = np_batch[:,:,:,::-1]
    if channels_first:
      np_batch = np.transpose(np_batch, (0, 3, 1, 2))
    np_batch = np_batch.astype(np.float32)
    preds = ort_sess.run(
      output_names=None, 
      input_feed={input_name: np_batch}
      )
    return preds
  #enddef
  
  #warmup  
  for i in range(n_warmup):
    log.p(' Warmup {}'.format(i))
    gen = data_generator(np_imgs=np_imgs_bgr, batch_size=batch_size)
    for np_batch in gen:
      _predict(np_batch)
  
  #iters
  for i in range(n_iters):
    log.p(' Iter {}'.format(i))
    gen = data_generator(np_imgs=np_imgs_bgr, batch_size=batch_size)
    lst_time = []
    for np_batch in gen:
      start = time()
      preds = _predict(np_batch)
      stop = time()
      lst_time.append(stop - start)
  #endfor
  return preds, lst_time


def benchmark_tf_graph(log, sess, tf_inp, tf_out, 
                           np_imgs_bgr, batch_size, n_warmup, n_iters,
                           as_rgb=False, resize=None):
  def _predict(np_batch):
    if resize:
      np_batch = np.array([cv2.resize(x, resize) for x in np_batch])
    if as_rgb:
      np_batch = np_batch[:,:,:,::-1]
    out_scores = sess.run(
      tf_out,
      feed_dict={tf_inp: np_batch},
      options=tf_runoptions
      )
    return out_scores
  
  #warmup
  for i in range(n_warmup):
    log.p(' Warmup {}'.format(i))
    gen = data_generator(np_imgs=np_imgs_bgr, batch_size=batch_size)
    for np_batch in gen:
      _predict(np_batch)
  
  #iters
  for i in range(n_iters):
    log.p(' Iter {}'.format(i))
    gen = data_generator(np_imgs=np_imgs_bgr, batch_size=batch_size)
    lst_time = []
    for np_batch in gen:
      start = time()
      out_scores = _predict(np_batch)
      stop = time()
      lst_time.append(stop - start)
  #endfor
  return out_scores, lst_time
