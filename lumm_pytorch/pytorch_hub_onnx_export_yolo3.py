"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov3.pt --img 640 --batch 1
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'pytorch', 'yolov3'))
import time
import argparse

import torch as th
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from libraries import Logger
from lumm_pytorch.utils import create_onnx_model

DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

if __name__ == '__main__':  
  log = Logger(
    lib_name='BENCHMARK', 
    config_file='config.txt', 
    TF_KERAS=False
    )
  
  # weights = 'yolov3.pt'
  # weights = 'yolov3-spp.pt'
  # weights = 'yolov3-tiny.pt'
  img_size = [640, 640]
  batch_size = 1
  set_logging()
  t = time.time()

  # Load PyTorch model
  model = attempt_load(weights, map_location=DEVICE)  # load FP32 model
  labels = model.names

  # Checks
  gs = int(max(model.stride))  # grid size (max stride)
  img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples

  # Input
  input_shape = (batch_size, 3, *img_size)
  img = th.zeros(*input_shape).to(DEVICE)

  # Update model
  for k, m in model.named_modules():
      m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
      if isinstance(m, models.common.Conv):  # assign export-friendly activations
          if isinstance(m.act, nn.Hardswish):
              m.act = Hardswish()
          elif isinstance(m.act, nn.SiLU):
              m.act = SiLU()
      # elif isinstance(m, models.yolo.Detect):
      #     m.forward = m.forward_export  # assign forward (optional)
  model.model[-1].export = True  # set Detect() layer export=True
  y = model(img)  # dry run
  
  if False:
    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % th.__version__)
        f = weights.replace('.pt', '.torchscript.pt')
        ts = th.jit.trace(model, img)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

  # ONNX export
  try:
      import onnx

      print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
      f = weights.replace('.pt', '.onnx')
      f = f.replace('-', '_')
      
      create_onnx_model(
        log=log,
        model=model,
        input_shape=input_shape,
        file_name=f,
        use_dynamic_axes=True,
        output_names=['classes', 'boxes'] if y is None else ['output']
        )
      
      # Checks
      f = os.path.join(log.get_models_folder(), f)
      onnx_model = onnx.load(f)  # load onnx model
      onnx.checker.check_model(onnx_model)  # check onnx model
      # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
      print('ONNX export success, saved as %s' % f)
  except Exception as e:
      print('ONNX export failure: %s' % e)
  
  if False:
    # CoreML export
    try:
        import coremltools as ct
  
        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
        print('CoreML export success, saved as %s' % f)
    except Exception as e:
        print('CoreML export failure: %s' % e)

  # Finish
  print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
