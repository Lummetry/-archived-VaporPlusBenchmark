from .pytorch_applications_benchmark import benchmark_pytorch_models
#tf23
try:
  from .pytorch_applications_benchmark_trt import benchmark_pytorch_models_trt
except:
  pass
from .pytorch_applications_benchmark_onnx import benchmark_pytorch_models_onnx

#tf23py36
try:  
  from .pytorch_effdet_benchmark import benchmark_pytorch_effdet_models
except:
  pass

from .pytorch_hub_benchmark import benchmark_pytorch_hub_models
from .pytorch_hub_benchmark_onnx import benchmark_pytorch_hub_models_onnx
