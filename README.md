# VaporPlusBenchmark

Packages:

1. ONNX
pip install keras2onnx
pip install tf2onnx

2. TRT

Problems:

1. Tensorflow
1.1 ONNX
1.1.1 VaporPlus
EFFDET0 conversion errs: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
1.1.2 AUTOML EFFDET
EFFDET conversion errs: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
Requires tf24 environment to run keras model
1.1.3 PYTORCH_EFFDET
Could not convert to ONNX: RuntimeError: Sizes of tensors must match except in dimension 2. Got 135 and 136 (The offending index is 0)
1.1.4 PYTORCH_HUB:
RuntimeError: Sizes of tensors must match except in dimension 2. Got 135 and 136 (The offending index is 0)


1.2 TRT
1.2.1 Keras applications
Could not convert any default keras applycations models into TRT with tensorflow-gpu==2.1.0
1.2.2 AutoML
Could not convert any effdet to TRT ValueError: NodeDef mentions attr 'explicit_paddings' not in Op<name=MaxPool; signature=input:T -> output:T; attr=T:type,default=DT_FLOAT,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16, DT_QINT8]; attr=ksize:list(int),min=4; attr=strides:list(int),min=4; attr=padding:string,allowed=["SAME", "VALID"]; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW", "NCHW_VECT_C"]>; NodeDef: {{node resample_p6/max_pooling2d/MaxPool}}. (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).
The conversion was made with native automl python conversion script into UBUNTU 18.04
1.2.3 PYTORCH_EFFDET
Could not convert to TRT: RuntimeError: Sizes of tensors must match except in dimension 2. Got 135 and 136 (The offending index is 0)
1.2.4 PYTORCH_HUB - YOLOS v3/v5
RuntimeError: Sizes of tensors must match except in dimension 2. Got 135 and 136 (The offending index is 0)


ENVIRONMENTS USED:
1. tf23py36:
conda create -n tf23py36 python=3.6 anaconda opencv tensorflow-gpu=2.1.0
pip install tensorflow-gpu==2.3.0
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

cd third_party/pytorch/torch2trt
python setup.py install

BENCHMARKS:
PYTORCH_TRT
AUTOML_EFFDET_TRT

2. tf23: ---will have python=3.7.9
conda create -n tf23 anaconda tensorflow-gpu=2.1.0
pip install tensorflow-gpu==2.3.0
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge onnx
pip install onnxruntime-gpu
conda install -c conda-forge timm

BENCHMARKS:
PYTORCH
PYTORCH_ONNX
PYTORCH_EFFDET
PYTORCH_HUB
PYTORCH_HUB_ONNX

3. tf24
conda create -n tf24 anaconda
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install tensorflow-gpu==2.4.1
cd third_party/lumm_tensorflow
pip install -r automl_effdet_requirements.txt

BENCHMARKS:
AUTOML_EFFDET


