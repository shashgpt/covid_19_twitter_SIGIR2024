#!/bin/bash

source /opt/anaconda2/bin/activate env_python_3.9_tensorflow
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

python3 main.py \
--asset_name $1 \
--folder_number $2
