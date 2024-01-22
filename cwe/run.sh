#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=UNLIMITED

#When PROTOTYPING interactively
asset="bertweet_cnn_prototype"
model_name="bertweet_cnn"
# timestamp="$(date +"%T")"
# mkdir -p "assets/logs"
# cd "assets/logs"
# rm "$asset$timestamp.out"
# exec 3>&1 4>&2
# trap 'exec 2>&4 1>&3' 0 1 2 3
# exec 1>$asset$timestamp.out 2>&1
# cd $OLDPWD
source /opt/anaconda2/bin/activate env_python_3.9_tensorflow
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
python3 main.py \
--asset_name $asset \
--model_name $model_name \
--seed_value 11 \
--dataset_name "covid19-twitter" \
--dataset_path "datasets/covid19-twitter/raw_dataset.pickle" \
--fine_tune_word_embeddings "True" \
--generate_explanation_for_one_instance "True" \
--train_model "True" \
--evaluate_model "True" \
--generate_explanation "True" \
--optimizer "adam" \
--learning_rate 1e-6 \
--mini_batch_size 50 \
--train_epochs 1 \
--dropout 0.5 \
--lime_no_of_samples 1000 \
--hidden_units 100

# #When experimenting
# asset="bertweet_cnn"
# timestamp="$(date +"%T")"
# mkdir -p "assets/logs"
# cd "assets/logs"
# rm "$asset$timestamp.out"
# exec 3>&1 4>&2
# trap 'exec 2>&4 1>&3' 0 1 2 3
# exec 1>$asset$timestamp.out 2>&1
# cd $OLDPWD
# source /opt/anaconda2/bin/activate env_python_3.9_tensorflow
# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
# python3 main.py \
# --asset_name $asset \
# --model_name "bertweet_cnn" \
# --seed_value 11 \
# --dataset_name "covid19-twitter" \
# --dataset_path "datasets/covid19-twitter/raw_dataset.pickle" \
# --fine_tune_word_embeddings "True" \
# --generate_explanation_for_one_instance "False" \
# --train_model "True" \
# --evaluate_model "True" \
# --generate_explanation "True" \
# --optimizer "adam" \
# --learning_rate 1e-6 \
# --mini_batch_size 50 \
# --train_epochs 200 \
# --dropout 0.5 \
# --lime_no_of_samples 1000 \
# --hidden_units 128