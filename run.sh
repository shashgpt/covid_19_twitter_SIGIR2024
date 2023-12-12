#!/bin/bash
#SBATCH --gres=gpu:1080:1
#SBATCH --partition=gpu-dev-1080
#SBATCH --time=UNLIMITED
#SBATCH --nodelist=thorin

#Construct covid-19 tweets dataset
# start_init=1
# process_no_init=1
# for i in {0..55} # no of CPUs
# do  
#     process_no=$(($process_no_init + $i))
#     start=$(($start_init + 5*i))
#     stop=$(($start + 4))
#     screen -S "screen"$process_no -d -m taskset --cpu-list $process_no python preprocess_dataset.py $process_no $start $stop # start the preprocessing on a particular CPU on a particular screen
# done

# name="constructing_covid-19_twitter_"
# timestamp="$(date +"%T")"
# mkdir -p "assets/logs"
# cd "assets/logs"
# rm "$name$timestamp.out"
# exec 3>&1 4>&2
# trap 'exec 2>&4 1>&3' 0 1 2 3
# exec 1>$name$timestamp.out 2>&1
# cd $OLDPWD

source /opt/anaconda2/bin/activate env_python_3.9_tensorflow
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

# screen -S "screen"$process_no -d -m taskset --cpu-list $process_no python scripts/construct_covid19_dataset.py $process_no $start $stop
python3 scripts/construct_covid19_dataset.py \
--process_no 24 \
--start 1 \
--stop 2

# #When PROTOTYPING interactively
# timestamp="$(date +"%T")"
# model_name="transformer"
# mkdir -p "assets/logs"
# cd "assets/logs"
# rm "$model_name$timestamp.out"
# exec 3>&1 4>&2
# trap 'exec 2>&4 1>&3' 0 1 2 3
# exec 1>$model_name$timestamp.out 2>&1
# cd $OLDPWD

# source /opt/anaconda2/bin/activate env_python_3.9_tensorflow
# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

# python3 main.py \
# --asset_name "transformer_prototype" \
# --model_name "transformer" \
# --seed_value 11 \
# --dataset_name "covid19-twitter" \
# --dataset_path "datasets/covid19-twitter/raw_dataset.pickle" \
# --fine_tune_word_embeddings "False" \
# --optimizer "adam" \
# --learning_rate 3e-5 \
# --mini_batch_size 50 \
# --train_epochs 1 \
# --dropout 0.4 \
# --lime_no_of_samples 1000 \
# --generate_explanation_for_one_instance "True" \
# --train_model "True" \
# --evaluate_model "True" \
# --generate_explanation "True"