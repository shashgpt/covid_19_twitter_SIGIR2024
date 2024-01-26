import argparse
import pprint
import os
import logging
import warnings
import random
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import argparse
import subprocess as sp
import distutils
import pprint

#Change the code execution directory to current directory
os.chdir(os.getcwd())

#scripts
from scripts.dataset_processing.preprocess_dataset import Preprocess_dataset
from scripts.dataset_processing.word_vectors import Word_vectors
from scripts.dataset_processing.dataset_division import Dataset_division

from scripts.training.train_bertweet_mlp import train_bertweet_mlp
from scripts.training.train_bertweet_transformer import train_bertweet_transformer
from scripts.training.train_bertweet_cnn import train_bertweet_cnn
from scripts.training.train_bertweet_rnn import train_bertweet_rnn
from scripts.training.train_gpt2_mlp import train_gpt2_mlp
from scripts.training.train_gpt2_transformer import train_gpt2_transformer
from scripts.training.train_gpt2_rnn import train_gpt2_rnn
from scripts.training.train_elmo_mlp import train_elmo_mlp
from scripts.training.train_gpt2_cnn import train_gpt2_cnn

#disable warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

#set the gpu device with highest free memory
def mask_unused_gpus(leave_unmasked=1): # No of avaialbe GPUs on the system
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        available_gpus = [i for i, x in enumerate(memory_free_values)]
        if len(available_gpus) < leave_unmasked: raise ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
        gpu_with_highest_free_memory = 0
        highest_free_memory = 0
        for index, memory in enumerate(memory_free_values):
            if memory > highest_free_memory:
                highest_free_memory = memory
                gpu_with_highest_free_memory = index
        return str(gpu_with_highest_free_memory)
    except Exception as e:
        print('"nvidia-smi" is probably not installed. GPUs are not masked', e)
os.environ["CUDA_VISIBLE_DEVICES"] = mask_unused_gpus()

if __name__=='__main__':

    #gather parser arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_name",
                        type=str,
                        required=True)
    parser.add_argument("--model_name",
                        type=str,
                        required=True)
    parser.add_argument("--seed_value",
                        type=int,
                        required=True)
    parser.add_argument("--dataset_name",
                        type=str,
                        required=True)
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True)
    parser.add_argument("--fine_tune_word_embeddings",
                        type=lambda x:bool(distutils.util.strtobool(x)),
                        required=True) 
    parser.add_argument("--optimizer",
                        type=str,
                        required=True)
    parser.add_argument("--learning_rate",
                        type=float,
                        required=True)
    parser.add_argument("--mini_batch_size",
                        type=int,
                        required=True)
    parser.add_argument("--train_epochs",
                        type=int,
                        required=True)
    parser.add_argument("--dropout",
                        type=float,
                        required=True)
    parser.add_argument("--lime_no_of_samples",
                        type=int,
                        required=True)
    parser.add_argument("--generate_explanation_for_one_instance",
                        type=lambda x:bool(distutils.util.strtobool(x)),
                        required=True)
    parser.add_argument("--train_model",
                        type=lambda x:bool(distutils.util.strtobool(x)),
                        required=True)
    parser.add_argument("--evaluate_model",
                        type=lambda x:bool(distutils.util.strtobool(x)),
                        required=True)
    parser.add_argument("--generate_explanation",
                        type=lambda x:bool(distutils.util.strtobool(x)),
                        required=True)
    parser.add_argument("--hidden_units",
                        type=int,
                        required=True)
    args = parser.parse_args()
    config = vars(args)
    print("\n")
    pprint.pprint(config)
    print("\n")

    #set seed value
    os.environ['PYTHONHASHSEED']=str(config["seed_value"])
    random.seed(config["seed_value"])
    np.random.seed(config["seed_value"])
    # tf.random.set_seed(config["seed_value"])

    print("\nCreating input data")
    if os.path.exists("datasets/"+config["dataset_name"]+"/preprocessed_dataset.pickle"):
        with open("datasets/"+config["dataset_name"]+"/word_index.pickle", "rb") as handle:
            word_index = pickle.load(handle)
        with open("datasets/"+config["dataset_name"]+"/word_vectors.npy", "rb") as handle:
            word_vectors = np.load(handle)
        train_dataset = pickle.load(open("datasets/"+config["dataset_name"]+"/train_dataset.pickle", "rb"))
        val_datasets = pickle.load(open("datasets/"+config["dataset_name"]+"/val_dataset.pickle", "rb"))
        test_datasets = pickle.load(open("datasets/"+config["dataset_name"]+"/test_dataset.pickle", "rb"))
    else:
        raw_dataset = pickle.load(open(config["dataset_path"], "rb"))
        raw_dataset = pd.DataFrame(raw_dataset)
        preprocessed_dataset = Preprocess_dataset(config).preprocess_covid_tweets(raw_dataset)
        preprocessed_dataset = pd.DataFrame(preprocessed_dataset)
        word_vectors, word_index = Word_vectors(config).create_word_vectors(preprocessed_dataset)
        train_dataset, val_datasets, test_datasets = Dataset_division(config).train_val_test_split(preprocessed_dataset)
    
    #create model
    print("\nBuilding model")
    if config["model_name"] == "bertweet_mlp":
        train_bertweet_mlp(config).train_model(train_dataset, val_datasets, test_datasets)
    elif config["model_name"] == "bertweet_transformer":
        train_bertweet_transformer(config).train_model(train_dataset, val_datasets, test_datasets)
    elif config["model_name"] == "bertweet_cnn":
        train_bertweet_cnn(config).train_model(train_dataset, val_datasets, test_datasets)
    elif config["model_name"] in ["bertweet_lstm", "bertweet_bilstm", "bertweet_gru", "bertweet_bigru"]:
        train_bertweet_rnn(config).train_model(train_dataset, val_datasets, test_datasets)
    elif config["model_name"] == "elmo_mlp":
        train_elmo_mlp(config).train_model(train_dataset, val_datasets, test_datasets)
    elif config["model_name"] == "gpt2_mlp":
        train_gpt2_mlp(config).train_model(train_dataset, val_datasets, test_datasets)
    elif config["model_name"] == "gpt2_transformer":
        train_gpt2_transformer(config).train_model(train_dataset, val_datasets, test_datasets)
    elif config["model_name"] in ["gpt2_lstm", "gpt2_bilstm", "gpt2_gru", "gpt2_bigru"]:
        train_gpt2_rnn(config).train_model(train_dataset, val_datasets, test_datasets)
    elif config["model_name"] == "gpt2_cnn":
        train_gpt2_cnn(config).train_model(train_dataset, val_datasets, test_datasets)