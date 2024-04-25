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

from scripts.training.train_ikd_cnn import train_ikd_cnn

#disable warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

#set the gpu execution device with highest free memory
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
        preprocessed_dataset = Preprocess_dataset(config).preprocess_dataset()
        preprocessed_dataset = pd.DataFrame(preprocessed_dataset)
        word_vectors, word_index = Word_vectors(config).create_word_vectors(preprocessed_dataset)
        train_dataset, val_datasets, test_datasets = Dataset_division(config).train_val_test_split(preprocessed_dataset)
    
    #create model
    print("\nBuilding model")
    if config["model_name"] == "ikd_cnn":
        train_ikd_cnn(config).train_model(train_dataset, val_datasets, test_datasets, word_index, word_vectors)