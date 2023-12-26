import os
import shutil
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from transformers import TFAutoModel

from scripts.additional_validation_sets import AdditionalValidationSets

class bert_tweet(Model):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        if self.config["model_name"] == "bert_tweet":
            self.bert_encoder = TFAutoModel.from_pretrained("vinai/bertweet-covid19-base-cased")
            for layer in self.bert_encoder.layers:
                layer.trainable = config["fine_tune_word_embeddings"]
        self.out = layers.Dense(1, activation='sigmoid', name='output')
    
    def compute_attention_masks(self, input_ids):
        zero = tf.constant(0, dtype=tf.int64)
        attention_masks = tf.cast(tf.not_equal(input_ids, zero), dtype=tf.int64)
        return attention_masks
    
    def call(self, input_ids, attention_masks=None, **kwargs):
        
        #input
        input_ids = tf.cast(input_ids, dtype=tf.int64)

        #create attention masks
        if attention_masks == None:
            attention_masks = self.compute_attention_masks(input_ids)

        #bert_tweet output
        sentence_embedding = self.bert_encoder(input_ids, attention_masks)[1]
       
        #output
        out = self.out(sentence_embedding)

        return out

class train(object):
    def __init__(self, config):
        self.config = config

    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset using bert tokenizer
        """
        if self.config["model_name"] == "bert_tweet":
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-covid19-base-cased", use_fast=False)
        max_len = 0
        input_ids = []
        for sentence in sentences:
            tokenized_context = tokenizer.encode(sentence)
            input_id = tokenized_context
            input_ids.append(input_id)
            if len(input_id) > max_len:
                max_len = len(input_id)
        for index, input_id in enumerate(input_ids):
            padding_length = max_len - len(input_ids[index])
            input_ids[index] = input_ids[index] + ([0] * padding_length)
        return np.array(input_ids)
    
    def train_bert_tweet(self, train_dataset, val_datasets, test_datasets):
        
        model = bert_tweet(self.config)

        #make paths
        if not os.path.exists("assets/training_history/"):
            os.makedirs("assets/training_history/")
        
        #create train dataset
        train_sentences = self.vectorize(train_dataset["sentence"])
        train_sentiment_labels = np.array(train_dataset["sentiment_label"])

        #create validation dataset
        val_sentences = self.vectorize(val_datasets["val_dataset"]["sentence"])
        val_sentiment_labels = np.array(val_datasets["val_dataset"]["sentiment_label"])

        train_dataset = (train_sentences, train_sentiment_labels)
        val_dataset = (val_sentences, val_sentiment_labels)

        #create additional validation datasets
        additional_validation_datasets = []
        for key, value in test_datasets.items():
            # if key in ["test_dataset_one_rule"]:
            #     continue
            sentences = self.vectorize(test_datasets[key]["sentence"])
            sentiment_labels = np.array(test_datasets[key]["sentiment_label"])
            dataset = (sentences, sentiment_labels, key)
            additional_validation_datasets.append(dataset)

        #define callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # 1. Calculate val_loss_1 
                                                        min_delta = 0,                  # 2. Check val_losses for next 10 epochs 
                                                        patience=10,                    # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                        verbose=0,                      # 4. Get the trained weights corresponding to val_loss_1
                                                        mode="min",
                                                        baseline=None, 
                                                        restore_best_weights=True)
        my_callbacks = [early_stopping_callback, AdditionalValidationSets(additional_validation_datasets, self.config)]
        
        #model compilation and summarization
        model.compile(tf.keras.optimizers.legacy.Adam(
                learning_rate=self.config["learning_rate"]), 
                loss=['binary_crossentropy'], 
                metrics=['accuracy'])
        model.build(input_shape = train_dataset[0].shape)
        model.summary(line_length=150)

        #train the model
        model.fit(x=train_dataset[0], 
                y=train_dataset[1], 
                epochs=self.config["train_epochs"], 
                batch_size=self.config["mini_batch_size"], 
                validation_data=val_dataset, 
                callbacks=my_callbacks,
                shuffle=False)

        #save trained weights of the model
        if not os.path.exists("assets/trained_models/"):
            os.makedirs("assets/trained_models/")
        model.save_weights("assets/trained_models/"+self.config["asset_name"]+".h5")
        