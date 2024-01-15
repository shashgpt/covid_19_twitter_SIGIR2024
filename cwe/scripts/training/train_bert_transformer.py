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

class BERTweet_transformer(Model):
    def __init__(self, config, embed_dim, num_heads, rate=0.1, **kwargs):
        super().__init__()
        self.config = config
        if self.config["model_name"] == "bert_tweet":
            self.bert_encoder = TFAutoModel.from_pretrained("vinai/bertweet-covid19-base-cased")
            for layer in self.bert_encoder.layers:
                layer.trainable = config["fine_tune_word_embeddings"]

        self.att = layers.MultiHeadAttention(num_heads=num_heads, 
                                             key_dim=embed_dim)
        ff_dim = num_heads*4
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

        self.global_average_pooling_1d = layers.GlobalAveragePooling1D()
        self.dense = tf.keras.layers.Dense(20, activation='relu')
        self.output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    
    def compute_attention_masks(self, input_ids):
        zero = tf.constant(0, dtype=tf.int64)
        attention_masks = tf.cast(tf.not_equal(input_ids, zero), dtype=tf.int64)
        return attention_masks
    
    def call(self, input_ids, training, attention_masks=None, **kwargs):
        
        #input
        input_ids = tf.cast(input_ids, dtype=tf.int64)

        #create attention masks
        if attention_masks == None:
            attention_masks = self.compute_attention_masks(input_ids)

        #bert_tweet output
        word_embedding = self.bert_encoder(input_ids, attention_masks)[0]
       
        #transformer block
        attn_output = self.att(word_embedding, word_embedding)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(word_embedding + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        layernorm2 = self.layernorm2(out1 + ffn_output)
        pooled_output = self.global_average_pooling_1d(layernorm2)
        droput = self.dropout2(pooled_output)
        dense = self.dense(droput)
        x = self.dropout2(dense)
        output = self.output(x)
        return output

class train_bertweet_transformer(object):
    def __init__(self, config):
        self.config = config

    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset using bert tokenizer
        """
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

    def pad(self, sentences, maxlen):
        """
        right pad sequence with 0 till max token length sentence
        """
        return tf.keras.utils.pad_sequences(sentences, value=0, padding='post', maxlen=maxlen)

    def prediction(self, text):
        x = self.vectorize(text)
        x = self.pad(x, self.maxlen)
        pred_prob_1 = self.model.predict(x, batch_size=1000)
        pred_prob_0 = 1 - pred_prob_1
        prob = np.concatenate((pred_prob_0, pred_prob_1), axis=1)
        return prob
    
    def train_bert_tweet(self, train_dataset, val_datasets, test_datasets):

        #make paths
        if not os.path.exists("assets/training_history/"):
            os.makedirs("assets/training_history/")
        
        # Create train, val, and test datasets
        train_sentences = self.vectorize(train_dataset["sentence"])
        train_sentiment_labels = np.array(train_dataset["sentiment_label"])
        val_sentences = self.vectorize(val_datasets["val_dataset"]["sentence"])
        val_sentiment_labels = np.array(val_datasets["val_dataset"]["sentiment_label"])
        test_sentences = self.vectorize(test_datasets["test_dataset"]["sentence"])
        test_sentiment_labels = np.array(test_datasets["test_dataset"]["sentiment_label"])
        maxlen = max([train_sentences.shape[1], val_sentences.shape[1], test_sentences.shape[1]])
        self.maxlen = maxlen
        train_sentences = self.pad(train_sentences, maxlen)
        val_sentences = self.pad(val_sentences, maxlen)
        test_sentences = self.pad(test_sentences, maxlen)
        train_dataset = (train_sentences, train_sentiment_labels)
        val_dataset = (val_sentences, val_sentiment_labels)
        test_dataset = (test_sentences, test_sentiment_labels)

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
        my_callbacks = [
                        # early_stopping_callback, 
                        AdditionalValidationSets(additional_validation_datasets, self.config)
                        ]
        
        #model compilation and summarization
        model = BERTweet_transformer(
                                    self.config,
                                    embed_dim, 
                                    num_heads
                                    )
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
        