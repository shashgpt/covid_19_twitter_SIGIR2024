from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input, Softmax, Lambda
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.initializers import Constant
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle
from lime import lime_text
import traceback
from tqdm import tqdm

from scripts.additional_validation_sets import AdditionalValidationSets

def cnn(config, word_vectors): # Similar to one used in pytorch code for CIKM submission

    input_sentence = Input(shape=(None,), dtype="int64")
    word_embeddings = layers.Embedding(word_vectors.shape[0], 
                                        word_vectors.shape[1], 
                                        embeddings_initializer=Constant(word_vectors), 
                                        trainable=config["fine_tune_word_embeddings"], 
                                        mask_zero=True, 
                                        name="word_embeddings")(input_sentence)

    word_embeddings_reshaped = tf.keras.backend.expand_dims(word_embeddings, axis=1) # batch_size x 1 x sent_len x embedding_dim

    conv_1 = layers.Conv2D(filters = 100, 
                            kernel_size = (3, 300),
                            strides = 1,
                            dilation_rate = 1,
                            padding = "valid",
                            data_format = "channels_first",
                            name = "conv2D_1")(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
    conv1_reshaped = tf.keras.backend.squeeze(conv_1, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
    conv1_reshaped_relu = layers.ReLU()(conv1_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
    max_pool_1 = layers.GlobalMaxPooling1D(data_format="channels_first",
                                            name="maxpool1D_1")(conv1_reshaped_relu) # batch size x n_filters

    conv_2 = layers.Conv2D(filters = 100, 
                            kernel_size = (4, 300),
                            strides = 1,
                            dilation_rate = 1,
                            padding = "valid",
                            data_format = "channels_first",
                            name = "conv2D_2")(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
    conv2_reshaped = tf.keras.backend.squeeze(conv_2, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
    conv2_reshaped_relu = layers.ReLU()(conv2_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
    max_pool_2 = layers.GlobalMaxPooling1D(data_format="channels_first",
                                            name="maxpool1D_2")(conv2_reshaped_relu) # batch size x n_filters

    conv_3 = layers.Conv2D(filters = 100, 
                            kernel_size = (5, 300),
                            strides = 1,
                            dilation_rate = 1,
                            padding = "valid",
                            data_format = "channels_first",
                            name = "conv2D_3")(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
    conv3_reshaped = tf.keras.backend.squeeze(conv_3, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
    conv3_reshaped_relu = layers.ReLU()(conv3_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
    max_pool_3 = layers.GlobalMaxPooling1D(data_format="channels_first",
                                            name="maxpool1D_3")(conv3_reshaped_relu) # batch size x n_filters

    concat = layers.Concatenate(axis=1, name="concatenate")([max_pool_1, max_pool_2, max_pool_3])
    concat_dropout = layers.Dropout(rate=config["dropout"], seed=config["seed_value"], name="dropout")(concat)       
    out = layers.Dense(1, activation='sigmoid', name='output')(concat_dropout)

    model = Model(inputs=[input_sentence], outputs=[out])

    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    elif config["optimizer"] == "adadelta":
        model.compile(tf.keras.optimizers.Adadelta(learning_rate=config["learning_rate"], rho=0.95, epsilon=1e-06), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

class train_cnn(object):
    def __init__(self, config):
        self.config = config
        self.vectorize_layer = None
        self.maxlen = None
        self.model = None
    
    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        return self.vectorize_layer(np.array(sentences)).numpy()
    
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

    def train_model(self, train_dataset, val_datasets, test_datasets, word_index, word_vectors):

        # Make paths
        if not os.path.exists("assets/training_history/"):
            os.makedirs("assets/training_history/")
        
        #create vocab and define the vectorize layer
        vocab = [key for key in word_index.keys()]
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, 
                                                                                            split='whitespace', 
                                                                                            vocabulary=vocab)

        # Create Train and Val datasets
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

        # Create additional validation datasets
        additional_validation_datasets = []
        for key, value in test_datasets.items():
            # if key in ["test_dataset_one_rule"]:
            #     continue
            sentences = self.vectorize(test_datasets[key]["sentence"])
            sentences = self.pad(sentences, maxlen)
            sentiment_labels = np.array(test_datasets[key]["sentiment_label"])
            dataset = (sentences, sentiment_labels, key)
            additional_validation_datasets.append(dataset)

        # Define callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # 1. Calculate val_loss_1 
                                                        min_delta = 0,                  # 2. Check val_losses for next 10 epochs 
                                                        patience=10,                    # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                        verbose=0,                      # 4. Get the trained weights corresponding to val_loss_1
                                                        mode="min",
                                                        baseline=None, 
                                                        restore_best_weights=True)
        my_callbacks = [early_stopping_callback, 
                        AdditionalValidationSets(additional_validation_datasets, self.config)]

        #model compilation and summarization
        model = cnn(self.config, word_vectors)
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=self.config["learning_rate"]), 
                                                        loss=['binary_crossentropy'], 
                                                        metrics=['accuracy'])    
        model.summary()
        self.model = model

        # Train the model
        if self.config["train_model"] == True:
            model.fit(x=train_dataset[0], 
                    y=train_dataset[1], 
                    batch_size=self.config["mini_batch_size"], 
                    epochs=self.config["train_epochs"], 
                    validation_data=val_dataset, 
                    callbacks=my_callbacks)

            # Save trained model
            if not os.path.exists("assets/trained_models/"):
                os.makedirs("assets/trained_models/")
            model.save_weights("assets/trained_models/"+self.config["asset_name"]+".h5")
        
        if self.config["evaluate_model"] == True:

            #load model
            self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

            # Results to be created after evaluation
            results = {'sentence':[], 
                        'sentiment_label':[],  
                        'rule_label':[],
                        'contrast':[],
                        'sentiment_probability_output':[], 
                        'sentiment_prediction_output':[]}

            # Evaluation and predictions
            evaluations = self.model.evaluate(x=test_dataset[0], y=test_dataset[1])
            print("test loss, test acc:", evaluations)
            predictions = self.model.predict(x=test_dataset[0])

            for index, sentence in enumerate(test_datasets["test_dataset"]["sentence"]):
                results['sentence'].append(test_datasets["test_dataset"]['sentence'][index])
                results['sentiment_label'].append(test_datasets["test_dataset"]['sentiment_label'][index])
                results['rule_label'].append(test_datasets["test_dataset"]['rule_label'][index])
                results['contrast'].append(test_datasets["test_dataset"]['contrast'][index])
            for prediction in predictions:
                results['sentiment_probability_output'].append(prediction)
                prediction = np.rint(prediction)
                results['sentiment_prediction_output'].append(prediction[0])

            # Save the results
            if not os.path.exists("assets/results/"):
                os.makedirs("assets/results/")
            with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
                pickle.dump(results, handle)
        
        if self.config["generate_explanation"] == True:
            print("\nLIME explanations")

            #Load trained model
            self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

            #Results to be created after explanation
            explanations = {"sentence":[], 
                            "LIME_explanation":[], 
                            "LIME_explanation_normalised":[]}

            with open("assets/results/"+self.config["asset_name"]+".pickle", 'rb') as handle:
                results = pickle.load(handle)
            results = pd.DataFrame(results)

            test_sentences = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentence']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentence'])
            probabilities = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_probability_output']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_probability_output'])

            explainer = lime_text.LimeTextExplainer(class_names=["negative_sentiment", "positive_sentiment"], 
                                                    split_expression=" ", 
                                                    random_state=self.config["seed_value"])

            for index, test_datapoint in enumerate(tqdm(test_sentences)):
                probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                tokenized_sentence = test_datapoint.split()
                try:
                    exp = explainer.explain_instance(test_datapoint, self.prediction, num_features = len(tokenized_sentence), num_samples=self.config["lime_no_of_samples"])
                except:
                    traceback.print_exc()
                    text = test_datapoint
                    explanation = "couldn't process"
                    explanations["sentence"].append(text)
                    explanations["LIME_explanation"].append(explanation)
                    explanations["LIME_explanation_normalised"].append(explanation)
                    continue
                text = []
                explanation = []
                explanation_normalised = []
                for word in test_datapoint.split():
                    for weight in exp.as_list():
                        weight = list(weight)
                        if weight[0]==word:
                            text.append(word)
                            if weight[1] < 0:
                                weight_normalised_negative_class = abs(weight[1])*probability[0]
                                explanation_normalised.append(weight_normalised_negative_class)
                            elif weight[1] > 0:
                                weight_normalised_positive_class = abs(weight[1])*probability[1]
                                explanation_normalised.append(weight_normalised_positive_class)
                            explanation.append(weight[1])
                explanations['sentence'].append(text)
                explanations['LIME_explanation'].append(explanation)
                explanations['LIME_explanation_normalised'].append(explanation_normalised)

                if self.config["generate_explanation_for_one_instance"] == True:
                    print(explanation_normalised)
                    break
            
            if not os.path.exists("assets/lime_explanations/"):
                os.makedirs("assets/lime_explanations/")
            with open("assets/lime_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
                pickle.dump(explanations, handle)

        return model
