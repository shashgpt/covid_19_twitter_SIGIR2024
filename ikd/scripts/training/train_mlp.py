import tensorflow as tf
import os
import numpy as np
import pandas as pd
import pickle
from lime import lime_text
import traceback
from tqdm import tqdm
import keras_tuner

from scripts.training.additional_validation_sets import AdditionalValidationSets

def mlp(config, word_vectors, maxlen, hyperparameters_tuning=None):
    
    if hyperparameters_tuning == None:
        input_sentence = tf.keras.layers.Input(shape=(maxlen,), dtype="int64")
        word_embeddings = tf.keras.layers.Embedding(word_vectors.shape[0], 
                                                    word_vectors.shape[1], 
                                                    embeddings_initializer=tf.keras.initializers.Constant(word_vectors), 
                                                    trainable=config["fine_tune_word_embeddings"], 
                                                    mask_zero=True, 
                                                    name="word_embeddings")(input_sentence)
        word_embeddings_flatten = tf.keras.layers.Flatten()(word_embeddings)
        dense = tf.keras.layers.Dense(config["hidden_units"], activation='relu')(word_embeddings_flatten)
        dense_dropout = tf.keras.layers.Dropout(config["dropout"])(dense)
        out = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dense_dropout)
        model = tf.keras.Model(inputs=[input_sentence], outputs=[out])
        return model
    else:
        input_sentence = tf.keras.layers.Input(shape=(maxlen,), dtype="int64")
        word_embeddings = tf.keras.layers.Embedding(word_vectors.shape[0], 
                                                    word_vectors.shape[1], 
                                                    embeddings_initializer=tf.keras.initializers.Constant(word_vectors), 
                                                    trainable=config["fine_tune_word_embeddings"], 
                                                    mask_zero=True, 
                                                    name="word_embeddings")(input_sentence)
        word_embeddings_flatten = tf.keras.layers.Flatten()(word_embeddings)
        dense = tf.keras.layers.Dense(hyperparameters_tuning["units"], 
                                      activation='relu')(word_embeddings_flatten)
        # if hyperparameters_tuning["dropout"] == True:
        #     dense = tf.keras.layers.Dropout(config["dropout"])(dense)
        dense = tf.keras.layers.Dropout(config["dropout"])(dense)
        out = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dense)
        model = tf.keras.Model(inputs=[input_sentence], outputs=[out])
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=hyperparameters_tuning["lr"]), 
                                                        loss=['binary_crossentropy'], 
                                                        metrics=['accuracy']) 
        return model

class train_mlp(object):
    def __init__(self, config) -> None:
        self.config = config
        self.vectorize_layer = None
        self.maxlen = None
        self.model = None
        self.word_vectors = None

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

    def build_model(self, hp):

        # units = hp.Int("units", min_value=1, max_value=129, step=8)
        units = self.config["hidden_units"]
        # lr = hp.Float("lr", min_value=1e-5, max_value=3e-2, sampling="log")
        lr = self.config["learning_rate"]
        # activation = hp.Choice("activation", ["relu", "tanh"])
        # dropout = hp.Boolean("dropout")
        dropout = hp.Float("dropout", min_value=0.1, max_value=0.9, step=0.1)

        hyperparameters_tuning = {
                                 "units":units,
                                 "lr":lr,
                                 "dropout":dropout
                                 }
        
        model = mlp(self.config, 
                    self.word_vectors, 
                    self.maxlen, 
                    hyperparameters_tuning=hyperparameters_tuning)
        model.summary()
        return model
    
    def train_model(self, train_dataset, val_datasets, test_datasets, word_index, word_vectors):

        # Make paths
        if not os.path.exists("assets/training_history/"):
            os.makedirs("assets/training_history/")
        
        #create vocab and define the vectorize layer
        vocab = [key for key in word_index.keys()]
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=vocab)

        # Create Train and Val datasets
        train_sentences = self.vectorize(train_dataset["sentence"])
        train_sentiment_labels = np.array(train_dataset["sentiment_label"])
        val_sentences = self.vectorize(val_datasets["val_dataset"]["sentence"])
        val_sentiment_labels = np.array(val_datasets["val_dataset"]["sentiment_label"])
        test_sentences = self.vectorize(test_datasets["test_dataset"]["sentence"])
        test_sentiment_labels = np.array(test_datasets["test_dataset"]["sentiment_label"])
        maxlen = max([train_sentences.shape[1], val_sentences.shape[1], test_sentences.shape[1]])
        self.maxlen = maxlen
        train_sentences = self.pad(train_sentences, self.maxlen)
        val_sentences = self.pad(val_sentences, self.maxlen)
        test_sentences = self.pad(test_sentences, self.maxlen)
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
        my_callbacks = [
                        # early_stopping_callback,
                        AdditionalValidationSets(additional_validation_datasets, self.config)
                       ]
        
        #model compilation and summarization
        model = mlp(self.config, word_vectors, maxlen)
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=self.config["learning_rate"]), 
                      loss=['binary_crossentropy'], 
                      metrics=['accuracy']) 
        model.summary()
        self.model = model

        # #Hyperparameter tuning
        # self.word_vectors = word_vectors
        # self.build_model(keras_tuner.HyperParameters())
        # tuner = keras_tuner.RandomSearch(
        #                                 hypermodel=self.build_model,
        #                                 objective="val_loss",
        #                                 max_trials=3,
        #                                 executions_per_trial=2,
        #                                 overwrite=True,
        #                                 )
        # tuner.search_space_summary()

        # Train the model
        if self.config["train_model"] == True:
            self.model.fit(x=train_dataset[0], 
                            y=train_dataset[1], 
                            batch_size=self.config["mini_batch_size"], 
                            epochs=self.config["train_epochs"], 
                            validation_data=val_dataset, 
                            callbacks=my_callbacks)
            # tuner.search(train_dataset[0], 
            #              train_dataset[1],
            #              batch_size=self.config["mini_batch_size"],
            #              epochs=self.config["train_epochs"], 
            #              validation_data=val_dataset)
            # tuner.results_summary()
            # best_hps = tuner.get_best_hyperparameters(5)
            # print("\nHyperparameter values")
            # print(best_hps[0].get("units"))
            # self.model = self.build_model(best_hps[0])
            # self.model.fit(x=train_dataset[0], 
            #                 y=train_dataset[1], 
            #                 batch_size=self.config["mini_batch_size"], 
            #                 epochs=self.config["train_epochs"], 
            #                 validation_data=val_dataset, 
            #                 callbacks=my_callbacks)

            # Save trained model
            if not os.path.exists("assets/trained_models/"):
                os.makedirs("assets/trained_models/")
            self.model.save_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

            # # Save the configurations
            # self.config["hyperparameters"] = best_hps[0]
        
        if self.config["evaluate_model"] == True:

            #load model
            # self.model = self.build_model(self.config["hyperparameters"])
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
            # self.model = self.build_model(self.config["hyperparameters"])
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
        
        if not os.path.exists("assets/configurations/"):
            os.makedirs("assets/configurations/")
        with open("assets/configurations/"+self.config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)