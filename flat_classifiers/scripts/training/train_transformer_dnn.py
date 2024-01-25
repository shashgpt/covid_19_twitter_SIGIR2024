import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import os
import pickle
import pandas as pd
import traceback
from lime import lime_text
from tqdm import tqdm

from scripts.training.additional_validation_sets import AdditionalValidationSets

class PositionalEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        # self.pos_emb = self.positional_encoding(length=maxlen, depth=embed_dim)

    # def positional_encoding(self, length, depth):
    #     depth = depth/2
    #     positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    #     depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    #     angle_rates = 1 / (10000**depths)         # (1, depth)
    #     angle_rads = positions * angle_rates      # (pos, depth)
    #     pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    #     return tf.cast(pos_encoding, dtype=tf.float32)
    
    def compute_mask(self, *args, **kwargs):
        return self.token_emb.compute_mask(*args, **kwargs)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
        # length = tf.shape(x)[1]
        # x = self.token_emb(x)
        # x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        # x = x + self.pos_emb[tf.newaxis, :length, :]
        # return x

class TransformerBlock(Model):
    def __init__(self, config, embed_dim, maxlen, num_heads, vocab_size, epsilon):
        super().__init__()
        self.config = config
        # self.input_layer = layers.Input(shape=(maxlen,), dtype="float32")
        self.positional_embedding_layer = PositionalEmbedding(maxlen, vocab_size, embed_dim)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([layers.Dense(num_heads*4, activation="relu"), 
                                        layers.Dense(embed_dim)])
        self.layernorm = layers.LayerNormalization(epsilon=epsilon)
        self.dropout = layers.Dropout(self.config["dropout"])
        self.global_average_pooling_1d = layers.GlobalAveragePooling1D()
        self.add = tf.keras.layers.Add()
        self.dense = tf.keras.layers.Dense(20, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation='sigmoid', name='output')

    def call(self, inputs, training):

        #positional embeddings
        positional_embeddings = self.positional_embedding_layer(inputs)
        positional_embeddings = self.dropout(positional_embeddings, training=training)
        positional_embeddings = self.layernorm(positional_embeddings)

        #encoder
        encoder_output = self.att(positional_embeddings, positional_embeddings)
        encoder_output = self.dropout(encoder_output, training=training)
        encoder_output = self.add([positional_embeddings, encoder_output])
        encoder_output = self.layernorm(encoder_output)
        encoder_output_ffn = self.ffn(encoder_output)
        encoder_output_ffn = self.dropout(encoder_output_ffn, training=training)
        encoder_output = self.add([encoder_output, encoder_output_ffn])
        encoder_output = self.layernorm(encoder_output)

        #output (taking average of all hidden states)
        output = self.global_average_pooling_1d(encoder_output)
        # output = self.dense(output)
        # output = self.dropout(output)
        output = self.out(output)
        return output

    def build_model(self, input_shape):
        input_data = layers.Input(shape=(input_shape,), dtype="float32")
        model = tf.keras.Model(inputs=input_data, outputs=self.call(input_data, training=False))
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=self.config["learning_rate"]), 
                                                        loss=['binary_crossentropy'], 
                                                        metrics=['accuracy'])
        model.summary()
        return model

class train_transformer(object):
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
        train_sentences = self.pad(train_sentences, maxlen)
        val_sentences = self.pad(val_sentences, maxlen)
        test_sentences = self.pad(test_sentences, maxlen)
        train_dataset = (train_sentences, train_sentiment_labels)
        val_dataset = (val_sentences, val_sentiment_labels)
        test_dataset = (test_sentences, test_sentiment_labels)

        # Create additional validation datasets
        additional_validation_datasets = []
        for key, value in test_datasets.items():
            sentences = self.vectorize(test_datasets[key]["sentence"])
            sentences = self.pad(sentences, maxlen)
            sentiment_labels = np.array(test_datasets[key]["sentiment_label"])
            dataset = (sentences, sentiment_labels, key)
            additional_validation_datasets.append(dataset)

        # Define callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',              # 1. Calculate val_loss_1 
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
        vocab_size = len(vocab)
        embed_dim = word_vectors.shape[1]
        model = TransformerBlock(config=self.config, 
                                embed_dim=embed_dim, 
                                maxlen=maxlen, 
                                num_heads=self.config["hidden_units"], 
                                vocab_size=vocab_size,
                                epsilon=1e-6).build_model(input_shape = maxlen)
        # input_data = layers.Input(shape=(maxlen,), dtype="float32")
        # transformer_block = TransformerBlock(config=self.config, 
        #                                     embed_dim=embed_dim, 
        #                                     maxlen=maxlen, 
        #                                     num_heads=self.config["hidden_units"], 
        #                                     vocab_size=vocab_size,
        #                                     epsilon=1e-6)
        # output = transformer_block(input_data)
        # model = tf.keras.Model(inputs=input_data, outputs=output)
        # model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=self.config["learning_rate"]), 
        #                                                 loss=['binary_crossentropy'], 
        #                                                 metrics=['accuracy'])
        # model.summary()

        self.model = model

        #Train the model
        if self.config["train_model"] == True:
            model.fit(x=train_dataset[0], 
                    y=train_dataset[1], 
                    batch_size=self.config["mini_batch_size"], 
                    epochs=self.config["train_epochs"], 
                    validation_data=val_dataset, 
                    callbacks=my_callbacks)

            #Save trained model
            if not os.path.exists("assets/trained_models/"):
                os.makedirs("assets/trained_models/")
            model.save_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

        if self.config["evaluate_model"] == True:

            #Load model
            self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

            #Results to be created after evaluation
            results = test_datasets["test_dataset"].copy()

            #Evaluation and predictions
            evaluations = self.model.evaluate(x=test_dataset[0], y=test_dataset[1])
            print("test loss, test acc:", evaluations)
            predictions = self.model.predict(x=test_dataset[0][0])
            print(len(predictions))

            #Create results
            results['sentiment_probability_output'] = []
            results['sentiment_prediction_output'] = []
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

            # print(len(test_sentences))
            # predictions_for_lime = self.prediction(test_sentences)
            # print(predictions_for_lime)
            # test_datapoint = test_sentences[0]
            # print(test_datapoint)
            # tokenized_sentence = test_datapoint.split()
            # print(tokenized_sentence)
            # exp = explainer.explain_instance(test_datapoint, self.prediction, num_features = len(tokenized_sentence), num_samples=self.config["lime_no_of_samples"])
            # print(exp.as_list())

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
        
        #save the configuration parameters (hyperparameters)
        if not os.path.exists("assets/configurations/"):
            os.makedirs("assets/configurations/")
        with open("assets/configurations/"+self.config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)