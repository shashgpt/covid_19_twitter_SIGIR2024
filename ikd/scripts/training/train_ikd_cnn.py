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
import random
from lime import lime_text
import traceback
from tqdm import tqdm
from transformers import TFAutoModel
from transformers import AutoTokenizer

from scripts.training.additional_validation_sets import AdditionalValidationSets

class FOL_rules(object):
    def __init__(self, classes, input, features):
        self.classes = classes
        self.input = input
        self.features = features

class FOL_A_but_B(FOL_rules):
    def __init__(self, classes, input, features):
        assert classes == 1
        super(FOL_A_but_B, self).__init__(classes, input, features)

    def log_distribution(self, w, batch_size, X=None, F=None):
        if F == None:
            X, F = self.input, self.features
        F_mask = F[:,0] #f_but_ind
        F_fea = F[:,1] #f_but_y_pred_p
        distr_y1 = tf.math.multiply(w, tf.math.multiply(F_mask, F_fea)) #y = 1 
        distr_y0 = tf.math.multiply(w, tf.math.multiply(F_mask, tf.math.subtract(1.0, F_fea))) #y = 0
        distr_y0 = tf.reshape(distr_y0, [batch_size, self.classes])
        distr_y1 = tf.reshape(distr_y1, [batch_size, self.classes])
        distr = tf.concat([distr_y0, distr_y1], axis=1)
        return distr

class Teacher_network(object):
    def __init__(self, batch_size, classes, rules, rules_lambda, teacher_regularizer):
        self.batch_size = batch_size
        self.classes = classes
        self.rules = rules
        self.rules_lambda = rules_lambda
        self.teacher_regularizer = teacher_regularizer

    def calc_rule_constraints(self, rules, rules_lambda, teacher_regularizer, batch_size, classes, new_data=None, new_rule_fea=None):
        if new_rule_fea==None:
            new_rule_fea = [None]*len(rules)
        distr_all = tf.zeros([batch_size, classes], dtype=tf.dtypes.float32)
        for i, rule in enumerate(rules):
            distr = rule.log_distribution(teacher_regularizer*rules_lambda[i], batch_size, new_data, new_rule_fea[i])
            distr_all = tf.math.add(distr_all, distr)
        distr_all = tf.math.add(distr_all, distr)
        distr_y0 = distr_all[:,0]
        distr_y0 = tf.reshape(distr_y0, [batch_size, 1])
        distr_y0_copies = tf.concat([tf.identity(distr_y0), tf.identity(distr_y0)], axis=1)
        distr_all = tf.math.subtract(distr_all, distr_y0_copies)
        distr_all = tf.math.maximum(tf.math.minimum(distr_all, tf.constant([60.])), tf.constant([-60.])) # truncate to avoid over-/under-flow
        distr_all = tf.math.exp(distr_all)
        return distr_all

    def teacher_output(self, student_output):
        distr = self.calc_rule_constraints(rules = self.rules, 
                                            rules_lambda = self.rules_lambda, 
                                            teacher_regularizer = self.teacher_regularizer, 
                                            batch_size = self.batch_size, 
                                            classes = self.classes)
        q_y_given_x = tf.math.multiply(student_output, distr)
        teacher_output = tf.math.divide(q_y_given_x, tf.reshape(tf.math.reduce_sum(q_y_given_x, axis=1), [-1, 1]))
        teacher_output = teacher_output[:,1]
        return teacher_output

class iteration_tracker(tf.keras.metrics.Metric):
    def __init__(self, name='iteration', **kwargs):
        super(iteration_tracker, self).__init__(name=name, **kwargs)
        self.iteration = self.add_weight(name='iteration', initializer='zeros')

    def update_state(self, curr_iter, sample_weight=None):
        self.iteration.assign_add(curr_iter)

    def result(self):
        return self.iteration

    def reset_states(self):
        self.iteration.assign(self.iteration)

class distillation_loss(tf.keras.metrics.Metric):
    def __init__(self, name='iteration', **kwargs):
        super(distillation_loss, self).__init__(name=name, **kwargs)
        self.distillation_loss = self.add_weight(name='distillation_loss', initializer='zeros')

    def update_state(self, distillation_loss, sample_weight=None):
        self.distillation_loss.assign(distillation_loss)

    def result(self):
        return self.distillation_loss

    def reset_states(self):
        self.distillation_loss.assign(0)

acc_tracker_per_epoch = tf.keras.metrics.BinaryAccuracy(name="accuracy")
iteration_tracker_metric = iteration_tracker()
distillation_loss_metric = distillation_loss()
MINI_BATCH_SIZE = 0

class IKD(Model):

    def compute_mask(self, inputs, mask=None):
        # Also split the mask into 2 if it presents.
        if mask is None:
            return None
        return tf.split(mask, 2, axis=1)

    def train_step(self, data): # an iteration
        x,  y = data
        sentences = x[0]
        rule_features = x[1]
        sentiment_labels = y[0]
        rule_features_ind = y[1]

        with tf.GradientTape() as tape: # Forward propagation and loss calculation

            # # IKD from my understanding
            # y_pred = self(sentences, training=True)  #Forward pass
            # f_but_y_pred_p = self(rule_features, training=True)
            # distr = tf.math.multiply(f_but_y_pred_p, rule_features_ind, name=None) #check
            # distr = tf.math.maximum(tf.math.minimum(distr, tf.constant([60.])), tf.constant([-60.]))
            # multiply_but_exp = tf.math.exp(distr) #check
            # q_y_given_x = tf.math.multiply(y_pred, multiply_but_exp, name=None) #check
            # teacher = tf.math.divide(q_y_given_x, tf.reshape(tf.math.reduce_sum(q_y_given_x, axis=1), [-1, 1]))
            # loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # loss_fn_rule = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # m = tf.math.multiply(self.iteration_tracker_metric.result(), 1./1408)
            # e = tf.math.pow(0.95, m)
            # max = tf.math.maximum(e, 0.0)
            # distillation_str = tf.math.subtract(1.0, max)
            # s1 = tf.math.subtract(1.0, distillation_str)
            # l1 = tf.math.multiply(loss_fn_data(sentiment_labels, y_pred), s1)
            # l2 = tf.math.multiply(loss_fn_rule(teacher, y_pred), distillation_str)
            # loss_value = tf.math.add(l1, l2)

            # IKD from authors code
            y_pred = self(sentences, training=True)
            if len(rule_features) == 4:
                f_but_y_pred_p = self(rule_features[0], training=True)
                f_yet_y_pred_p = self(rule_features[1], training=True)
                f_though_y_pred_p = self(rule_features[2], training=True)
                f_while_y_pred_p = self(rule_features[3], training=True)
                f_but_full = tf.concat([rule_features_ind[0], f_but_y_pred_p], axis=1)
                f_yet_full = tf.concat([rule_features_ind[1], f_yet_y_pred_p], axis=1)
                f_though_full = tf.concat([rule_features_ind[2], f_though_y_pred_p], axis=1)
                f_while_full = tf.concat([rule_features_ind[3], f_while_y_pred_p], axis=1)
                rules = [FOL_A_but_B(classes = 1, input = input, features = f_but_full), 
                        FOL_A_but_B(classes = 1, input = input, features = f_yet_full),
                        FOL_A_but_B(classes = 1, input = input, features = f_though_full),
                        FOL_A_but_B(classes = 1, input = input, features = f_while_full)]
                
            elif len(rule_features) == 1:
                y_pred = self(sentences, training=True)
                f_but_y_pred_p = self(rule_features[0], training=True)
                f_but_full = tf.concat([rule_features_ind[0], f_but_y_pred_p], axis=1)
                rules = [FOL_A_but_B(classes = 1, input = input, features = f_but_full)]

            class_object = Teacher_network(batch_size = MINI_BATCH_SIZE, 
                                           classes = 1, 
                                           rules = rules, 
                                           rules_lambda = [1.0, 1.0, 1.0, 1.0], 
                                           teacher_regularizer = 1.0)
            teacher = class_object.teacher_output(student_output = y_pred)
            loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            loss_fn_rule = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            m = tf.math.multiply(iteration_tracker_metric.result(), 1./1408)
            e = tf.math.pow(0.95, m)
            max = tf.math.maximum(e, 0.0)
            distillation_str = tf.math.subtract(1.0, max)
            s1 = tf.math.subtract(1.0, distillation_str)
            l1 = tf.math.multiply(loss_fn_data(sentiment_labels, y_pred), s1)
            l2 = tf.math.multiply(loss_fn_rule(teacher, y_pred), distillation_str)
            loss_value = tf.math.add(l1, l2)
            # loss_value = loss_fn_data(sentiment_labels, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        distillation_loss_metric.update_state(loss_value)
        acc_tracker_per_epoch.update_state(sentiment_labels, y_pred)
        iteration_tracker_metric.update_state(1.0)
        return {
                "loss": distillation_loss_metric.result(), 
                "accuracy": acc_tracker_per_epoch.result(), 
                "iteration": iteration_tracker_metric.result()
               }

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [distillation_loss_metric, acc_tracker_per_epoch, iteration_tracker_metric]
    
    def test_step(self, data):

        x, y = data
        sentences = x[0]
        rule_features = x[1]
        sentiment_labels = y[0]
        rule_features_ind = y[1]

        # Compute predictions
        y_pred = self(sentences, training=True)
        loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        l1 = loss_fn_data(sentiment_labels, y_pred)

        # Compute our own metrics
        distillation_loss_metric.update_state(l1)
        acc_tracker_per_epoch.update_state(sentiment_labels, y_pred)
        return {"loss": distillation_loss_metric.result(), 
                "accuracy": acc_tracker_per_epoch.result(), 
                "iteration": iteration_tracker_metric.result()}

def cnn(config, word_vectors): # Similar to one used in pytorch code for CIKM submission
    global MINI_BATCH_SIZE
    MINI_BATCH_SIZE = config["mini_batch_size"]

    input_sentence = Input(shape=(None,), dtype="int64")
    word_embeddings = layers.Embedding(word_vectors.shape[0], 
                                        word_vectors.shape[1], 
                                        embeddings_initializer=Constant(word_vectors), 
                                        trainable=config["fine_tune_word_embeddings"], 
                                        mask_zero=True, 
                                        name="word_embeddings")(input_sentence)

    word_embeddings_reshaped = tf.keras.backend.expand_dims(word_embeddings, axis=1) # batch_size x 1 x sent_len x embedding_dim

    conv_1 = layers.Conv2D(filters = config["hidden_units"], 
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

    conv_2 = layers.Conv2D(filters = config["hidden_units"], 
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

    conv_3 = layers.Conv2D(filters = config["hidden_units"], 
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

    model = IKD(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":   
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"]), 
                                                      loss=['binary_crossentropy'], 
                                                      metrics=['accuracy'])   
    elif config["optimizer"] == "adadelta":
        model.compile(tf.keras.optimizers.Adadelta(learning_rate=config["learning_rate"], 
                                                   rho=0.95, epsilon=1e-06), 
                                                   loss=['binary_crossentropy'], 
                                                   metrics=['accuracy'])

    return model

class train_ikd_cnn(object):
    def __init__(self, config):
        self.config = config
    
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

    def rule_conjunct_extraction(self, dataset, rule):
        """
        Extracts the rule_conjuncts from sentences containing the logic rule corresponding to rule_keyword
        """
        if self.config["dataset_name"] in ["covid19-twitter", "covid19-twitter_RulesBalanced"]:
            rule_conjuncts = []
            rule_label_ind = []
            for index, sentence in enumerate(list(dataset['sentence'])):
                tokenized_sentence = sentence.split()
                rule_label = dataset['rule_label'][index]
                contrast = dataset['contrast'][index]
                if rule_label == rule and contrast==1:
                    if rule_label == 1:
                        b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("but")+1:]
                        b_part_sentence = ' '.join(b_part_tokenized_sentence)
                        rule_conjuncts.append(b_part_sentence)
                        rule_label_ind.append(1)
                    elif rule_label == 2:
                        b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("yet")+1:]
                        b_part_sentence = ' '.join(b_part_tokenized_sentence)
                        rule_conjuncts.append(b_part_sentence)
                        rule_label_ind.append(1)
                    elif rule_label == 3:
                        a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("though")]
                        a_part_sentence = ' '.join(a_part_tokenized_sentence)
                        rule_conjuncts.append(a_part_sentence)
                        rule_label_ind.append(1)
                    elif rule_label == 4:
                        a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("while")]
                        a_part_sentence = ' '.join(a_part_tokenized_sentence)
                        rule_conjuncts.append(a_part_sentence)
                        rule_label_ind.append(1)
                else:
                    rule_conjuncts.append('')
                    rule_label_ind.append(0)
            return rule_conjuncts, rule_label_ind

        elif self.config["dataset_name"] in ["sst2"]:
            rule_conjuncts = []
            rule_label_ind = []
            for index, sentence in enumerate(list(dataset['sentence'])):
                tokenized_sentence = sentence.split()
                rule_label = dataset['rule_label'][index]
                if rule_label == 1:
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("but")+1:]
                    b_part_sentence = ' '.join(b_part_tokenized_sentence)
                    rule_conjuncts.append(b_part_sentence)
                    rule_label_ind.append(1)
                else:
                    rule_conjuncts.append('')
                    rule_label_ind.append(0)
            return rule_conjuncts, rule_label_ind
    
    def remove_extra_samples(self, sample):
        sample = sample[:(len(sample)-len(sample)%self.config["mini_batch_size"])]
        return sample
    
    def create_dataset(self, sentences, sentiment_labels, dataset, key=None):

        if self.config["dataset_name"] in ["covid19-twitter", "covid19-twitter_RulesBalanced"]:
            sentences_but_features, sentences_but_features_ind = self.rule_conjunct_extraction(dataset, rule=1)
            sentences_yet_features, sentences_yet_features_ind = self.rule_conjunct_extraction(dataset, rule=2)
            sentences_though_features, sentences_though_features_ind = self.rule_conjunct_extraction(dataset, rule=3)
            sentences_while_features, sentences_while_features_ind = self.rule_conjunct_extraction(dataset, rule=4)
            sentences_but_features = self.vectorize(sentences_but_features)
            sentences_but_features_ind = np.array(sentences_but_features_ind).astype(np.float32)
            sentences_but_features_ind = sentences_but_features_ind.reshape(sentences_but_features_ind.shape[0], 1)
            sentences_yet_features = self.vectorize(sentences_yet_features)
            sentences_yet_features_ind = np.array(sentences_yet_features_ind).astype(np.float32)
            sentences_yet_features_ind = sentences_yet_features_ind.reshape(sentences_yet_features_ind.shape[0], 1)
            sentences_though_features = self.vectorize(sentences_though_features)
            sentences_though_features_ind = np.array(sentences_though_features_ind).astype(np.float32)
            sentences_though_features_ind = sentences_though_features_ind.reshape(sentences_though_features_ind.shape[0], 1)
            sentences_while_features = self.vectorize(sentences_while_features)
            sentences_while_features_ind = np.array(sentences_while_features_ind).astype(np.float32)
            sentences_while_features_ind = sentences_while_features_ind.reshape(sentences_while_features_ind.shape[0], 1)
            sentences = self.remove_extra_samples(sentences)
            sentiment_labels = self.remove_extra_samples(sentiment_labels)
            sentences_but_features = self.remove_extra_samples(sentences_but_features)
            sentences_yet_features = self.remove_extra_samples(sentences_yet_features)
            sentences_though_features = self.remove_extra_samples(sentences_though_features)
            sentences_while_features = self.remove_extra_samples(sentences_while_features)
            sentences_but_features_ind = self.remove_extra_samples(sentences_but_features_ind)
            sentences_yet_features_ind = self.remove_extra_samples(sentences_yet_features_ind)
            sentences_though_features_ind = self.remove_extra_samples(sentences_though_features_ind)
            sentences_while_features_ind = self.remove_extra_samples(sentences_while_features_ind)

            if key == None:
                dataset_rule_features = ([sentences, [sentences_but_features, 
                                                        sentences_yet_features, 
                                                        sentences_though_features, 
                                                        sentences_while_features]], 
                                        [sentiment_labels, [sentences_but_features_ind, 
                                                            sentences_yet_features_ind, 
                                                            sentences_though_features_ind, 
                                                            sentences_while_features_ind]])
            else:
                dataset_rule_features = ([sentences, [sentences_but_features, 
                                                        sentences_yet_features, 
                                                        sentences_though_features, 
                                                        sentences_while_features]], 
                                        [sentiment_labels, [sentences_but_features_ind, 
                                                            sentences_yet_features_ind, 
                                                            sentences_though_features_ind, 
                                                            sentences_while_features_ind]], key)

            return dataset_rule_features
        
        elif self.config["dataset_name"] in ["sst2"]:
            sentences_but_features, sentences_but_features_ind = self.rule_conjunct_extraction(dataset, rule=1)
            sentences_but_features = self.vectorize(sentences_but_features)
            sentences_but_features_ind = np.array(sentences_but_features_ind).astype(np.float32)
            sentences_but_features_ind = sentences_but_features_ind.reshape(sentences_but_features_ind.shape[0], 1)
            sentences = self.remove_extra_samples(sentences)
            sentiment_labels = self.remove_extra_samples(sentiment_labels)
            sentences_but_features = self.remove_extra_samples(sentences_but_features)
            sentences_but_features_ind = self.remove_extra_samples(sentences_but_features_ind)

            if key == None:
                dataset_rule_features = ([sentences, [sentences_but_features]], 
                                        [sentiment_labels, [sentences_but_features_ind]])
            else:
                dataset_rule_features = ([sentences, [sentences_but_features]], 
                                        [sentiment_labels, [sentences_but_features_ind]], key)

            return dataset_rule_features

    def train_model(self, train_dataset, val_datasets, test_datasets, word_index, word_vectors):

        #Make paths
        if not os.path.exists("assets/training_history/"):
            os.makedirs("assets/training_history/")
        
        #create vocab and define the vectorize layer
        vocab = [key for key in word_index.keys()]
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=vocab)

        #Create Train, Val, and Test datasets
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
        train_dataset = self.create_dataset(train_sentences, train_sentiment_labels, train_dataset)
        val_dataset = self.create_dataset(val_sentences, val_sentiment_labels, val_datasets["val_dataset"])
        test_dataset = self.create_dataset(test_sentences, test_sentiment_labels, test_datasets["test_dataset"])

        #Create additional validation datasets
        additional_validation_datasets = []
        for key, value in test_datasets.items():
            sentences = self.vectorize(test_datasets[key]["sentence"])
            sentences = self.pad(sentences, maxlen)
            sentiment_labels = np.array(test_datasets[key]["sentiment_label"])
            dataset = self.create_dataset(sentences, sentiment_labels, test_datasets[key], key)
            additional_validation_datasets.append(dataset)

        #Define callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',              # 1. Calculate val_loss_1 
                                                                    min_delta = 0,                  # 2. Check val_losses for next 10 epochs 
                                                                    patience=10,                    # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                    verbose=0,                      # 4. Get the trained weights corresponding to val_loss_1
                                                                    mode="min",
                                                                    baseline=None, 
                                                                    restore_best_weights=True)
        my_callbacks = [
                        #early_stopping_callback, 
                        AdditionalValidationSets(additional_validation_datasets, self.config)
                       ]

        #model compilation and summarization
        model = cnn(self.config, word_vectors)  
        model.summary()
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

            #Remove extra samples
            for key in test_datasets["test_dataset"].keys():
                test_datasets["test_dataset"][key] = self.remove_extra_samples(test_datasets["test_dataset"][key])

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

            #Save the results
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

            if self.config["dataset_name"] in ["covid19-twitter", "covid19-twitter_RulesBalanced"]:
                test_sentences = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentence']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentence'])
                probabilities = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_probability_output']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_probability_output'])
            elif self.config["dataset_name"] in ["sst2"]:
                test_sentences = list(results.loc[(results["rule_label"]!=0)]['sentence']) + list(results.loc[(results["rule_label"]!=0)]['sentence'])
                probabilities = list(results.loc[(results["rule_label"]!=0)]['sentiment_probability_output']) + list(results.loc[(results["rule_label"]!=0)]['sentiment_probability_output'])

            explainer = lime_text.LimeTextExplainer(class_names=["negative_sentiment", "positive_sentiment"], 
                                                    split_expression=" ", 
                                                    random_state=self.config["seed_value"])

            for index, test_datapoint in enumerate(tqdm(test_sentences)):
                probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                tokenized_sentence = test_datapoint.split()
                try:
                    exp = explainer.explain_instance(test_datapoint, 
                                                     self.prediction, 
                                                     num_features = len(tokenized_sentence), 
                                                     num_samples=self.config["lime_no_of_samples"])
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
        
        #Save the configuration parameters (hyperparameters)
        if not os.path.exists("assets/configurations/"):
            os.makedirs("assets/configurations/")
        with open("assets/configurations/"+self.config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)
