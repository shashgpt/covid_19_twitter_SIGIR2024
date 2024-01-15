from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input, Softmax, Lambda
from tensorflow.keras.initializers import RandomUniform
import tensorflow as tf
import numpy as np
import os
import pickle

from scripts.training.additional_validation_sets import AdditionalValidationSets

class SimpleAttnClassifier(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, maxlen, label_dim=1, scale=10, attn_type='dot'):
        super(SimpleAttnClassifier, self).__init__()

        self.maxlen = maxlen
        self.embeddings = Embedding(vocab_size, 
                                    embed_dim, 
                                    embeddings_initializer=RandomUniform(minval=-0.1, maxval=0.1))
        self.dropout = Dropout(0.5)
        self.affine = Dense(hidden_dim)
        self.sigmoid = Dense(1, activation='sigmoid')
        self.attn_linear = Dense(hidden_dim, activation='tanh')
        self.scale = scale
        self.V = tf.Variable(tf.random.normal((hidden_dim, 1)))
        self.decoder = Dense(label_dim, use_bias=False)
        self.attn_type = attn_type

    def call(self, inputs, training=False):
        seq_ids = inputs
        seq_lengths = self.maxlen

        seq_embs = self.embeddings(seq_ids)
        seq_embs = self.dropout(seq_embs, training=training)
        batch_size, max_len, hidden_dim = seq_embs.shape

        hidden_vecs = seq_embs
        if self.attn_type == 'dot':
            inter_out = hidden_vecs
        else:
            inter_out = self.attn_linear(hidden_vecs)

        scores = tf.matmul(inter_out, self.V)
        scores = scores / self.scale

        # Mask the padding values
        mask = tf.sequence_mask(seq_lengths, max_len, dtype=tf.float32)
        mask = tf.expand_dims(1.0 - mask, axis=-1)
        scores = scores + mask * -1e9  # Adding a large negative value to mask padding

        attn = Softmax(axis=-2)(scores)
        final_vec = tf.reduce_sum(attn * hidden_vecs, axis=-2)
        final_vec = self.dropout(final_vec, training=training)

        senti_scores = self.decoder(final_vec)
        probs = self.sigmoid(senti_scores)

        return probs

class train_attention(object):
    def __init__(self, config):
        self.config = config
        self.vectorize_layer = None
    
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
        maxlen = max([train_sentences.shape[1], val_sentences.shape[1], test_sentences.shape[1]])
        train_sentences = self.pad(train_sentences, maxlen)
        val_sentences = self.pad(val_sentences, maxlen)
        train_dataset = (train_sentences, train_sentiment_labels)
        val_dataset = (val_sentences, val_sentiment_labels)

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
        my_callbacks = [early_stopping_callback, AdditionalValidationSets(additional_validation_datasets, self.config)]

        #model compilation and summarization
        vocab_size = len(vocab)
        embed_dim = word_vectors.shape[1]
        model = SimpleAttnClassifier(vocab_size, embed_dim, 128, maxlen)
        model.compile(tf.keras.optimizers.legacy.Adam(
                learning_rate=self.config["learning_rate"]), 
                loss=['binary_crossentropy'], 
                metrics=['accuracy'])
        model.build(input_shape = ((None, None), (None,)))
        model.summary()

        # Train the model
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

# # Example usage:
# vocab_size = 10000  # Replace with your actual vocabulary size
# embed_dim = 300    # Replace with your desired embedding dimension
# hidden_dim = 128   # Replace with your desired hidden dimension
# model = SimpleAttnClassifier(vocab_size, embed_dim, hidden_dim)
# model.build(input_shape=((None, None), (None,)))
# model.summary()


