import tensorflow as tf
import numpy as np
import spacy
import re
import json

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks
import tensorflow_hub as hub
import tensorflow_text as text

MAX_SENT_LENGTH = 64

# Create the model


class SegmentationModel(tf.keras.Model):
    def __init__(self, bert_encoder, bert_preprocess_model, lstm_units, num_classes):
        super(SegmentationModel, self).__init__()

        self.lstm_units = lstm_units

        # Define BERT tokenizer
        self.tokenize = hub.KerasLayer(bert_preprocess_model.tokenize)

        # Define BERT preprocessing model with sentence length = max_sent_length
        self.bert_pack_inputs = hub.KerasLayer(
            bert_preprocess_model.bert_pack_inputs, arguments=dict(seq_length=MAX_SENT_LENGTH))

        self.bert_encoder = bert_encoder

        # Define dropout layer
        self.dropout = tf.keras.layers.Dropout(0.1)

        # Define output layer
        self.dense = tf.keras.layers.Dense(num_classes)

        # Define LSTM cell
        self.lstm_cell = tf.keras.layers.LSTM(
            lstm_units, return_state=True, return_sequences=True)

    def call(self, x):
        # x: tensor/numpy list of strings # (batch_size, MAX_SENTENCES_IN_BATCH)

        hidden_state, cell_state = self.initial_states(x.shape[0])

        outputs = []

        # Go through each sentence
        for i in range(x.shape[1]):
            # Tokenize input strings
            # x[:, i] # (batch_size, )
            tokenized_inputs = self.tokenize(x[:, i])

            # Pass input_word_ids to preprocessing model to generate
            # inputs compatible with BERT model
            encoder_inputs = self.bert_pack_inputs([tokenized_inputs])

            # Pass inputs through BERT encoder
            encoder_output = self.bert_encoder(encoder_inputs)

            # Get context embeddings of all input tokens
            # (256 is the size of one contextual embedding)
            # (batch_size, 256)
            context_embedding = encoder_output['pooled_output']

            # Add dimension to context embedding
            context_embedding = tf.expand_dims(context_embedding, axis=1)

            # Pass context_embedding to lstm_cell
            _, hidden_state, cell_state = self.lstm_cell(
                context_embedding, initial_state=[hidden_state, cell_state])

            # Pass hidden stats through drop out
            output = self.dropout(hidden_state)  # (batch_size, lstm_units)

            # Get classes
            output = self.dense(output)  # (batch_size, classes)

            # (MAX_SENTENCES_IN_BATCH, batch_size, classes)
            outputs.append(output)

        # (MAX_SENTENCES_IN_BATCH, batch_size, classes)
        outputs = tf.convert_to_tensor(outputs)

        # Return outputs over input tokens
        # (batch_size, MAX_SENTENCES_IN_BATCH, classes)
        return tf.reshape(outputs, shape=[x.shape[0], x.shape[1], -1])

    def initial_states(self, batch_size):
        return tf.zeros((batch_size, self.lstm_units)), tf.zeros((batch_size, self.lstm_units))


# Function to preprocess string
def preprocess_string(sent):
    # Lowercase string
    sent = sent.lower()
    # Remove dialog from string
    sent = re.sub('".*?"', '', sent)
    # Remove puntuations except for full stops and question marks
    sent = re.sub('[^\w^ ^\.^\?]', '', sent)
    # Give space between punctuations
    sent = re.sub(r'[\.]', ' .', sent)
    sent = re.sub(r'[\?]', ' ?', sent)
    # Remove Multiple spaces to single space
    sent = re.sub(' {2,}', ' ', sent)

    return sent


def inference(text):
    inp = []
    # Preproces string
    text = preprocess_string(text)
    # Seperate sentences
    sentences = list(spacy_model(text).sents)
    sentences = list(map(lambda sent: sent.text, sentences))

    inp = np.array([sentences], dtype=object)

    # Get output
    results = segmentation_model(inp)

    # Segment background
    results = tf.argmax(results[0], axis=1)
    results = results.numpy()

    # Add first sentence to scenes
    scenes = [sentences[0]]

    # Go through each sentences
    for i in range(1, len(results)):
        # Check if new background is found
        if results[i] == segmentation_classes['begin']:
            scenes.append(sentences[i])
        else:
            # Append the sentence to the last sentences in scenes
            scenes[-1] += ' ' + sentences[i]

    return scenes


# Loads and return the instance of the model
def init():
    # Load segmentation classes
    with open('Data/Segmentation Classes.txt', mode='r') as f:
        segmentation_classes = json.load(f)

    # Add class for pad
    segmentation_classes['pad'] = 0

    # Load bert preprocess model
    spacy_model = spacy.load("en_core_web_sm")

    bert_preprocess_model_path = 'bert_preprocess_model/'
    bert_preprocess_model = hub.load(bert_preprocess_model_path)

    # Define BERT encoder
    bert_encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/2",
        trainable=False)

    lstm_units = 128

    # Initialize model
    model = SegmentationModel(bert_encoder, bert_preprocess_model, lstm_units, len(
        segmentation_classes.keys()))

    model.load_weights('Weights/Segmentation Model Weights/weights')

    return model, spacy_model, segmentation_classes


segmentation_model, spacy_model, segmentation_classes = init()
