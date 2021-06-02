import tensorflow as tf
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

MAX_SENT_LENGTH = 256

# Create the model


class BackgroundClassificationModel(tf.keras.Model):
    def __init__(self, bert_encoder, bert_preprocess_model, num_classes):
        super(BackgroundClassificationModel, self).__init__()

        # Define BERT tokenizer
        self.tokenize = hub.KerasLayer(bert_preprocess_model.tokenize)

        # Define BERT preprocessing model with sentence length = MAX_STRING_LENGTH
        self.bert_pack_inputs = hub.KerasLayer(
            bert_preprocess_model.bert_pack_inputs, arguments=dict(seq_length=MAX_SENT_LENGTH))

        self.bert_encoder = bert_encoder

        # Define output layer
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        # x: tensor/numpy array of strings

        # Tokenize input strings
        tokenized_inputs = self.tokenize(x)

        # Pass input_word_ids to preprocessing model to generate
        # inputs compatible with BERT model
        encoder_inputs = self.bert_pack_inputs([tokenized_inputs])

        # Pass inputs through BERT encoder
        encoder_outputs = self.bert_encoder(encoder_inputs)

        # Get cls embeddings
        cls_embeddings = encoder_outputs['pooled_output']  # (batch_size, 256)

        # Get output class for each token
        outputs = self.dense(cls_embeddings)  # (batch_size, num_classes)

        return outputs


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
    # Preprocess and clean text
    text = preprocess_string(text)

    # Get output
    result = background_model([text])

    # Determine backgrounds
    background = tf.argmax(result[0])

    return background.numpy()


# Loads and return the instance of the model
def init():
    # Load bert preprocess model
    bert_preprocess_model_path = 'bert_preprocess_model/'
    bert_preprocess_model = hub.load(bert_preprocess_model_path)

    # Load BERT encoder
    bert_encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/2",
        trainable=True)

    # Load background classes
    with open('Data/Backgrounds.txt', mode='r') as f:
        background_classes = json.load(f)

    # Initialize model
    model = BackgroundClassificationModel(
        bert_encoder, bert_preprocess_model, len(background_classes.keys()))

    model.load_weights(
        'Weights/Background Classification Model Weights/weights')

    return model


background_model = init()
