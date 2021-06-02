from spacy.matcher import Matcher
import spacy
import re
import concurrent.futures
import json

import tensorflow as tf
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


MAX_SENT_LENGTH = 255


class CharacterInformationExtractionModel(tf.keras.Model):
    def __init__(self, bert_preprocess_model, n_character_types, n_actions, n_character_present):
        super(CharacterInformationExtractionModel, self).__init__()

        self.n_actions = n_actions

        # Define BERT tokenizer
        self.tokenize = hub.KerasLayer(
            bert_preprocess_model.tokenize,
            name='tokenizer_1')

        # Define BERT preprocessing model with sentence length = MAX_SENT_LENGTH
        self.bert_pack_inputs = hub.KerasLayer(
            bert_preprocess_model.bert_pack_inputs,
            arguments=dict(seq_length=MAX_SENT_LENGTH),
            name='preprocessor_1')

        # Load BERT encoder
        self.bert_encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2",
            trainable=True,
            name='encoder_1')

        # Define dense layer for character types
        self.character_type_layer = tf.keras.layers.Dense(
            n_character_types,
            activation='softmax',
            name='dense_1')

        # Define dense layer for character is present
        self.character_is_present_layer = tf.keras.layers.Dense(
            n_character_present,
            activation='softmax',
            name='dense_2')

        # Define layer to classify character action
        self.character_action_layer = tf.keras.layers.Dense(
            n_actions,
            activation='softmax',
            name='dense_3')

    def call(self, x):
        # x: tokenized input (batch_size, 1, 1, MAX_SENT_LENGTH)

        # Pass input_word_ids to preprocessing model to generate
        # inputs compatible with BERT model
        encoder_inputs = self.bert_pack_inputs([x])

        # Pass inputs through BERT encoder
        bert_output = self.bert_encoder(encoder_inputs)

        # Get CLS token output for character type classifification
        # and whether character is present or not
        pooled_output = bert_output['pooled_output']  # (batch_size, 256)

        # Get context embeddings of each word for action classification
        # (batch_size, sentence_length, 256)
        sequence_output = bert_output['sequence_output']

        # Pass cls embeddings through character classification layer
        character_type = self.character_type_layer(
            pooled_output)  # (batch_size, n_character_types)

        # Pass cls embeddings through character is present classification layer
        character_is_present = self.character_is_present_layer(
            pooled_output)  # (batch_size, n_character_present)

        actions = []

        # Either calculate the average embeddings, or concatenate the embeddings
        # of three words
        for i in range(0, sequence_output.shape[1], 3):
            # Concatenate embeddings
            embeddings = tf.reshape(
                sequence_output[:, i: i + 3, :],
                shape=[sequence_output.shape[0], -1])  # (batch_size, 256 * 3)

            # Make classification of actions
            action = self.character_action_layer(
                embeddings)  # (batch_size, n_actions)

            # (sentence_length / 3, batch_size, n_actions)
            actions.append(action)

            # (batch_size, sentence_length / 3, n_actions)
        return tf.reshape(tf.convert_to_tensor(actions), [x.shape[0], -1, self.n_actions]), character_type, character_is_present


def extract_characters(text):
    # Create spacy document from text
    doc = spacy_model(text)

    # Define matcher class object to get characters
    # from text based on some rules
    matcher = Matcher(spacy_model.vocab)

    # Pattern to find adjective noun pairs. Each adjective noun pair can have atmost 2 and
    # at least 1 adjective and 1 noun
    pattern_1 = [{'POS': 'ADJ', 'OP': '?'},
                 {'POS': 'ADJ'},
                 {'POS': 'NOUN'}]

    # Extract Proper nouns from text
    pattern_2 = [{'POS': 'PROPN'}]

    # Extract Nouns from text
    pattern_3 = [{'POS': 'NOUN'}]

    matcher.add("Proper Noun", [pattern_2])
    matcher.add("Adjective_Noun", [pattern_1])
    matcher.add("Noun", [pattern_3])

    matches = matcher(doc)

    spans = []

    # Extract characters
    for match_id, start, end in matches:
        # Get string representation
        string_id = spacy_model.vocab.strings[match_id]
        span = doc[start:end]  # The matched span

        # Extract proper nouns
        if(string_id == 'Proper Noun'):
            spans.append(span)

        # Extract relevant nouns
        if(re.findall('(child|man|woman|girl|boy|grandmother|grandfather|father|mother|parents|sister|brother|son|daughter)', span.text)):
            spans.append(span)

        # Extract animals
        if(re.findall('(^rabbit$|^bunny$|^hare$|^dog$|^puppy$|^doggy$|^bitch$|^hound$|^cat$|^kitty$|^kitten$|^pussy$|^puss$|^tom$|^bird$|^fowl$|^pigeon$)', span.text)):
            spans.append(span)

    # Filter out overlapping characters
    spans = spacy.util.filter_spans(spans)

    # Remove duplicates
    characters = set()
    for char in spans:
        characters.add(char.text)

    characters = list(characters)

    # Filter out same character
    for i in range(len(characters)):
        for j in range(len(characters)):
            # Split words
            words = characters[j].split(' ')

            # Check words contains the currently selected character
            if i != j and characters[i] in words:
                characters[j] = ''

    return list(filter(lambda string: string != '', characters))


# Function to preprocess string
def preprocess_string(string):
    # Lowercase text
    string = string.lower()
    # Remove dialog from string
    string = re.sub('".*?"', '', string)
    # Remove escape characters
    string = re.sub('\s', ' ', string)
    # Remove Multiple spaces to single space
    string = re.sub(' {2,}', ' ', string)

    return string

# Function to make inference for single character


def single_character_inference(tokenized_text, character1_name, character_position):
    # Append character to text and seperate with [SEP] token
    character1_name_tokenized = bert_tokenizer([character1_name])[0]
    character2_name_tokenized = bert_tokenizer(['no character'])[0]

    # Get id for SEP token
    sep_token = [tokenizer.convert_tokens_to_ids(['[SEP]'])]

    # Concatenate character names with text
    tokenized_text = tf.concat([tokenized_text, sep_token, character1_name_tokenized,
                                sep_token, character2_name_tokenized, sep_token], axis=0)

    # Add extract dimension to tokenized_text for batch
    tokenized_text = tf.expand_dims(tokenized_text, axis=0)

    # Make inference
    actions, character_type, character_present = character_model(
        tokenized_text)

    # Find the actions, characer_type and character_present
    character_type = tf.argmax(character_type, axis=1)[0]
    character_present = tf.argmax(character_present, axis=1)[0]
    actions = tf.argmax(actions, axis=2)[0]

    return {'character_position': character_position,
            'actions': actions.numpy(),
            'character_name': character1_name,
            'character_type': character_type.numpy(),
            'character_present': character_present.numpy()}


# Function to make inference for pair of characterrs
def character_pair_inference(tokenized_text, character1_name, character2_name, character1_position, character2_position):
    # Append character to text and seperate with [SEP] token
    character1_name = bert_tokenizer([character1_name])[0]
    character2_name = bert_tokenizer([character2_name])[0]

    # Get id for SEP token
    sep_token = [tokenizer.convert_tokens_to_ids(['[SEP]'])]

    # Concatenate character names with text
    tokenized_text = tf.concat([tokenized_text,
                                sep_token,
                                character1_name,
                                sep_token,
                                character2_name,
                                sep_token], axis=0)

    # Add extract dimension to tokenized_text for batch
    tokenized_text = tf.expand_dims(tokenized_text, axis=0)

    # Make inference
    actions, character_type, character_present = character_model(
        tokenized_text)

    # Find the actions, characer_type and character_present
    character_type = tf.argmax(character_type, axis=1)[0]
    character_present = tf.argmax(character_present, axis=1)[0]
    actions = tf.argmax(actions, axis=2)[0]

    return {'character_positions': (character1_position, character2_position),
            'actions': actions.numpy()}


def inference(text):
    # Extract characters from text
    characters = extract_characters(text)

    # Preprocess sentence
    text = preprocess_string(text)

    # Tokenize text
    tokenized_text = bert_tokenizer([text])[0]

    # Get id for SEP token
    sep_token = [tokenizer.convert_tokens_to_ids(['[SEP]'])]

    n = len(characters)

    single_characters_output = []
    character_pairs_output = []

    # Make a thread pool of 10 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        single_character_inference_futures = []
        character_pair_inference_futures = []

        # Go through each character
        for i in range(n):
            # Append the function to futures list, this way we will be able to
            # execute all the futures at once using threads
            single_character_inference_futures.append(executor.submit(
                single_character_inference,
                tokenized_text=tokenized_text,
                character1_name=characters[i],
                character_position=i))

        # Go through each character pair
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                # Append the function to futures list, this way we will be able to
                # execute all the futures at once using threads
                character_pair_inference_futures.append(executor.submit(
                    character_pair_inference,
                    tokenized_text=tokenized_text,
                    character1_name=characters[i],
                    character2_name=characters[j],
                    character1_position=i,
                    character2_position=j))

        # Make inference using threads
        for future in concurrent.futures.as_completed(single_character_inference_futures):
            single_characters_output.append(future.result())

        # Make inference using threads
        for future in concurrent.futures.as_completed(character_pair_inference_futures):
            character_pairs_output.append(future.result())

    return single_characters_output, character_pairs_output


# Loads and return the instance of the model
def init():
    # Setup tokenizer to tokenize sentence
    tokenizer = bert.tokenization.FullTokenizer(
        vocab_file="Data/vocab.txt",
        do_lower_case=True
    )

    # Load spacy model and add Hugging faces' coreference model to the pipeline
    # also add 2 other models to pipeline
    spacy_model = spacy.load("en_core_web_sm")
    spacy_model.add_pipe('merge_noun_chunks')
    spacy_model.add_pipe('merge_entities', after='ner')

    # Load bert preprocess model
    bert_preprocess_model_path = 'bert_preprocess_model/'
    bert_preprocess_model = hub.load(bert_preprocess_model_path)
    bert_tokenizer = hub.KerasLayer(bert_preprocess_model.tokenize)

    # Load actions from file
    with open('Data/actions.txt', mode='r') as f:
        actions_dict = json.load(f)

    # Load character types from file
    with open('Data/character types.txt', mode='r') as f:
        character_types_dict = json.load(f)

    character_present_dict = {
        'p': 1,
        'n': 0
    }

    # Initialize model
    model = CharacterInformationExtractionModel(bert_preprocess_model,
                                                len(character_types_dict.keys()),
                                                len(actions_dict.keys()),
                                                len(character_present_dict.keys()))

    return model, spacy_model, bert_tokenizer, tokenizer


character_model, spacy_model, bert_tokenizer, tokenizer = init()

# Dummy inference to set the variables and allow the tensorflow to load weights
inference('Little Coraline wandered around the woods')

# Load model weights
character_model.load_weights('Weights/Action Model Weights/weights.h5')
