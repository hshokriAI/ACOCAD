import pickle
import numpy as np
import json
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from keras.utils import pad_sequences

my_file = open('train_data.json', 'r', errors='ignore')
question_file = json.load(my_file)
question_oe = question_file['questions']
squestion_all_train = [i['question'] for i in question_oe]

my_file = open('valid_data.json', 'r', errors='ignore')
question_file = json.load(my_file)
question_oe_valid = question_file['questions']
question_all_valid = [i['question'] for i in question_oe_valid]

squestion_all = squestion_all_train + question_all_valid
# Limit on the number of features to K features.
TOP_K = 20000
EMBEDDING_VECTOR_LENGTH = 300
# Limit on the length of text sequences.
# Sequences longer than this will be truncated.
# and less than it will be padded
MAX_SEQUENCE_LENGTH = 14


class CustomTokenizer:
    def __init__(self, train_texts):
        self.train_texts = train_texts
        self.tokenizer = Tokenizer(num_words=TOP_K)

    def train_tokenize(self):
        # Get max sequence length.
        max_length = len(max(self.train_texts, key=len))
        self.max_length = min(max_length, MAX_SEQUENCE_LENGTH)

        # Create vocabulary with training texts.
        self.tokenizer.fit_on_texts(self.train_texts)

    def vectorize_input(self, tweets):
        # Vectorize training and validation texts.

        tweets = self.tokenizer.texts_to_sequences(tweets)
        # Fix sequence length to max value. Sequences shorter than the length are
        # padded in the beginning and sequences longer are truncated
        # at the beginning.
        tweets = pad_sequences(tweets, maxlen=self.max_length, truncating='post', padding='post')
        return tweets


def construct_embedding_matrix(glove_file, word_index):
    embedding_dict = {}
    with open(glove_file, 'r') as f:
        for line in f:
            values = line.split()
            # get the word
            word = values[0]
            if word in word_index.keys():
                vector = np.asarray(values[1:], 'float32')
                embedding_dict[word] = vector

    num_words = len(word_index) + 1
    # initialize it to 0
    embedding_matrix = np.zeros((num_words, EMBEDDING_VECTOR_LENGTH))

    for word, i in tqdm(word_index.items()):
        if i < num_words:
            vect = embedding_dict.get(word, [])
            if len(vect) > 0:
                embedding_matrix[i] = vect[:EMBEDDING_VECTOR_LENGTH]
    return embedding_matrix


def construct_embedding_matrix2(glove_file, word_index):
    embedding_dict = {}
    with open(glove_file, 'rb') as f:
        embedding_word = pickle.load(f)

        for word in embedding_word[0]:
            index = embedding_word[0][word]
            values = embedding_word[1][index]
            vector = np.asarray(values, 'float32')
            embedding_dict[word] = vector

    num_words = len(word_index) + 1
    # initialize it to 0
    embedding_matrix = np.zeros((num_words, EMBEDDING_VECTOR_LENGTH))

    for word, i in tqdm(word_index.items()):
        if i < num_words:
            vect = embedding_dict.get(word, [])
            if len(vect) > 0:
                embedding_matrix[i] = vect[:EMBEDDING_VECTOR_LENGTH]
    return embedding_matrix


def emd():
    tokenizer = CustomTokenizer(train_texts=squestion_all)
    # fit on the train
    tokenizer.train_tokenize()
    embedding_matrix = construct_embedding_matrix('glove.6B.300d.txt', tokenizer.tokenizer.word_index)
    return embedding_matrix, tokenizer.tokenizer.word_index


def emd_vec(txt):
    tokenizer = CustomTokenizer(train_texts=squestion_all)
    # fit o the train
    tokenizer.train_tokenize()
    tokenized_vec = tokenizer.vectorize_input(txt)
    return tokenized_vec
