import tensorflow as tf
import numpy as np
import nltk

from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Concatenate
from tensorflow.python.ops import array_ops

BATCH_SIZE = 2
MAX_DOC_LENGTH = 12
MAX_PARA_LENGTH = 32
MAX_SENT_LENGTH = 255
EMBEDDING_DIM = 100


def categorical_labels(labels):
    eye = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    
    result = []
    for item in labels:
        result.append(eye[int(item) - 1])
        
    return np.array(result)


def dense_mask(M, sent_len=MAX_SENT_LENGTH, para_len=MAX_PARA_LENGTH, doc_len=MAX_DOC_LENGTH):
    Z = np.zeros((len(M), doc_len, para_len, sent_len), dtype=bool)
    for docidx, doc in enumerate(M):
        for paraidx, para in enumerate(doc):
            for sentidx, sent in enumerate(para):
                for tokenidx, token in enumerate(sent):
                    Z[docidx, paraidx, sentidx, tokenidx] = True
    return Z


def pad_to_dense(M, sent_len=MAX_SENT_LENGTH, para_len=MAX_PARA_LENGTH, doc_len=MAX_DOC_LENGTH):
    Z = np.zeros((len(M), doc_len, para_len, sent_len))
    for docidx, doc in enumerate(M):
        for paraidx, para in enumerate(doc):
            for sentidx, sent in enumerate(para):
                sentnp = np.hstack(np.array(sent))
                Z[docidx, paraidx, sentidx, :len(sentnp)] += sentnp
    return Z


def tokenize(text, tok=None):
    return [
        [
            tok.texts_to_sequences(nltk.word_tokenize(sent))
            if tok else
            nltk.word_tokenize(sent)
            for sent in nltk.sent_tokenize(para)
        ]
        for para in text.splitlines()
        if len(para) > 0
    ]


class BahdanauAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttentionLayer, self).__init__(**kwargs)
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, values):
        # (batch_size, max_length, 1)
        scores = self.V(tf.nn.tanh(self.W(values)))
        
        # (batch_size, max_length, 1) normalized lulz
        attention_weights = tf.nn.softmax(scores, axis=1)
        
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)
        
        return context_vector, attention_weights


class AttentiveSequenceEncoder(tf.keras.layers.Layer):
    def __init__(self, lstm_units, attention_units, **kwargs):
        super().__init__(**kwargs)
        self.lstm = Bidirectional(LSTM(lstm_units, recurrent_dropout=0.0001, return_sequences=True))
        self.concat = Concatenate()
        self.attention = BahdanauAttentionLayer(attention_units)
        
    def call(self, inputs, mask):
        encoded = self.lstm(inputs, mask=mask)
        output, attention_weights = self.attention(encoded)
        
        return output, attention_weights

    
class DocModel(tf.keras.Model):
    def __init__(self, lstm_units, hidden_units, dropout, embedding_matrix, vocab_size, batch_size=BATCH_SIZE):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            trainable=False,
            input_shape=(MAX_DOC_LENGTH, MAX_PARA_LENGTH, MAX_SENT_LENGTH))
        
        self.sent_encoder = Bidirectional(LSTM(lstm_units, recurrent_dropout=0.0001))
        self.para_encoder = Bidirectional(LSTM(lstm_units, recurrent_dropout=0.0001))
        self.doc_encoder = Bidirectional(LSTM(lstm_units, recurrent_dropout=0.0001))
        
        self.hidden = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.classifier = tf.keras.layers.Dense(3, activation='sigmoid')
        
        self.dropout.build((BATCH_SIZE, hidden_units))
    
    def call(self, inputs, training=False):
        (inputs, sent_mask, para_mask, doc_mask) = inputs
        
        embedded = self.embedding(inputs)
        embedded = array_ops.reshape(
            embedded, (BATCH_SIZE * MAX_DOC_LENGTH * MAX_PARA_LENGTH, MAX_SENT_LENGTH, -1))
        sent_mask = array_ops.reshape(
            sent_mask, (BATCH_SIZE * MAX_DOC_LENGTH * MAX_PARA_LENGTH, MAX_SENT_LENGTH))

        sent_embedded = self.sent_encoder(embedded, mask=sent_mask)
        sent_embedded = array_ops.reshape(
            sent_embedded, (BATCH_SIZE * MAX_DOC_LENGTH, MAX_PARA_LENGTH, -1))
        para_mask = array_ops.reshape(
            para_mask, (BATCH_SIZE * MAX_DOC_LENGTH, MAX_PARA_LENGTH))
        
        para_embedded = self.para_encoder(sent_embedded, mask=para_mask)
        para_embedded = array_ops.reshape(
            para_embedded, (BATCH_SIZE, MAX_DOC_LENGTH, -1))
        
        x = self.doc_encoder(para_embedded, mask=doc_mask)
        x = self.hidden(x)
        x = self.dropout(x)
        
        return self.classifier(x)
    

class AttentiveDocModel(tf.keras.Model):
    def __init__(self, lstm_units, hidden_units, attention_units, dropout, embedding_matrix, vocab_size, batch_size=BATCH_SIZE):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            trainable=False,
            input_shape=(MAX_DOC_LENGTH, MAX_PARA_LENGTH, MAX_SENT_LENGTH))
        
        self.sent_encoder = AttentiveSequenceEncoder(lstm_units, attention_units)
        self.para_encoder = AttentiveSequenceEncoder(lstm_units, attention_units)
        self.doc_encoder = AttentiveSequenceEncoder(lstm_units, attention_units)
        
        self.hidden = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.classifier = tf.keras.layers.Dense(3, activation='sigmoid')
        
        self.dropout.build((BATCH_SIZE, hidden_units))
    
    def call(self, inputs, training=False):
        (inputs, sent_mask, para_mask, doc_mask) = inputs
        
        embedded = self.embedding(inputs)
        embedded = array_ops.reshape(
            embedded, (BATCH_SIZE * MAX_DOC_LENGTH * MAX_PARA_LENGTH, MAX_SENT_LENGTH, -1))
        sent_mask = array_ops.reshape(
            sent_mask, (BATCH_SIZE * MAX_DOC_LENGTH * MAX_PARA_LENGTH, MAX_SENT_LENGTH))

        sent_embedded, sent_weights = self.sent_encoder(embedded, mask=sent_mask)
        sent_embedded = array_ops.reshape(
            sent_embedded, (BATCH_SIZE * MAX_DOC_LENGTH, MAX_PARA_LENGTH, -1))
        para_mask = array_ops.reshape(
            para_mask, (BATCH_SIZE * MAX_DOC_LENGTH, MAX_PARA_LENGTH))
        
        para_embedded, para_weights = self.para_encoder(sent_embedded, mask=para_mask)
        para_embedded = array_ops.reshape(
            para_embedded, (BATCH_SIZE, MAX_DOC_LENGTH, -1))
        
        x, doc_weights = self.doc_encoder(para_embedded, mask=doc_mask)
        x = self.hidden(x)
        x = self.dropout(x)
        
        if not training:
            self.sent_weights = sent_weights
            self.para_weights = para_weights
            self.doc_weights = doc_weights
        
        return self.classifier(x)
    
    
class SmallDocModel(tf.keras.Model):
    def __init__(self, lstm_units, hidden_units, dropout, embedding_matrix, vocab_size, batch_size=BATCH_SIZE):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            trainable=False,
            input_shape=(MAX_DOC_LENGTH, MAX_PARA_LENGTH, MAX_SENT_LENGTH))
        
        self.sent_encoder = Bidirectional(LSTM(lstm_units, recurrent_dropout=0.0001))
        self.doc_encoder = Bidirectional(LSTM(lstm_units, recurrent_dropout=0.0001))
        
        self.hidden = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.classifier = tf.keras.layers.Dense(3, activation='sigmoid')
        
        self.dropout.build((BATCH_SIZE, hidden_units))
    
    def call(self, inputs, training=False):
        (inputs, sent_mask, para_mask, doc_mask) = inputs
        
        embedded = self.embedding(inputs)
        embedded = array_ops.reshape(
            embedded, (BATCH_SIZE * MAX_DOC_LENGTH * MAX_PARA_LENGTH, MAX_SENT_LENGTH, -1))
        sent_mask = array_ops.reshape(
            sent_mask, (BATCH_SIZE * MAX_DOC_LENGTH * MAX_PARA_LENGTH, MAX_SENT_LENGTH))

        sent_embedded = self.sent_encoder(embedded, mask=sent_mask)
        sent_embedded = array_ops.reshape(
            sent_embedded, (BATCH_SIZE, MAX_DOC_LENGTH * MAX_PARA_LENGTH, -1))
        doc_mask = array_ops.reshape(
            para_mask, (BATCH_SIZE, MAX_DOC_LENGTH * MAX_PARA_LENGTH))
        
        x = self.doc_encoder(sent_embedded, mask=doc_mask)
        x = self.hidden(x)
        x = self.dropout(x)
        
        return self.classifier(x)
    

class SmallAttentiveDocModel(tf.keras.Model):
    def __init__(self, lstm_units, hidden_units, attention_units, dropout, embedding_matrix, vocab_size, batch_size=BATCH_SIZE):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            trainable=False,
            input_shape=(MAX_DOC_LENGTH, MAX_PARA_LENGTH, MAX_SENT_LENGTH))
        
        self.sent_encoder = AttentiveSequenceEncoder(lstm_units, attention_units)
        self.doc_encoder = AttentiveSequenceEncoder(lstm_units, attention_units)
        
        self.hidden = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.classifier = tf.keras.layers.Dense(3, activation='sigmoid')
        
        self.dropout.build((BATCH_SIZE, hidden_units))
    
    def call(self, inputs, training=False):
        (inputs, sent_mask, para_mask, doc_mask) = inputs
        
        embedded = self.embedding(inputs)
        embedded = array_ops.reshape(
            embedded, (BATCH_SIZE * MAX_DOC_LENGTH * MAX_PARA_LENGTH, MAX_SENT_LENGTH, -1))
        sent_mask = array_ops.reshape(
            sent_mask, (BATCH_SIZE * MAX_DOC_LENGTH * MAX_PARA_LENGTH, MAX_SENT_LENGTH))

        sent_embedded, sent_weights = self.sent_encoder(embedded, mask=sent_mask)
        sent_embedded = array_ops.reshape(
            sent_embedded, (BATCH_SIZE, MAX_DOC_LENGTH * MAX_PARA_LENGTH, -1))
        doc_mask = array_ops.reshape(
            para_mask, (BATCH_SIZE, MAX_DOC_LENGTH * MAX_PARA_LENGTH))
        
        x, doc_weights = self.doc_encoder(sent_embedded, mask=doc_mask)
        x = self.hidden(x)
        x = self.dropout(x)
        
        if not training:
            self.sent_weights = sent_weights
            self.doc_weights = doc_weights
        
        return self.classifier(x)

