# regular import
import pandas as pd
import numpy as np
from ast import literal_eval
import datetime, os

#tensorflow imports
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, Dense, LSTM, Input, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizer import Adam


# File Parameters
TRAIN_TEXT_COL = 'comment_text_clean2'
TEST_TEXT_COL = 'comment_text_clean2'
TRAIN_TARGET_COL = 'target'
TEST_TARGET_COL = 'target'
EMBEDDING_FILE = 'embeds/glove.840B.300d.txt' #change this if you are using a different embedding file
CHECKPOINT_PATH = "NN_models/cp.ckpt"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)


## Text related parameters
MAX_VOCAB_SIZE = 200000 # there are 563693 words in the vocabulary 
MAX_LEN_SEQ = 300
EMBED_DIM = 300 #change this if you have chose different embedding dimensions

# Model hyperparameters
DROPOUT_RATE = 0.2
LSTM_UNITS = 128
BATCH_SIZE = 128
NUM_EPOCHS = 4
LEARNING_RATE = 0.001


### MODEL FUNCTIONS ####

# TOKENIZER:

def train_tokenizer(train_data, vocab_size):

    tokenizer = text.tokenizer(num_words = MAX_VOCAB_SIZE, filters='', split=' ', lower=False)
    tokenizer.fit_on_texts(train_data)

    return tokenizer

# PAD SEQUENCES:

def text_padder(text, tokenizer):

    return sequence.pad_sequences(tokenizer.texts_to_sequences(text), maxlen=MAX_LEN_SEQ)


def build_embedding_matrix(tokenizer_word_index, EMBEDDING_FILE):

    embedding_dict = {}


    with open(EMBEDDING_FILE) as file:

        for line in file:
            line.split(' ') # each word and vector is seperated by a whitespace

            word = line[0] #word is first part of the line
            word_vec = line[1:] #rest of line is word vector

            #convert the word_vector to numpy_array
            word_vec = np.asarray(word_vec, dtype=np.float32)

            embedding_dict[word] = word_vec

    # Now we build the embedding matrix for our vocabulary OOV words will be assigned 0

    embedding_matrix = np.zeroes((len(tokenizer_word_index)+1, EMBED_DIM))


    for word, i in tokenizer_word_index.items():
        
        # checks if word index is outide max vocab size. If true we just continue
        if i >= MAX_VOCAB_SIZE:
            continue
        
        # gets the vector to the corresponding word from the previous dictionary and sets it to the variable
        embedding_vector = embedding_dict.get(word)
        # We check whether the embedding_vector is not none (i.e the word was in the original embedding file)
        if embedding_vector is not None:
            # Append the embedding vector to index i in the embedding matrix. 
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix

def build_model(embedding_matrix, tokenizer):

    input_words = Input(shape=(MAX_LEN_SEQ,), dtype='int32')

    embedding = Embedding(len(tokenizer.word_index)+1, EMBED_DIM,
                          weights=[embedding_matrix],
                          input_length = MAX_LEN_SEQ,
                          #mask_zero = True
                          trainable = False) (input_words)

    x = Dropout(DROPOUT_RATE)(embedding)

    x = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(x)

    x = GlobalMaxPooling1D()(x)

    x = Dense(462, activation='relu')(x)

    prediction = Dense(1, activation='sigmoid')(x)

    opt = Adam(lr=LEARNING_RATE)

    model = Model(inputs=input_words, outputs=prediction, name='baseline-LSTM')

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', AUC()])

    return model

def lr_scheduler(epoch):
    LR_START = LEARNING_RATE

    #set lr drop
    LR_DROP = 0.5
    EPOCHS_DROP = 1
    #we are going to use step decay here
    lr = LR_START * LR_DROP**np.floor(epoch / EPOCHS_DROP)

    return lr

def train_model(train_df, val_df, tokenizer):
    '''
    Function to train a tensorflow model. This function runs a number of the previous functions
    
    INPUT:
    train_df - training data
    val_df - validation data
    tokenizer - fitted keras tokenizer
    
    OUTPUT:
    model - tensorflow model
    fitted_model - fit history
    '''
    # Create processed and padded train and targets
    print('padding_text')
    X_train = text_padder(train_df[TRAIN_TEXT_COL], tokenizer)
    X_val = text_padder(val_df[TRAIN_TEXT_COL], tokenizer)
    y_train = train_df[TRAIN_TARGET_COL]
    y_val = val_df[TRAIN_TARGET_COL])
    
    print('building embedding matrix')
    # build embedding matrix
    embed_matrix = build_embedding_matrix(tokenizer.word_index, EMBEDDING_FILE)
    
    # build model
    print('building model')
    model = build_model(embed_matrix, tokenizer)
    
    # set up checkpoint callbacks
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
   
    # train model batch size and epochs were set up earlier.
    print('training model')
    fitted_model = model.fit(X_train, y_train,
                             batch_size = BATCH_SIZE,
                             epochs = NUM_EPOCHS,
                             validation_data=(X_val, y_val),
                             callbacks=[cp_callback],
                             verbose = 1)

    return model, fitted_model




