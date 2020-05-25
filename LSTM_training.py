# regular import
import pandas as pd
import numpy as np
import datetime, os
import matplotlib.pyplot as plt
import yaml
import argparse
import json
import io

#sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

#tensorflow imports
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, Dense, LSTM, Input, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam


# File parameters
TRAIN_TEXT_COL = 'comment_text_clean2'
TEST_TEXT_COL = 'comment_text_clean2'
TRAIN_TARGET_COL = 'target'
TEST_TARGET_COL = 'target'
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'] # Don't change these unless you have edited the preprocessing script to include more
EMBEDDING_FILE = 'embeds/glove.840B.300d.txt' #change this if you are using a different embedding file
CHECKPOINT_PATH = "NN_models/cp.ckpt"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

# General parameters



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

    tokenizer = text.Tokenizer(num_words = MAX_VOCAB_SIZE, filters='', split=' ', lower=False)
    tokenizer.fit_on_texts(train_data)

    return tokenizer

# PAD SEQUENCES:

def text_padder(text, tokenizer):

    return sequence.pad_sequences(tokenizer.texts_to_sequences(text), maxlen=MAX_LEN_SEQ, padding='post')


def build_embedding_matrix(tokenizer_word_index, EMBEDDING_FILE):

    embedding_dict = {}


    with open(EMBEDDING_FILE) as file:

        for line in file:
            line = line.split(' ') # each word and vector is seperated by a whitespace

            word = line[0] #word is first part of the line
            word_vec = line[1:] #rest of line is word vector

            #convert the word_vector to numpy_array
            word_vec = np.asarray(word_vec, dtype=np.float32)

            embedding_dict[word] = word_vec

    # Now we build the embedding matrix for our vocabulary OOV words will be assigned 0

    embedding_matrix = np.zeros((len(tokenizer_word_index)+1, EMBED_DIM))


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

def build_model(embedding_matrix, tokenizer, model_name):

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

    model = Model(inputs=input_words, outputs=prediction, name=model_name)

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

def train_model(train_df, val_df, tokenizer, model_name):
    '''
    Function to train a tensorflow model. This function runs a number of the previous functions
    
    INPUT:
    DF: train_df - training data
    DF: val_df - validation data
    OBJ: tokenizer - fitted keras tokenizer
    STR: model_name - Name for model 
    
    OUTPUT:
    OBJ: model - tensorflow model
    OBJ: model_hist - fit history
    '''
    # Create processed and padded train and targets
    print('Padding text...\n')
    X_train = text_padder(train_df[TRAIN_TEXT_COL], tokenizer)
    X_val = text_padder(val_df[TRAIN_TEXT_COL], tokenizer)
    y_train = np.asarray(train_df[TRAIN_TARGET_COL])
    y_val = np.asarray(val_df[TRAIN_TARGET_COL])
    
    print('Building embedding matrix...\n')
    # build embedding matrix
    embed_matrix = build_embedding_matrix(tokenizer.word_index, EMBEDDING_FILE)
    
    # build model
    print('Building model...\n')
    model = build_model(embed_matrix, tokenizer, model_name)
    
    # set up checkpoint callbacks
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
   
    # train model batch size and epochs were set up earlier.
    print('Training model...\n')
    model_hist = model.fit(X_train, y_train,
                             batch_size = BATCH_SIZE,
                             epochs = NUM_EPOCHS,
                             validation_data=(X_val, y_val),
                             callbacks=[cp_callback, lr_schedule],
                             verbose = 1)

    return model, model_hist

# Set up parser
def set_args():
    parser = argparse.ArgumentParser(description='LSTM training script')
    parser.add_argument('train_file', type=str, help='Filepath of the training csv file')
    parser.add_argument('model_name', type=str, help='Name to give the trained model')
    parser.add_argument('yml_filepath', type=str, help='Location of yml file with model parameters')


    return parser.parse_args()

#app
if __name__ == '__main__':

    

    # Read in YAML file
    stream = open(YAML_FILE, 'r')
    param_dict = yaml.load(stream, Loader=yaml.SafeLoader)
    
    TRAIN_TEXT_COL = param_dict['TRAIN_TEXT_COL']
    TEST_TEXT_COL = param_dict['TEST_TEXT_COL']
    TRAIN_TARGET_COL = param_dict['TRAIN_TARGET_COL']
    TEST_TARGET_COL = param_dict['TEST_TARGET_COL']
    IDENTITY_COLS = param_dict['IDENTITY_COLS']
    
    EMBEDDING_FILE = param_dict['EMBEDDING_FILE']
    EMBEDDING_DIMS = param_dict['EMBEDDING_DIMS']
    MAX_VOCAB_SIZE = param_dict['MAX_VOCAB_SIZE']
    MAX_LEN_SEQ = param_dict['MAX_LEN_SEQ']
    VAL_SIZE = param_dict['VAL_SIZE']

    LSTM_UNITS = param_dict['LSTM_UNITS']
    BATCH_SIZE = param_dict['BATCH_SIZE']
    NUM_EPOCHS = param_dict['NUM_EPOCHS']
    LEARNING_RATE = param_dict['LEARNING_RATE']

    CHECKPOINT_PATH = param_dict['CHECKPOINT_PATH']
    MODEL_SAVE_PATH = param_dict['MODEL_SAVE_PATH']

    # Make directories to save model files now
    # This way if there is an error with file path we dont waste time with training first
    # We aren't handling this error because we want the program to crash if the makedirs fails to force user
    # to put in a correct file name
    
    if not os.path.isdir(os.path.join(MODEL_SAVE_PATH, MODEL_NAME)):
        os.makedirs(os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
    else:
        print('Subdirectory for the chosen model_name exists, model save will overwrite existing files')
    
    if not os.path.isdir(os.path.join(MODEL_SAVE_PATH,MODEL_NAME,'weights')):
        os.makedirs(os.path.join(MODEL_SAVE_PATH, MODEL_NAME, 'weights'))
    else:
        print('Subdirectory for the chosen model_name exists, weights save will overwrite existing files')

    # Load in data
    train_df = pd.read_csv(TRAIN_FILE, index_col=0)
    train_df = train_df.dropna()
    # Train val split
    # Create train val split, stratify on target - random state set to 1 for reproducibility, you can remove this
    train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, stratify=train_df['target'], random_state=1)

    # Train tokenizer
    tokenizer = train_tokenizer(train_df[TRAIN_TEXT_COL], MAX_VOCAB_SIZE)

    # Model training
    model, history = train_model(train_df, val_df, tokenizer, MODEL_NAME)

    # Final model save - saves full model using h5 format -- change
    print(f'Saving model @ {os.path.join(MODEL_SAVE_PATH, MODEL_NAME)}')

    model.save(os.path.join(MODEL_SAVE_PATH, MODEL_NAME,f'{MODEL_NAME}.h5', save_format='h5'))

    # Model weight save
    model.save_weights(os.path.join(MODEL_SAVE_PATH, MODEL_NAME,'weights',f'{MODEL_NAME}_weights.h5', save_format='h5'))

    print('Model saved!')

    # Save the tokenizer with the model

    print(f'Saving Tokenizer @ {os.path.join(MODEL_SAVE_PATH, MODEL_NAME)}' )
    tokenizer_json = tokkenizer.to_json()
    with io.open(os.path.join(MODEL_SAVE_PATH, MODEL_NAME,f'{MODEL_NAME}_tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    
