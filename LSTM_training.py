# regular import
import pandas as pd
import numpy as np
import datetime, os
import matplotlib.pyplot as plt

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
from tensorflow.keras.optimizer import Adam


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

    tokenizer = text.tokenizer(num_words = MAX_VOCAB_SIZE, filters='', split=' ', lower=False)
    tokenizer.fit_on_texts(train_data)

    return tokenizer

# PAD SEQUENCES:

def text_padder(text, tokenizer):

    return sequence.pad_sequences(tokenizer.texts_to_sequences(text), maxlen=MAX_LEN_SEQ, padding='post')


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
    print('padding text')
    X_train = text_padder(train_df[TRAIN_TEXT_COL], tokenizer)
    X_val = text_padder(val_df[TRAIN_TEXT_COL], tokenizer)
    y_train = train_df[TRAIN_TARGET_COL]
    y_val = val_df[TRAIN_TARGET_COL]
    
    print('building embedding matrix')
    # build embedding matrix
    embed_matrix = build_embedding_matrix(tokenizer.word_index, EMBEDDING_FILE)
    
    # build model
    print('building model')
    model = build_model(embed_matrix, tokenizer, model_name)
    
    # set up checkpoint callbacks
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
   
    # train model batch size and epochs were set up earlier.
    print('training model')
    model_hist = model.fit(X_train, y_train,
                             batch_size = BATCH_SIZE,
                             epochs = NUM_EPOCHS,
                             validation_data=(X_val, y_val),
                             callbacks=[cp_callback, lr_schedule],
                             verbose = 1)

    return model, model_hist

def process_test_data(test_data, tokenizer):

    return text_padder(test_data, tokenizer)

    
def model_evaluation(test_preds, test_labels, test_df, model_name):
    '''
    Evaluates the models against accuracy, F1 Score, and the final weighted bias metric as discussed in the project write-up
    Inputs:
    Array: test_preds - Predicted labels for the test set
    Array: test_labels - True labels for the test set 
    Dataframe:  test_df - test dataframe, needed for the final bias metric calculation

    Outputs:
    nn_bias_metrics_df_test = dataframe showing bias metric scores against the specific identify subgroups
    results_df = dataframe of metrics for the model 
        
    '''
    # Define subgroup metrics
    SUBGROUP_AUC = 'subgroup_auc'
    BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
    BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

    # These calculations have been provided by Jigsaw AI for scoring based on the metrics of the kaggle competition
    # https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation

    # They work by filtering the relevant dataframe into specific subgroups and using the roc_auc_score metric from sklearn.

    def compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def compute_subgroup_auc(df, subgroup, label, model_name):
        subgroup_examples = df[df[subgroup]]
        return compute_auc(subgroup_examples[label], subgroup_examples[model_name])

    def compute_bpsn_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
        subgroup_negative_examples = df.loc[df[subgroup] & ~df[label]]
        non_subgroup_positive_examples = df.loc[~df[subgroup] & df[label]]
        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        return compute_auc(examples[label], examples[model_name])

    def compute_bnsp_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
        subgroup_positive_examples = df.loc[df[subgroup] & df[label]]
        non_subgroup_negative_examples = df.loc[~df[subgroup] & ~df[label]]
        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        return compute_auc(examples[label], examples[model_name])

    def compute_bias_metrics_for_model(dataset,
                                    subgroups,
                                    model,
                                    label_col,
                                    include_asegs=False):
        """Computes per-subgroup metrics for all subgroups and one model."""
        records = []
        for subgroup in subgroups:
            record = {
                'subgroup': subgroup,
                'subgroup_size': len(dataset.loc[dataset[subgroup]])
            }
            record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
            record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
            record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
            records.append(record)
        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

    def calculate_overall_auc(df, model_name):
        true_labels = df[TEST_TARGET_COL]
        predicted_labels = df[model_name]
        return roc_auc_score(true_labels, predicted_labels)

    def power_mean(series, p):
        total = sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
        bias_score = np.average([
            power_mean(bias_df[SUBGROUP_AUC], POWER),
            power_mean(bias_df[BPSN_AUC], POWER),
            power_mean(bias_df[BNSP_AUC], POWER)
        ])
        return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)
    
    # Binarize the y_predictions:

    test_preds_bin = np.where(test_preds >=0.5, 1, 0)

    # Compute standard metrics 
    model_accuracy = accuracy_score(test_labels,test_preds_bin)
    model_precision = precision_score(test_labels, test_preds_bin)
    model_recall = recall_score(test_labels,test_preds_bin)
    model_f1 = f1_score(test_labels, test_preds_bin)

    # For the bias metrics we need the additional identity columns from the test_dataframe. Therefore merge
    # Test predictions back to dataframe.

    test_results = test_df.merge(test_preds_bin, how='inner', on='id')

    # Compute the bias metrics dataframe and final weighted score
    nn_bias_metrics_df_test = compute_bias_metrics_for_model(test_results, IDENTITY_COLUMNS, model_name, TEST_TARGET_COL)
    nn_final_metric_test = get_final_metric(nn_bias_metrics_df_test, calculate_overall_auc(test_results, model_name))

    results_df = pd.DataFrame(data=[[model_accuracy, model_precision, model_recall, model_f1, nn_final_metric_test]], index=[model_name], 
                                columns=['model_accuracy','model_precision', 'model_recall', 'model_f1', 'final_weighted_bias'],
                            )

    return nn_bias_metrics_df_test, results_df

        



