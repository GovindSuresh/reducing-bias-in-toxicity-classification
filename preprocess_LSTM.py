
# Text Preprocessing for LSTM model. 
# The code is based off the experimentation done in the EDA.ipynb and Preprocessing.ipynb.
# Please see these notebooks to read more about the cleaning/preprocessing steps for this exercise.
# This script takes in the training dataset for this project and outputs the processed dataset to be used for the LSTM model.

#Imports
import pandas as pd 
import numpy as np 
import re
import string
import argparse

def load_embed(file):
    '''
    Function to load in the embedding file and get the words contained
    
    INPUT:
    file - filepath with word embedding txt file. Needs to be formatted like GLOVE embedding files 
    i.e word<whitespace>[coefficients]
    
    OUTPUT:
    embedding_dict - dictionary with words from the embedding file as the key
    '''
    # helper function to get word embeddings out of the file
    # *arr collects all the positional arguments passed in as a tuple object
    def get_words(word,*arr):
        # we return the word, and an array containing only the first element of the line which is the word.
        return word, np.asarray(arr, dtype='float16')[:1]
    
    # we create a dictionary containing or words here
    # note how we call get_words(*line.strip().....)
    # the asterix here UNPACKS a tuple/list passed into positional arguments.
    # so for each line of the file we return the word and the first coefficient that goes with it.
    embedding_dict = dict(get_words(*line.strip().split(" ")) for line in open(file))
        
    return embedding_dict

def build_vocab(texts):
    '''
    Function that reads through data and builds out a dictionary containing every word and its word count
    
    INPUT:
    texts - Series object of strings
    
    OUTPUT:
    vocab - dictionary of unique words contained in the texts
    '''
    # split each row into individual words
    sentences = texts.apply(lambda x: x.split()).values
    
    vocab = {}
    # for each row
    for sentence in sentences:
        # for each token in the row
        for word in sentence:
            try:
                # increase wordcount by 1
                vocab[word] += 1
            except KeyError:
                # add word to dictionary and set wordcount to 1
                vocab[word] = 1
    return vocab


#Function to do the pre-processing

def text_preprocess(df, col_name, clean_col_name):
    '''
    Carries out the text pre-processing for the LSTM model. Changes are done in place.

    INPUTS:
    df: Dataframe with data
    col_name: text column
    clean_col_name: Name of new column to hold the clean data 
    '''

    # Build set of symbols to remove from our dataset as there are no word-embeddings for them
    # We are also isolating various latin_based_charachters which are not in the english alphabet
    non_symbols = string.digits + string.ascii_letters
    latin_based_char = 'ÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ'
    non_symbols = string.digits + string.ascii_letters + latin_based_char

    #Load in glove file and build dataset vocab
    glove = load_embed(GLOVE_FILE)
    vocab = build_vocab(df[col_name])

    # The below lines of code create strings of words in our vocab and then isolate the symbols only as those are what we want to remove via a filter
    clean_vocab_chars = ''.join([char for char in vocab if len(char) == 1])
    clean_vocab_symbols = ''.join([char for char in clean_vocab_chars if not char in non_symbols])
    # Need a list of symbols which have a word embedding
    glove_char = ''.join([char for char in glove if len(char) == 1])
    glove_symbols = ''.join([char for char in glove_char if not char in non_symbols])
    # we want to drop the symbols for which there isnt a Glove embedding
    drop_symbol = ''.join([char for char in clean_vocab_symbols if char not in glove_symbols])

    # Use string translate to remove the symbols identified to be drop
    symb_table = str.maketrans('', '', drop_symbol)
    df[clean_col_name] = df[col_name].apply(lambda x: x.translate(symb_table))

    #Removing possessive apostrophes 
    df[clean_col_name] = df[clean_col_name].apply(lambda x: re.sub("'s?", " ", x))

    #splitting punctuation off the end of words.
    df[clean_col_name] = df[clean_col_name].apply(lambda x: re.sub('([!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~])', r' \1 ', x))
    df[clean_col_name] = df[clean_col_name].apply(lambda x: re.sub('\s{2,}', ' ', x))
    
    return None

# Function to remove null values from dataset and drop uneeded columns:
def dataframe_cleaner(df):
    '''
    This function removes the null values from our dataset as decided in the EDA.ipynb
    INPUTS: df - Dataframe
    OUTPUTS: dataframe with nulls removed
    '''

    #Colums to keep 
    cols_to_keep = ['id', 'target', 'comment_text',
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

    # drop all columns not in above list
    df.drop([i for i in df.columns if i not in cols_to_keep], axis=1, inplace=True)

    # first is to fill the nulls in the identity column. 
    columns = df.loc[:,'black':'white'].columns
    for i in columns:
        df[i].fillna(0, inplace=True)

    #binarize target column and identity columns
    cols_to_binarize = df.drop(['id','comment_text'], axis=1).columns
    
    def convert_df_to_binary(df, cols):
        bin_df = df.copy()
        for col in cols :
             bin_df[col] = np.where(df[col] >= 0.5, 1, 0)
        
        return bin_df

    df = convert_df_to_binary(df, cols_to_binarize)

    return df


#set up parser
def set_args():
    parser = argparse.ArgumentParser(description='LSTM preprocess script')
    parser.add_argument('data_file', type=str, help='Filepath of the data csv file')
    parser.add_argument('text_col', type=str, help='Text column in dataframe')
    parser.add_argument('text_col_clean',  type=str, help='Name of cleaned column')
    parser.add_argument('cleaned_file_destination',  type=str, help='Filepath to save the cleaned csv (includes file_name)- Has to be an existing directory')
    parser.add_argument('embedding_file', type=str, help='Filepath of word embeddings')

    return parser.parse_args()

# Main app

if __name__ == '__main__':
    
    args = set_args()

    #load data
    DATA_FILE = args.data_file
    TEXT_COL = args.text_col
    TEXT_COL_CLEAN = args.text_col_clean
    CLEANED_FILE = args.cleaned_file_destination
    GLOVE_FILE = args.embedding_file

    print("Loading Data...")
    
    df = pd.read_csv(DATA_FILE)
    
    print("Processing file...")
    
    #Drop uneeded columns fill nulls and binarize columns
    df = dataframe_cleaner(df)

    #Run text preprocessing function
    text_preprocess(df, TEXT_COL, TEXT_COL_CLEAN)
    
    print("Finished processing file...")
    print("Saving File...")
    
    # Save down cleaned file
    df.to_csv(CLEANED_FILE)
    print(f'SAVED FILE TO:{CLEANED_FILE}')
