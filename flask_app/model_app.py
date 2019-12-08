import base64
import numpy as np
import pandas as pd 
from ast import literal_eval
import io
import tensorflow
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, Dense, LSTM, MaxPooling1D, Input, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
    global model
    model = load_model('INPUT MODEL FILENAME')
    print(" MODEL LOADED")

def text_padder(text, tokenizer):
    return sequence.pad_sequences(tokenizer.texts_to_sequences(text), maxlen=250)

def train_tokenizer(train_data, vocab_size):
    # Use Keras tokenizer to create vocabulary dictionary 
    # default arguments will filter punctuation and convert to lower, we do not want this given our use 
    # of pre-trained word embeddings
    tokenizer = text.Tokenizer(num_words = vocab_size, filters='', lower=False)
    tokenizer.fit_on_texts(train_data)
    return tokenizer

def load_training_data():
    global training_data
    training_data = pd.read_csv('../data/train_for_nn.csv',converters={"comment_text_clean2": literal_eval)
    training_data = training_data['comment_text_clean2']
    return training_data

print("Loading Keras Model...")
get_model()
print("Loading training data")
train_data = load_training_data()
print("Training tokenizer")
tokenizer = train_tokenizer(train_data, 200000)

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    comment = message['comment']
    processed_comment = text_padder(comment, tokenizer)

    prediction = model.predict(processed_comment).tolist()

    response = {
        'prediction' : {
            non-toxic: prediction[0][0]
            toxic: prediction[0][1]

        }
    }
    return jsonify(response)
    
