import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = 'model/sentiment_model.keras'
TOKENIZER_PATH = 'model/tokenizer.pkl'

model = None
tokenizer = None


def init():
    """
    Loads a pre-trained model and tokenizer
    :return: tuple: (model, tokenizer)
            model (tf.keras.Model): model for sentiment analysis
            tokenizer (Tokenizer): tokenizer
    """
    global model, tokenizer
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Model and tokenizer loaded")
    return model, tokenizer


def predict_sentiment(text):
    """
    Predicts the sentiment class of a text and the confidence of the prediction
    :param text: input text for analysis
    :return: tuple: (class_idx, confidence)
            class_idx (int): class index (0 - negative, 1 - neutral, 2 - positive)
            confidence (float): model probability for a given class
    """
    if model is None or tokenizer is None:
        raise Exception("Model and tokenizer are not loaded. Call init() first.")
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=27, padding='post', truncating='post')
    preds = model.predict(padded)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_idx]
    return class_idx, confidence
