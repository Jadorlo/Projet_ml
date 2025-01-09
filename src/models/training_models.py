import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from keras.api.layers import LSTM, InputLayer, Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.metrics import RootMeanSquaredError
from dotenv import load_dotenv
from transform_data import load_data, get_close_price, build_window_matrix, train_test_val
load_dotenv()

print(tf.__version__)

wd = os.getenv('working_directory')

def prepare_data():
    """
    Renvoie X_train, y_train, X_val, y_val, X_test, y_test pour le df
    """
    WINDOW_SIZE = 5
    df = load_data("raw")
    close = get_close_price(df)
    X, y = build_window_matrix(close, window_size=WINDOW_SIZE)
    return train_test_val(X, y)

def prepare_model():
    """
    Set up le model
    """
    model=Sequential()
    model.add(LSTM(64, input_shape=(5,1)))
    model.add(Dense(10, 'relu'))
    model.add(Dense(1, 'sigmoid'))

    print(model.summary())

prepare_model()