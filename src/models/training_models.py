import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, InputLayer, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from transform_data import load_data, get_close_price, build_window_matrix, train_test_val
load_dotenv()


WD = os.getenv('working_directory')

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
    model = Sequential()
    model.add(InputLayer(input_shape=(5, 1)))  # Spécification explicite de input_shape
    model.add(LSTM(256))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

def prepare_checkpoint():
    """
    """
    return ModelCheckpoint(f'{WD}/src/models/models_cp/', save_best_only=True,save_format='tf')

def train(model):
    """
    """
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    return model

def fit(model, cp, X_train, y_train, X_val, y_val):
    """
    """
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

def load():
    """
    """
    return load_model(f'{WD}/src/models/models_cp/')

def predict(model, X_train):
    """
    """
    return model.predict(X_train).flatten()

def prediction(X_train):
    model=load()
    print(predict(model, X_train))

def main():

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    print(X_train)
    cp = prepare_checkpoint()
    model=prepare_model()
    model=train(model)
    fit(model, cp, X_train, y_train, X_val, y_val)

    prediction(X_train)

main()