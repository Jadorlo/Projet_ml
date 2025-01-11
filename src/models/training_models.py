import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from pre_processing import load_data, get_close_price, build_window_matrix, train_test_val, get_date, WINDOW_SIZE
load_dotenv()


WD = os.getenv('working_directory')

def prepare_data():
    """
    Renvoie X_train, y_train, X_val, y_val, X_test, y_test pour le df
    """
    df = load_data("transform_data")
    close = get_close_price(df)
    date = get_date(df)
    X, y, scaler = build_window_matrix(close, window_size=WINDOW_SIZE)
    return train_test_val(X, y, date, train_size=0.8), scaler

def prepare_model():
    """
    Set up le model
    """
    model = Sequential([Input((WINDOW_SIZE, 1)),
                        LSTM(64),
                        Dense(32, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(1)
                        ])
    return model

def prepare_checkpoint():
    """
    """
    return ModelCheckpoint(f'{WD}/src/models/models_cp/', save_best_only=True, save_format='tf', monitor='loss')

def save(model):
    """
    """
    model.save(f'{WD}/src/models/models_cp/', save_format='tf')

def train(model, learning_rate):
    """
    """
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    return model

def fit(model, cp, X_train, y_train, X_val, y_val):
    """
    """
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])
    return model

def load():
    """
    """
    return load_model(f'{WD}/src/models/models_cp/', custom_objects={'LSTM': LSTM})

def predict(model, X):
    """
    """
    return model.predict(X)

def inverse_scaler(scaler, array):
    """
    """
    return scaler.inverse_transform(array)
    

def analyze(date, y, predictions):
    """
    """
    r2 = r2_score(y, predictions)
    print(f'{r2*100}%')
    df_analyze = pd.concat([pd.DataFrame(date), pd.DataFrame(y), pd.DataFrame(predictions)], axis=1)
    df_analyze.columns = ['Date', 'Actuals', 'Predictions']
    return df_analyze
    
def plot(df, is_train):
    """
    """
    plt.clf()
    plt.plot(df['Date'], df['Actuals'], label='Actuals') 
    plt.plot(df['Date'], df['Predictions'], label='Predictions')
    plt.savefig(f'{WD}/figure/{"train_" if is_train else "test_"}comp_act_pred.png')

def main():

    (date_train, X_train, y_train, date_val, X_val, y_val, date_test, X_test, y_test), scaler = prepare_data()
    
    cp = prepare_checkpoint()
    model = prepare_model()
    model = train(model, 0.01)
    model = fit(model, cp, X_train, y_train, X_val, y_val)

    loaded_model = load()

    # #Prediction TRAIN
    # train_predictions = predict(loaded_model, X_train)
    # y_train_price = inverse_scaler(scaler, y_train)
    # train_predictions_price = inverse_scaler(scaler, train_predictions)
    
    # df_analyze = analyze(date_train, y_train_price, train_predictions_price)
    # print("#######TRAIN#######\n",df_analyze)
    # plot(df_analyze,True)

    # #Prediction TEST
    # test_predictions = predict(loaded_model, X_test)
    # y_test_price = inverse_scaler(scaler, y_test)
    # test_predictions_price = inverse_scaler(scaler, test_predictions)
    
    # df_analyze = analyze(date_test, y_test_price, test_predictions_price)
    # print("#######TEST#######\n",df_analyze)
    # plot(df_analyze,False)

    #Prediction TRAIN NO SCALE
    train_predictions = predict(loaded_model, X_train)
    #y_train_price = inverse_scaler(scaler, y_train)
    #train_predictions_price = inverse_scaler(scaler, train_predictions)
    
    df_analyze = analyze(date_train, y_train, train_predictions)
    print("#######TRAIN NO SCALE #######\n",df_analyze)
    plot(df_analyze,True)

    #Prediction TEST NO SCALE
    test_predictions = predict(loaded_model, X_test)
    #y_test_price = inverse_scaler(scaler, y_test)
    #test_predictions_price = inverse_scaler(scaler, test_predictions)
    
    df_analyze = analyze(date_test, y_test, test_predictions)
    print("#######TEST NO SCALE#######\n",df_analyze)
    plot(df_analyze,False)
    


    

main()