import os
import pandas as pd
import numpy as np
import datetime
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
load_dotenv()

WD = os.getenv('working_directory')


def load_data(name):
    """
    Load un dataframe depuis un fichier csv data/name.csv
    """
    return pd.read_csv(f'{WD}/data/{name}.csv')

def train_test_val(X, y, date, train_size):
    """
    """
    q_train_size = int(len(X)*train_size)
    q_val_test_size = int(len(X)*(train_size+(1-train_size)/2))
    date_numpy = df_to_numpy(date)

    X_train, y_train, date_train = X[:q_train_size], y[:q_train_size], date_numpy[:q_train_size]
    X_val, y_val, date_val = X[q_train_size:q_val_test_size], y[q_train_size:q_val_test_size], date_numpy[q_train_size:q_val_test_size]
    X_test, y_test, date_test = X[q_val_test_size:], y[q_val_test_size:], date_numpy[q_val_test_size:]

    return date_train, X_train, y_train, date_val, X_val, y_val, date_test, X_test, y_test

def main_pre_processing_log_close_price(name):
    """
    Renvoie X_train, y_train, X_val, y_val, X_test, y_test pour le df
    """
    WINDOW_SIZE = 7
    df = load_data(name)
    date = get_date(df)
    df_close_price = get_log_close_price(df)
    X, y, scaler_features, scaler_target = build_window_matrix_one_var(df_close_price, window_size=WINDOW_SIZE)
    return train_test_val(X, y, date, train_size=0.8), scaler_features, scaler_target