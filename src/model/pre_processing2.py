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

def get_close_price(df):
    """
    Extrait seulement le prix de cloture de la journée
    """
    return df['Close_price'].copy()

def formatting_data(df):
    """
    Prepare le fichier dataset_v2_1d_2017_08_17_2025_01_08
    """
    df.dropna(inplace=True)
    df.drop('Open_time', axis=1, inplace=True)
    df.drop('Close_time', axis=1, inplace=True)
    columns = ['Close_price'] + [col for col in df.columns if col != 'Close_price']
    df = df[columns]
    return df

def get_log_close_price(df):
    """
    Extrait seulement le prix de cloture de la journée en log
    """
    return np.log(df['Close_price'].copy())

def get_date(df):
    return df['Open_time'].copy()


def df_to_numpy(df):
    return df.to_numpy()

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

def window_matrix(df, window_size):
    df_numpy = df_to_numpy(df).reshape(len(df), -1)
    X = []
    y = []
    for i in range(len(df_numpy)-window_size):
        row = [a for a in df_numpy[i:i+window_size]]
        X.append(row)
        y.append(df_numpy[i+window_size])

    return np.array(X).astype(np.float32), np.array(y).astype(np.float32)

def normalization(X, y):
    print("X_shape av:", X.shape)
    print("y_shape av:", y.shape)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_train_scaled = scaler.fit_transform(y.reshape(-1, 1))

    return X_train_scaled, y_train_scaled, scaler

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

def main_pre_processing2_log_close_price(name, window_size):
    
    df = load_data(name)
    date = get_date(df)
    df_log_close_price = get_log_close_price(df)
    X, y = window_matrix(df_log_close_price, window_size)
    date_train, X_train, y_train, date_val, X_val, y_val, date_test, X_test, y_test = train_test_val(X, y, date, 0.8)
    
    datasets = {}
    
    # TRAIN
    X_train_norm, y_train_norm, scaler_train = normalization(X_train, y_train)
    datasets['TRAIN']=[[date_train, X_train_norm, y_train_norm, scaler_train]]
    
    # TEST
    X_test_norm, y_test_norm, scaler_test = normalization(X_test, y_test)
    datasets['TEST']=[[date_test, X_test_norm, y_test_norm, scaler_test]]
    
    # VAL
    X_val_norm, y_val_norm, scaler_val= normalization(X_val, y_val)
    datasets['VAL']=[[date_val, X_val_norm, y_val_norm, scaler_val]]

    return datasets



