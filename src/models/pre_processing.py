import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
load_dotenv()

wd = os.getenv('working_directory')
WINDOW_SIZE = 7

def load_data(name):
    """
    Load un dataframe depuis un fichier csv data/name.csv
    """
    return pd.read_csv(f'{wd}/data/{name}.csv')

def get_close_price(df):
    """
    Extrait seulement le prix de cloture de la journée
    """
    return df['Close_price'].copy()

def get_date(df):
    date = df['date_utc'].copy()
    return date

def df_to_numpy(df):
    return df.to_numpy()
    

def build_window_matrix(df, window_size):
    """
    """
    scaler = MinMaxScaler()

    df_numpy = df_to_numpy(df).reshape(-1, 1)
    df_numpy_scaled = scaler.fit_transform(df_numpy)
    X = []
    y = []
    for i in range(len(df_numpy_scaled)-window_size):
        row = [a for a in df_numpy_scaled[i:i+window_size]]
        X.append(row)
        y.append(df_numpy_scaled[i+window_size])

    return np.array(X).astype(np.float32), np.array(y).astype(np.float32), scaler

def train_test_val(X, y, date, train_size):
    """
    """
    q_train_size = int(len(X)*train_size)
    q_val_test_size = int(len(X)*(train_size+(1-train_size)/2))
    print(q_train_size, q_val_test_size)

    X_train, y_train = X[:q_train_size], y[:q_train_size]
    X_val, y_val = X[q_train_size:q_val_test_size], y[q_train_size:q_val_test_size]
    X_test, y_test = X[q_val_test_size:], y[q_val_test_size:]

    date_numpy = df_to_numpy(date)
    date_train, date_val, date_test = date_numpy[:q_train_size], date_numpy[q_train_size:q_val_test_size], date_numpy[q_val_test_size:]
    return date_train, X_train, y_train, date_val, X_val, y_val, date_test, X_test, y_test


def main():

    df = load_data('transform_data')
    close = get_close_price(df)
    X, y = build_window_matrix(close, window_size=WINDOW_SIZE)
    print(X)

