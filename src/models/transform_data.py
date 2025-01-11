import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

wd = os.getenv('working_directory')

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

def build_window_matrix(df, window_size):
    """
    """
    close_price_numpy = df.to_numpy()
    X = []
    y = []
    for i in range(len(close_price_numpy)-window_size):
        row = [[a] for a in close_price_numpy[i:i+window_size]]
        X.append(row)
        y.append(close_price_numpy[i+window_size])

    return np.array(X), np.array(y)

def train_test_val(X, y):
    """
    """
    X_train, y_train = X[:345], y[:345]
    X_val, y_val = X[345:420], y[345:420]
    X_test, y_test = X[420:], y[420:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    WINDOW_SIZE = 5
    df = load_data('transform_data')
    close = get_close_price(df)
    X, y = build_window_matrix(close, window_size=WINDOW_SIZE)
    print(X.shape, y.shape)
