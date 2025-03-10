import os
import pandas as pd
import numpy as np
import datetime
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
load_dotenv()

WD = os.getenv('working_directory')
#WINDOW_SIZE = 7

def load_data(name):
    """
    Load un dataframe depuis un fichier csv data/name.csv
    """
    return pd.read_csv(f'{WD}/data/{name}.csv')

def rearrange_data(df):
    df.dropna(inplace=True)
    df.drop('Ignore', axis=1, inplace=True)
    df.drop('Open_time', axis=1, inplace=True)
    df.drop('Close_time', axis=1, inplace=True)
    columns = ['Close_price'] + [col for col in df.columns if col != 'Close_price']
    df = df[columns]
    return df

def get_close_price(df):
    """
    Extrait seulement le prix de cloture de la journée
    """
    return df['Close_price'].copy()

def get_log_close_price(df):
    """
    Extrait seulement le prix de cloture de la journée en log
    """
    return np.log(df['Close_price'].copy())

def get_date(df):
    return df['Open_time'].copy()

def get_pourcentage(df):
    return df['evolution_prct'].copy().dropna()

def df_to_numpy(df):
    return df.to_numpy()
    
def build_window_matrix_multi_var(df, window_size):
    """
    """
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    df_numpy = df_to_numpy(df)

    X = []
    y = []
    for i in range(len(df_numpy)-window_size):
        row = [a for a in df_numpy[i:i+window_size]]
        X.append(row)
        y.append(df_numpy[i+window_size][0])

    #Scaling de X
    X = np.array(X)
    X_reshape = X.reshape(X.shape[0], -1)
    X_scaled = scaler_features.fit_transform(X_reshape)
    X_scaled = np.array(X_scaled)
    X = X_scaled.reshape(X.shape[0], X.shape[1], X.shape[2])
    X = np.array(X)

    #Scaling de y
    y = np.array(y)
    y_reshape = y.reshape(-1, 1)
    y_scaled = scaler_target.fit_transform(y_reshape)
    y_scaled = np.array(y_scaled)
    y = y_scaled.reshape(y.shape[0],)

    return X.astype(np.float32), y.astype(np.float32), scaler_features, scaler_target

def build_window_matrix_one_var(df, window_size):
    """
    """
    scaler = MinMaxScaler()
    df_numpy = df_to_numpy(df).reshape(len(df), -1)
    print("df_numpy.shape : ",df_numpy.shape)
    df_numpy_scaled = scaler.fit_transform(df_numpy)
    print("df_numpy_scaled.shape : ",df_numpy_scaled.shape)
    
    X = []
    y = []

    for i in range(len(df_numpy_scaled)-window_size):
        row = [a for a in df_numpy_scaled[i:i+window_size]]
        X.append(row)
        y.append(df_numpy_scaled[i+window_size])

    print('X_shape', np.array(X).shape)
    scaler_features = scaler
    scaler_target = scaler

    return np.array(X).astype(np.float32), np.array(y).astype(np.float32), scaler_features, scaler_target

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

def main_pre_processing_close_price(name):
    """
    Renvoie X_train, y_train, X_val, y_val, X_test, y_test pour le df
    """
    df = load_data(name)
    date = get_date(df)
    df = get_close_price(df)
    X, y, scaler_features, scaler_target = build_window_matrix_one_var(df, window_size=WINDOW_SIZE)
    return train_test_val(X, y, date, train_size=0.8), scaler_features, scaler_target

def main_pre_processing_log_close_price(name):
    """
    Renvoie X_train, y_train, X_val, y_val, X_test, y_test pour le df
    """
    df = load_data(name)
    date = get_date(df)
    df_close_price = get_log_close_price(df)
    X, y, scaler_features, scaler_target = build_window_matrix_one_var(df_close_price, window_size=WINDOW_SIZE)
    return train_test_val(X, y, date, train_size=0.8), scaler_features, scaler_target

def main_pre_processing_all_var(name):
    """
    Renvoie X_train, y_train, X_val, y_val, X_test, y_test pour le df
    """
    df = load_data(name)
    date = get_date(df)
    df = rearrange_data(df)
    X, y, scaler_features, scaler_target = build_window_matrix_multi_var(df, window_size=WINDOW_SIZE)
    return train_test_val(X, y, date, train_size=0.8), scaler_features, scaler_target

def main_pre_processing_pourcentage(name):
    """
    Renvoie X_train, y_train, X_val, y_val, X_test, y_test pour le df
    """
    df = load_data(name)
    date = get_date(df)
    df = get_pourcentage(df)
    print(df.head())
    X, y, scaler_features, scaler_target = build_window_matrix_one_var(df, window_size=WINDOW_SIZE)
    return train_test_val(X, y, date, train_size=0.8), scaler_features, scaler_target

# (date_train, X_train, y_train, date_val, X_val, y_val, date_test, X_test, y_test), scaler_features, scaler_target = main_pre_processing_log_close_price("dataset_raw_1d_2017_08_17_2025_01_08")
# print(f'X_train : {X_train.shape}')
# print(f'y_train : {y_train.shape}')
# print(f'X_val : {X_val.shape}')
# print(f'y_val : {y_val.shape}')
# print(f'X_test : {X_test.shape}')
# print(f'y_test : {y_test.shape}')
# #print(f'{date_train[0]}|{scaler_features.inverse_transform(X_train[0,:,:])}| {y_train[0]}')
# #datetime.datetime.fromtimestamp(date_train[0]/1000).date()