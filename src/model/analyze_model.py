import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, r2_score

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Input, Dense

from pre_processing import main_pre_processing_pourcentage, main_pre_processing_close_price, main_pre_processing_log_close_price

from dotenv import load_dotenv
load_dotenv()

WINDOW_SIZE = 7
WD = os.getenv('working_directory')

def create_directory_if_not_exists(directory_name):
    """
    Crée un répertoire s'il n'existe pas déjà.

    Parameters:
    directory_name (str): Le nom du répertoire à créer.
    """
    if not os.path.exists(directory_name):
        try:
            os.makedirs(directory_name)
            print(f"Répertoire '{directory_name}' créé avec succès.")
        except Exception as e:
            print(f"Erreur lors de la création du répertoire '{directory_name}': {e}")
    else:
        print(f"Le répertoire '{directory_name}' existe déjà.")


def load(name):
    """
    """
    return load_model(f'{WD}/src/model/models/{name}/', custom_objects={'LSTM': LSTM})

def predict(model, X):
    """
    """
    return model.predict(X)

def inverse_scaler(scaler, array):
    """
    """
    return scaler.inverse_transform(array)
    

def analyze(date, X, y, predictions):
    """
    """
    df_analyze = pd.concat([
                            pd.DataFrame(date), 
                            pd.DataFrame(X[:, WINDOW_SIZE-1, 0]), 
                            pd.DataFrame(y.reshape(-1)), 
                            pd.DataFrame(predictions)
                            ], axis=1).dropna()

    df_analyze.columns = ['Date', 'Actual N', 'Actual N+1', 'Predictions N+1']

    df_analyze['Actual UP or DOWN'] = pd.DataFrame((df_analyze['Actual N+1'] > df_analyze['Actual N']).astype(int))
    df_analyze['Predicted UP or DOWN'] = pd.DataFrame((df_analyze['Predictions N+1'] > df_analyze['Actual N']).astype(int))

    r2 = r2_score(df_analyze['Actual N+1'].values, df_analyze['Predictions N+1'].values)
    print(f'R2 : {r2*100}%')
    cm = confusion_matrix(df_analyze['Actual UP or DOWN'], df_analyze['Predicted UP or DOWN'])
    acc = cm[0][0] + cm[1][1]
    tot = cm.sum()
    print(f'Accuracy : {acc/tot*100}%')
    cm_df = pd.DataFrame(cm, index=['Actual DOWN', 'Actual UP'], columns=['Predicted DOWN', 'Predicted UP'])
    
    return df_analyze, cm_df

def analyze_pourcentage(date, X, y, predictions):
    """
    """
    df_analyze = pd.concat([
                            pd.DataFrame(date), 
                            pd.DataFrame(X[:, WINDOW_SIZE-1, 0]), 
                            pd.DataFrame(y.reshape(-1)), 
                            pd.DataFrame(predictions)
                            ], axis=1)

    print(df_analyze)

    df_analyze.columns = ['Date', 'Actual N', 'Actual N+1', 'Predictions N+1']

    df_analyze['Actual UP or DOWN'] = pd.DataFrame((df_analyze['Actual N+1'] > df_analyze['Actual N']).astype(int))
    df_analyze['Predicted UP or DOWN'] = pd.DataFrame((df_analyze['Predictions N+1'] > df_analyze['Actual N']).astype(int))
    
    cm = confusion_matrix(df_analyze['Actual UP or DOWN'], df_analyze['Predicted UP or DOWN'])
    acc = cm[0][0] + cm[1][1]
    tot = cm.sum()
    print(f'Accuracy : {acc/tot*100}%')
    cm_df = pd.DataFrame(cm, index=['Actual DOWN', 'Actual UP'], columns=['Predicted DOWN', 'Predicted UP'])
    
    return df_analyze, cm_df
    
def plot(df, model_name, is_train):
    """
    """
    create_directory_if_not_exists(f'{WD}/figure/{model_name}')
    plt.clf()
    plt.plot(df['Date'], df['Actual N+1'], label='Actuals') 
    plt.plot(df['Date'], df['Predictions N+1'], label='Predictions')
    plt.legend()
    plt.savefig(f"{WD}/figure/{model_name}/COMPARAISON_{'train_' if is_train==1 else 'test_' if is_train==0 else'val_'}prediction.png")

def analyze_for_data(model, scaler_features, scaler_target, date_train, X_train, y_train, date_test, X_test, y_test, date_val, X_val, y_val, is_train):

    if is_train==1:
        date = date_train
        X = X_train
        y = y_train
    elif is_train==0:
        date = date_test
        X = X_test
        y = y_test
    else :
        date = date_val
        X = X_val
        y = y_val
    print("X.shape", X.shape)

    pred = predict(model, X)
    X_reshape = X.reshape(X.shape[0], -1)
    X_inv_scaled = inverse_scaler(scaler_features, X_reshape)
    X_inv_scaled = np.array(X_inv_scaled)
    X = X_inv_scaled.reshape(X.shape[0], X.shape[1], X.shape[2])
    #y = inverse_scaler(scaler_target, [y])
    y = inverse_scaler(scaler_target, y)
    predictions = inverse_scaler(scaler_target, pred)

    df_analyze, cm_df = analyze(date, X, y, predictions)
    print(f"#####{'TRAIN' if is_train==1 else 'TEST' if is_train==0 else 'VAL'}#####\n",df_analyze)
    print(f'######MATRICE DE CONFUSION#######\n {cm_df}')
    plot(df_analyze, model.name, is_train)

def main_analyze_model(model_name):
    data_name = model_name.split('-')[1]
    loaded_model = load(model_name)

    (date_train, X_train, y_train, date_val, X_val, y_val, date_test, X_test, y_test), scaler_features, scaler_target = main_pre_processing_log_close_price(data_name)
    for is_train in (1, 2, 0):
        analyze_for_data(loaded_model, scaler_features, scaler_target, date_train, X_train, y_train, date_test, X_test, y_test, date_val, X_val, y_val, is_train)

# model_name = "test_Close_price_log_jaquart_dense64-dataset_raw_1d_2017_08_17_2025_01_08-300"
# main_analyze_model(model_name)