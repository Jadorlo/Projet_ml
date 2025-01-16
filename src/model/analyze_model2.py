import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, r2_score

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Input, Dense

from pre_processing2 import main_pre_processing2_log_close_price, main_pre_processing2_multivar

from dotenv import load_dotenv
load_dotenv()


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

def plot(df, model_name, data_type):
    """
    Affiche les graphes de comparaison
    """
    ticks_to_display = np.linspace(0, len(df) - 1, 5, dtype=int)  # 5 indices répartis uniformément
    create_directory_if_not_exists(f'{WD}/figure/{model_name}')
    plt.clf()
    plt.plot(df['Date'], df['Actual N+8'], label='Actuals') 
    plt.plot(df['Date'], df['Predictions N+8'], label='Predictions')
    plt.xticks(ticks_to_display, df["Date"].iloc[ticks_to_display], rotation=45)
    plt.title(f"Comparaison Réel et Prédiction pour le jeu de {data_type}")
    plt.legend()
    plt.savefig(f"{WD}/figure/{model_name}/COMPARAISON_{data_type}_prediction.png")
    plt.show()

def analyze_table(date, X, y, y_pred, scaler, window_size):
    """
    """

    def table(date, X_inv_scaled, y_inv_scaled, y_pred_inv_scaled):
        """
        Construit le tableau d'analyse
        """
        print(pd.DataFrame(y_pred_inv_scaled).head())
        df_analyze = pd.concat([
                            pd.DataFrame(date), 
                            pd.DataFrame(X_inv_scaled[:, window_size-1, 0]), 
                            pd.DataFrame(y_inv_scaled.reshape(-1)), 
                            pd.DataFrame(y_pred_inv_scaled)
                            ], axis=1).dropna()
        df_analyze.columns = ['Date', 'Actual N+7', 'Actual N+8', 'Predictions N+8']

        # Crée les variables dichotomiques targets pour les vrais valeurs et les prédictions
        df_analyze['Actual UP or DOWN'] = pd.DataFrame((df_analyze['Actual N+8'] > df_analyze['Actual N+7']).astype(int))
        df_analyze['Predicted UP or DOWN'] = pd.DataFrame((df_analyze['Predictions N+8'] > df_analyze['Actual N+7']).astype(int))
        
        
        return df_analyze
    
    def R2(df_analyze):
        """
        R2 sur les prédictions du log_close_prix
        """

        return r2_score(df_analyze['Actual N+8'].values, df_analyze['Predictions N+8'].values)*100
    
    def CM(df_analyze):

        """
        Construit la matrice de confusion et calcule l'accuracy
        """
        cm = confusion_matrix(df_analyze['Actual UP or DOWN'], df_analyze['Predicted UP or DOWN'])
        cm_df = pd.DataFrame(cm, index=['Actual DOWN', 'Actual UP'], columns=['Predicted DOWN', 'Predicted UP'])
        acc= cm[0][0] + cm[1][1]
        tot = cm.sum()
        accuracy = acc/tot*100
        return cm_df, accuracy

    # X Reshape, Rescaled
    X_reshape = X.reshape(X.shape[0], -1)
    X_inv_scaled_reshape = np.array(inverse_scaler(scaler, X_reshape))
    X_inv_scaled = X_inv_scaled_reshape.reshape(X.shape)

    # y Rescaled
    y_inv_scaled = inverse_scaler(scaler, y)
    y_pred_inv_scaled = inverse_scaler(scaler, y_pred)
    

    df_analyze = table(date, X_inv_scaled, y_inv_scaled, y_pred_inv_scaled)
    r2 = R2(df_analyze)
    cm, accuracy = CM(df_analyze)

    return df_analyze, r2, cm, accuracy



def analyze(model, data_type, datasets, windows_size):
    """
    """

    date = datasets[data_type][0][0]
    X = datasets[data_type][0][1]
    y = datasets[data_type][0][2]
    scaler = datasets[data_type][0][3]

    y_pred = predict(model, X)

    df_analyze, r2, cm, accuracy = analyze_table(date, X, y, y_pred, scaler, windows_size)
    print(f'R^2 = {r2}%')
    print(f'Accuracy = {accuracy}%')
    print(f"#####{data_type}#####\n", df_analyze)
    print(f'######MATRICE DE CONFUSION#######\n {cm}')
    plot(df_analyze, model.name, data_type)

def main_analyze_model2(model_name, window_size):

    data_name = model_name.split('-')[1]
    loaded_model = load(model_name)

    datasets = main_pre_processing2_log_close_price(data_name, window_size)
    #datasets = main_pre_processing2_multivar(data_name, window_size)

    for data_type in ('TRAIN', 'VAL', 'TEST'):
        analyze(loaded_model, data_type, datasets, window_size)

#main_analyze_model2("test_preprocess2_Close_price_log_jaquart_dense64_dense2_WS7-dataset_raw_1d_2017_08_17_2025_01_08-100", 7)




