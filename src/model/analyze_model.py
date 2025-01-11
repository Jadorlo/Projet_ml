import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from tensorflow.keras.models import load_model
from dotenv import load_dotenv
load_dotenv()

WD = os.getenv('working_directory')

def load():
    """
    """
    return load_model(f'{WD}/src/model/models/', custom_objects={'LSTM': LSTM})

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