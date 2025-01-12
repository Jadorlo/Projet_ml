import os
import sys
from datetime import datetime
import pandas as pd
import requests
import json
from dotenv import load_dotenv
load_dotenv()

#Get klines (candles) 
#GET /api/v3/klines

BASE_URL = 'https://api.binance.com/'
wd = os.getenv('working_directory')
BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"  # Paire BTC/USDT
INTERVAL = "1d"     # Intervalle des chandeliers : 6 heures
LIMIT = 500         # Limite par requête

def fetch_klines(symbol, interval, start_time, end_time=None, limit=500):
    """
    Récupère les données de chandeliers (klines) depuis l'API Binance.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": start_time,
    }
    if end_time:
        params["endTime"] = end_time
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()  # Génère une exception si l'API renvoie une erreur
    return response.json()

def api_call(url, startTime, endTime):
    """
    Permet de faire appel à l'api binance
    """
    url = BASE_URL + url
    params = {"symbol": "BTCUSDT", 
              "interval":"6h",
              "startTime":startTime,
              "endTime":endTime}
    response = requests.get(url, params)
    if response.status_code==200:
        data=response.json()
    else:
        print(response.status_code)
    return data

def get_inteval_data_for_500_days(symbol, interval, limit):
    """
    Récupère les données des 500 mêmes jours en chandeliers de 6 heures,
    concatène les résultats dans un DataFrame et les sauvegarde dans un fichier CSV.
    """
    # Timestamp pour les 500 derniers jours
    current_time = 1736294400000  # Timestamp actuel en ms
    past_time = 1693180800000  # 500 jours en arrière en ms
    output_csv=f"dataset_raw_{interval}.csv"
    
    # Initialiser le DataFrame pour concaténer les données
    all_data = []
    start_time = past_time

    while True:
        try:
            # Récupérer les données pour l'intervalle donné
            klines = fetch_klines(symbol, interval, start_time, limit=limit)
            
            # Arrêter la boucle si aucune donnée n'est retournée
            if not klines:
                break
            
            # Convertir les données en DataFrame
            df = pd.DataFrame(klines, columns=[
                "Open Time", "Open", "High", "Low", "Close", "Volume",
                "Close Time", "Quote Asset Volume", "Number of Trades",
                "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
            ])
            df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
            df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
            all_data.append(df)
            
            # Mettre à jour start_time pour le prochain appel
            last_close_time = int(klines[-1][6])  # Close Time du dernier chandelier
            start_time = last_close_time + 1  # Éviter le chevauchement
            
            # Vérifier si on a dépassé la période souhaitée
            if start_time > current_time:
                break
        except Exception as e:
            print(f"Erreur pendant la récupération : {e}")
            break

    # Concaténer toutes les données dans un seul DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Sauvegarder dans un fichier CSV
    final_df.to_csv(f'{wd}/data/{output_csv}', index=False)
    print(f"Les données ont été sauvegardées dans {output_csv}")

# Appeler la fonction pour récupérer les données
get_inteval_data_for_500_days(SYMBOL, INTERVAL, LIMIT)



def get_klines():
    """
    Appel API pour récupérer les bougies
    """
    return api_call('/api/v3/klines')

def get_long_klines(startTime, endTime):
    """
    """
    
def klines_to_raw_df():
    """
    data to dataframe
    """
    data = get_klines()
    df = pd.DataFrame(data)
    df.drop(11, axis=1, inplace=True)
    df.columns = ['Open_time', 'Open_price', 'High_price', 'Low_price', 'Close_price', 'Volume', 'Close_time', 'Quote_volume', 'Nb_trades', 'Taker_buy_base_volume', 'Taker_buy_quote_volume']
    return df

def save_df_csv(df, name):
    """
    save en csv dans le fichier data
    """
    df.to_csv(f'{wd}/data/{name}.csv', index=False)



def main():
    """
    """
    df = klines_to_raw_df()
    save_df_csv(df, "dataset_raw_6h")

# main()