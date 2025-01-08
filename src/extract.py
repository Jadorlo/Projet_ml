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


def api_call(url):
    """
    Permet de faire appel à l'api binance
    """
    url = BASE_URL + url
    params = {"symbol": "BTCUSDT", 
              "interval":"1d"}
    response = requests.get(url, params)
    if response.status_code==200:
        data=response.json()
    return data

def get_klines():
    """
    Appel API pour récupérer les bougies
    """
    return api_call('/api/v3/klines')
    
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
    df.to_csv(f'{wd}/data/{name}.csv')

def transform_klines():
    """"""


def main():
    """
    """
    df = klines_to_raw_df()
    save_df_csv(df, 'raw')

main()