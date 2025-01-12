import os
import sys
from datetime import datetime
import pandas as pd
import requests
import json
from dotenv import load_dotenv
load_dotenv()

wd = os.getenv('working_directory')

def get_data(name):
    """
    """
    return pd.read_csv(f'{wd}/data/{name}.csv')

def transform_klines(df):
    """
    """
    df['date_utc'] = pd.to_datetime(df['Open_time'], unit='ms', utc=True)
    df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).apply(pd.to_numeric, errors='coerce')
    #La date en premier
    cols = ['date_utc'] + [col for col in df.columns if col != 'date_utc']
    df = df[cols]
    return df

def create_target(df):
    """
    """
    df['target_up'] = (df['Close_price'].shift(+1) < df['Close_price']).astype(int)
    return df

def indicator_rsi(df, period=14):
    """
    Calcul le rsi pour df
    """
    delta = df['Close_price'].diff()
    up, down = delta.clip(lower=0), delta.clip(upper=0, lower=None)
    mean_up = up.ewm(alpha=1/period, min_periods=period).mean()
    mean_down = down.abs().ewm(alpha=1/period, min_periods=period).mean()

    # Calcul des moyennes mobiles exponentielles (RMA)  
    # mean_up = up.ewm(span=period, adjust=False).mean()
    # mean_down = down.abs().ewm(span=period, adjust=False).mean()

    rs = mean_up / mean_down

    rsi = 100 - (100 / (1 + rs))
    rsi.name = "RSI"
    df = pd.concat([df, rsi], axis=1)

    return df


def main():

    df = get_data("raw")
    df = transform_klines(df)
    df =  create_target(df)
    print(indicator_rsi(df, period=14))

main()