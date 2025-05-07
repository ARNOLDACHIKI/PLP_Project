# data_utils.py
import pandas as pd

def load_data(path='data/synthetic_battery_data.csv') -> pd.DataFrame:
    return pd.read_csv(path)

def save_data(df: pd.DataFrame, path):
    df.to_csv(path, index=False)
