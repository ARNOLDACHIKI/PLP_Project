# data_preprocessing.py
import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Simple preprocessing: clean nulls and normalize
    df = df.dropna()
    df['speed'] = df['speed'] / 120  # normalize
    df['acceleration'] = df['acceleration'] / 5
    df['temperature'] = (df['temperature'] - 15) / (40 - 15)
    df['battery_health'] = df['battery_health'] / 100
    return df
