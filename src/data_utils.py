import pandas as pd
import numpy as np
from pathlib import Path

def load_listings(path):
    return pd.read_csv(path)

def clean_price(df, price_col='price'):
    df = df.copy()
    df[price_col] = df[price_col].astype(str).str.replace(r'[\$,]', '', regex=True)
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    return df

def save_processed(df, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
