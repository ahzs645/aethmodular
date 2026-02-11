import pandas as pd
import pickle
import os

pkl_path = 'FTIR_HIPS_Chem/processed_sites/df_Addis_Ababa_9am_resampled.pkl'
if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    print(f"Aethalometer Sample Count (days): {len(df)}")
    if 'day_9am' in df.columns:
        print(f"Aethalometer Date Range: {df['day_9am'].min()} to {df['day_9am'].max()}")
    elif 'datetime_local' in df.columns:
        print(f"Aethalometer Date Range: {df['datetime_local'].min()} to {df['datetime_local'].max()}")
    else:
        print(f"Index range: {df.index.min()} to {df.index.max()}")
else:
    print(f"File not found: {pkl_path}")
