import pandas as pd
import pickle
import os
from pathlib import Path

import sys
# scripts/ is not an installed package and the CLI runs this file by path, so
# put scripts/ on sys.path to make `common` importable. See scripts/common/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO_ROOT, data_root  # noqa: E402

DATA_ROOT = data_root()


pkl_path = DATA_ROOT / "processed_sites" / "df_Addis_Ababa_9am_resampled.pkl"
if pkl_path.exists():
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
