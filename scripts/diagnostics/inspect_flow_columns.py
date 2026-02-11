#!/usr/bin/env python3
"""
Inspect the structure of the pickle files to find flow-related columns
"""

import pandas as pd
import pickle
from pathlib import Path
import os

# Load one of the datasets
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AETHMODULAR_DATA_ROOT", REPO_ROOT / "research" / "ftir_hips_chem"))
file_path = DATA_ROOT / "processed_sites" / "df_Addis_Ababa_9am_resampled.pkl"

with open(file_path, 'rb') as f:
    df = pickle.load(f)

print('='*80)
print('DATASET COLUMNS')
print('='*80)
print(f'Total columns: {len(df.columns)}\n')

# Show all columns
print('All columns:')
for i, col in enumerate(sorted(df.columns), 1):
    print(f'{i:3d}. {col}')

print('\n' + '='*80)
print('COLUMNS WITH "FLOW" IN THE NAME')
print('='*80)
flow_cols = [col for col in df.columns if 'flow' in col.lower()]
print(f'Found {len(flow_cols)} columns with "flow" in the name:\n')
for col in flow_cols:
    non_null = df[col].notna().sum()
    print(f'  {col:50s} - {non_null} non-null values')

print('\n' + '='*80)
print('COLUMNS WITH "BCc" IN THE NAME')
print('='*80)
bc_cols = [col for col in df.columns if 'BCc' in col or 'BC' in col]
print(f'Found {len(bc_cols)} columns:\n')
for col in bc_cols:
    non_null = df[col].notna().sum()
    print(f'  {col:50s} - {non_null} non-null values')

print('\n' + '='*80)
print('FIRST FEW ROWS OF BC COLUMNS')
print('='*80)
if bc_cols:
    print(df[bc_cols[:10]].head(10))

print('\n' + '='*80)
print('WAVELENGTH COLUMNS')
print('='*80)
wavelengths = ['UV', 'Blue', 'Green', 'Red', 'IR']
for wl in wavelengths:
    wl_cols = [col for col in df.columns if wl in col and 'BCc' in col]
    if wl_cols:
        print(f'\n{wl} wavelength columns:')
        for col in wl_cols:
            non_null = df[col].notna().sum()
            print(f'  {col:50s} - {non_null} non-null values')
