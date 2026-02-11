#!/usr/bin/env python3
"""
Compare two pickle files to understand their structure and differences
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Load both files
print("="*80)
print("LOADING PICKLE FILES")
print("="*80)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AETHMODULAR_DATA_ROOT", REPO_ROOT / "research" / "ftir_hips_chem"))

file1 = DATA_ROOT / "processed_sites" / "df_Addis_Ababa_9am_resampled.pkl"
file2 = DATA_ROOT / "df_Jacros_9am_resampled.pkl"

print(f"\nFile 1: {file1}")
print(f"File 2: {file2}")

df1 = pd.read_pickle(file1)
df2 = pd.read_pickle(file2)

print(f"\n✅ Both files loaded successfully")

# Basic information
print("\n" + "="*80)
print("BASIC INFORMATION")
print("="*80)

print(f"\nFile 1 (Addis_Ababa):")
print(f"  Shape: {df1.shape}")
print(f"  Data type: {type(df1)}")
print(f"  Memory usage: {df1.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\nFile 2 (Jacros):")
print(f"  Shape: {df2.shape}")
print(f"  Data type: {type(df2)}")
print(f"  Memory usage: {df2.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Column comparison
print("\n" + "="*80)
print("COLUMN COMPARISON")
print("="*80)

cols1 = set(df1.columns)
cols2 = set(df2.columns)

common_cols = cols1 & cols2
only_in_1 = cols1 - cols2
only_in_2 = cols2 - cols1

print(f"\nCommon columns: {len(common_cols)}")
print(f"Only in Addis_Ababa: {len(only_in_1)}")
print(f"Only in Jacros: {len(only_in_2)}")

if only_in_1:
    print(f"\nColumns only in Addis_Ababa:")
    for col in sorted(only_in_1):
        print(f"  - {col}")

if only_in_2:
    print(f"\nColumns only in Jacros:")
    for col in sorted(only_in_2):
        print(f"  - {col}")

# Show all columns for each file
print("\n" + "="*80)
print("ALL COLUMNS")
print("="*80)

print(f"\nAddis_Ababa columns ({len(df1.columns)}):")
for col in sorted(df1.columns):
    dtype = df1[col].dtype
    non_null = df1[col].notna().sum()
    print(f"  {col:50s} | {str(dtype):15s} | {non_null:6d} non-null")

print(f"\nJacros columns ({len(df2.columns)}):")
for col in sorted(df2.columns):
    dtype = df2[col].dtype
    non_null = df2[col].notna().sum()
    print(f"  {col:50s} | {str(dtype):15s} | {non_null:6d} non-null")

# Index comparison
print("\n" + "="*80)
print("INDEX COMPARISON")
print("="*80)

print(f"\nAddis_Ababa index:")
print(f"  Type: {type(df1.index)}")
print(f"  Name: {df1.index.name}")
print(f"  Range: {df1.index.min()} to {df1.index.max()}")

print(f"\nJacros index:")
print(f"  Type: {type(df2.index)}")
print(f"  Name: {df2.index.name}")
print(f"  Range: {df2.index.min()} to {df2.index.max()}")

# Data preview for common columns
print("\n" + "="*80)
print("DATA PREVIEW - COMMON KEY COLUMNS")
print("="*80)

# Check for common measurement columns
key_cols = ['IR BCc smoothed', 'UV BCc smoothed', 'BC', 'datetime', 'date']
available_key_cols = [col for col in key_cols if col in common_cols]

if available_key_cols:
    print(f"\nAddis_Ababa - First 5 rows:")
    print(df1[available_key_cols].head())

    print(f"\nJacros - First 5 rows:")
    print(df2[available_key_cols].head())

# Statistical summary for common numeric columns
print("\n" + "="*80)
print("STATISTICAL SUMMARY - COMMON NUMERIC COLUMNS")
print("="*80)

numeric_common = [col for col in common_cols if pd.api.types.is_numeric_dtype(df1[col])]

if numeric_common:
    print(f"\nComparing {len(numeric_common)} common numeric columns")

    comparison_data = []
    for col in sorted(numeric_common)[:10]:  # Show first 10
        row = {
            'Column': col,
            'Addis_Mean': df1[col].mean(),
            'Jacros_Mean': df2[col].mean(),
            'Addis_Std': df1[col].std(),
            'Jacros_Std': df2[col].std(),
            'Addis_Count': df1[col].notna().sum(),
            'Jacros_Count': df2[col].notna().sum()
        }
        comparison_data.append(row)

    comp_df = pd.DataFrame(comparison_data)
    print("\n", comp_df.to_string(index=False))

# Data quality comparison
print("\n" + "="*80)
print("DATA QUALITY COMPARISON")
print("="*80)

print(f"\nAddis_Ababa:")
print(f"  Total values: {df1.size}")
print(f"  Missing values: {df1.isna().sum().sum()}")
print(f"  Missing %: {df1.isna().sum().sum() / df1.size * 100:.2f}%")

print(f"\nJacros:")
print(f"  Total values: {df2.size}")
print(f"  Missing values: {df2.isna().sum().sum()}")
print(f"  Missing %: {df2.isna().sum().sum() / df2.size * 100:.2f}%")

# Key differences summary
print("\n" + "="*80)
print("KEY DIFFERENCES SUMMARY")
print("="*80)

print(f"\n1. SIZE DIFFERENCE:")
print(f"   Addis_Ababa: {df1.shape[0]} rows × {df1.shape[1]} columns")
print(f"   Jacros: {df2.shape[0]} rows × {df2.shape[1]} columns")
print(f"   Ratio: Jacros is {df2.shape[0]/df1.shape[0]:.2f}x larger in rows")

if len(only_in_1) > 0 or len(only_in_2) > 0:
    print(f"\n2. COLUMN DIFFERENCES:")
    print(f"   Unique to Addis_Ababa: {len(only_in_1)}")
    print(f"   Unique to Jacros: {len(only_in_2)}")
    print(f"   Common: {len(common_cols)}")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
