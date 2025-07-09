# Customizing the Aethalometer Analysis Notebook

This guide shows how to adapt the notebook for different aethalometer data files.

## Quick Customization

### 1. Change Data File Path

Edit the second code cell in the notebook:

```python
# Change this line to point to your data file
data_path = "/path/to/your/aethalometer_data.pkl"
```

### 2. Supported Data Formats

The system automatically detects:
- **Standard aethalometer format**: `'IR BCc', 'Blue BCc', 'Green BCc'` etc.
- **JPL repository format**: `'IR.BCc', 'Blue.BCc', 'Green.BCc'` etc.

### 3. Common Data File Locations

```python
# Example paths for different data sources
data_paths = {
    "uncleaned_jacros": "/Users/.../df_uncleaned_Jacros_API_and_OG.pkl",
    "cleaned_hourly": "/Users/.../df_cleaned_Jacros_hourly.pkl", 
    "merged_data": "/Users/.../merged_aethalometer_data.pkl",
    "custom_export": "/Users/.../my_aethalometer_export.pkl"
}

# Use any of these
data_path = data_paths["uncleaned_jacros"]
```

## Advanced Customization

### 1. Add Custom Analysis

Insert a new cell with your analysis:

```python
# Custom analysis example
if df is not None:
    # Your custom analysis here
    custom_result = df['Blue BCc'].rolling(window=24).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(custom_result)
    plt.title("24-hour Rolling Average - Blue BC")
    plt.show()
```

### 2. Filter Data by Time Period

```python
# Filter to specific date range
if df is not None and isinstance(df.index, pd.DatetimeIndex):
    # Filter to last 30 days
    recent_data = df.last('30D')
    
    # Filter to specific date range
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    filtered_data = df[start_date:end_date]
```

### 3. Export Results

```python
# Save processed data
if df is not None:
    # Export to CSV
    df.to_csv('processed_aethalometer_data.csv')
    
    # Export subset of columns
    bc_columns = [col for col in df.columns if 'BC' in str(col)]
    df[bc_columns].to_csv('bc_concentrations_only.csv')
    
    # Export summary statistics
    summary_stats = df.describe()
    summary_stats.to_csv('aethalometer_summary_stats.csv')
```

### 4. Custom Visualizations

```python
# Create custom plots
if df is not None:
    # Multi-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot different BC wavelengths
    bc_cols = ['IR BCc', 'Blue BCc', 'Green BCc', 'Red BCc']
    for i, col in enumerate(bc_cols[:4]):
        if col in df.columns:
            row, col_idx = i // 2, i % 2
            df[col].plot(ax=axes[row, col_idx], title=col)
    
    plt.tight_layout()
    plt.show()
```

## Troubleshooting

### Data Loading Issues

1. **File not found**:
   ```python
   # Check if file exists
   import os
   if os.path.exists(data_path):
       print("✅ File found")
   else:
       print("❌ File not found")
   ```

2. **Wrong format**:
   ```python
   # Force format detection
   loader = AethalometerPKLLoader(data_path, format_type="standard")
   # or
   loader = AethalometerPKLLoader(data_path, format_type="jpl")
   ```

3. **Memory issues with large files**:
   ```python
   # Load sample of data first
   df_sample = pd.read_pickle(data_path).head(1000)
   print(f"Sample loaded: {len(df_sample)} rows")
   ```

### Import Issues

If you get import errors:
```python
# Add this to the top of your notebook
import sys
sys.path.insert(0, '../src')  # Adjust path as needed
```

### Column Name Issues

```python
# Check available columns
print("Available columns:")
for i, col in enumerate(df.columns):
    print(f"{i:3d}: {col}")

# Find BC columns
bc_cols = [col for col in df.columns if 'BC' in str(col).upper()]
print(f"BC columns found: {bc_cols}")
```

## Tips for Large Datasets

1. **Use data chunking**:
   ```python
   # Process in chunks for large files
   chunk_size = 10000
   for chunk in pd.read_pickle(data_path, chunksize=chunk_size):
       # Process each chunk
       result = chunk.mean()
   ```

2. **Downsample for exploration**:
   ```python
   # Use every 10th row for initial exploration
   df_sample = df.iloc[::10]
   ```

3. **Memory monitoring**:
   ```python
   # Check memory usage
   print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
   ```

---

**📝 Remember to save your customized notebook with a new name to preserve the original template!**
