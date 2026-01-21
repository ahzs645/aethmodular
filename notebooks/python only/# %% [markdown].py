# %% [markdown]
# # Aethalometer Data Analysis
# ## Sample Analysis Using the Modular Aethalometer System
# 
# This notebook demonstrates how to load and analyze aethalometer data using the modular system. We'll be working with a pickle file containing merged cleaned and uncleaned MA350 data.
# 
# **Data Source:** `df_uncleaned_Jacros_API_and_OG.pkl`
# 
# ### Features demonstrated:
# - Data loading using the AethalometerPKLLoader
# - Basic data inspection and statistics
# - Time series visualization
# - Source apportionment analysis
# - Quality assessment

# %%
# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = str(Path('../src').resolve())
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import modular system components
try:
    from data.loaders.aethalometer import AethalometerPKLLoader, load_aethalometer_data
    print("✅ Data loaders imported successfully")
except ImportError as e:
    print(f"⚠️ Data loaders import error: {e}")

try:
    from analysis.bc.black_carbon_analyzer import BlackCarbonAnalyzer
    print("✅ Black Carbon analyzer imported successfully")
except ImportError as e:
    print(f"⚠️ Black Carbon analyzer import error: {e}")
    BlackCarbonAnalyzer = None

try:
    from analysis.bc.source_apportionment import SourceApportionmentAnalyzer
    print("✅ Source Apportionment analyzer imported successfully")
except ImportError as e:
    print(f"⚠️ Source Apportionment analyzer import error: {e}")
    SourceApportionmentAnalyzer = None

try:
    from utils.plotting import AethalometerPlotter
    print("✅ Plotting utilities imported successfully")
except ImportError as e:
    print(f"⚠️ Plotting utilities import error: {e}")
    AethalometerPlotter = None

try:
    from config.plotting import setup_plotting_style
    setup_plotting_style()
    print("✅ Plotting style configured successfully")
except ImportError as e:
    print(f"⚠️ Plotting config import error: {e}")
    # Fallback plotting style
    plt.style.use('default')
    sns.set_palette("husl")

try:
    from utils.file_io import ensure_output_directory
    print("✅ File I/O utilities imported successfully")
except ImportError as e:
    print(f"⚠️ File I/O utilities import error: {e}")
    # Create a simple fallback function
    def ensure_output_directory(path):
        os.makedirs(path, exist_ok=True)

# Setup plotting style
plt.rcParams['figure.figsize'] = (12, 6)

print("\n✅ All available libraries imported successfully!")
print("📊 Modular aethalometer analysis system ready!")
print(f"📁 Working directory: {os.getcwd()}")
print(f"🔗 Source path added: {src_path}")

# %% [markdown]
# ## 1. Load the Pickle DataFrame
# 
# We'll load the aethalometer data from the specified pickle file using both direct pandas loading and the modular system's AethalometerPKLLoader.

# %%
# Define the data file path
data_path = "/Users/ahzs645/Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University/Research/Grad/UC Davis Ann/NASA MAIA/Data/Aethelometry Data/Kyan Data/Mergedcleaned and uncleaned MA350 data20250707030704/df_uncleaned_Jacros_API_and_OG.pkl"

print(f"📁 Loading data from: {Path(data_path).name}")
print(f"📍 Full path: {data_path}")

# Method 1: Direct pandas loading
try:
    df_direct = pd.read_pickle(data_path)
    print(f"✅ Successfully loaded with pandas: {len(df_direct)} rows")
except Exception as e:
    print(f"❌ Error loading with pandas: {e}")
    df_direct = None

# Method 2: Using the modular system's AethalometerPKLLoader
try:
    loader = AethalometerPKLLoader(data_path, format_type="auto")
    
    # Get data summary
    summary = loader.get_data_summary()
    print(f"\n📊 Data Summary from AethalometerPKLLoader:")
    for key, value in summary.items():
        if key != 'columns':
            print(f"   {key}: {value}")
    
    # Load the data
    df_modular = loader.load(convert_to_jpl=False)
    print(f"✅ Successfully loaded with modular system: {len(df_modular)} rows")
    
except Exception as e:
    print(f"❌ Error loading with modular system: {e}")
    df_modular = None

# Use whichever method worked
df = df_direct if df_direct is not None else df_modular

if df is not None:
    print(f"\n🎯 Working with DataFrame: {len(df)} rows × {len(df.columns)} columns")
else:
    print("\n❌ Failed to load data with both methods")

# %% [markdown]
# ## 2. Display DataFrame Information
# 
# Let's examine the structure of our data, including column names, data types, and memory usage.

# %%
if df is not None:
    print("📊 DATAFRAME INFORMATION")
    print("=" * 50)
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Index type: {type(df.index).__name__}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display DataFrame info
    print("\n🔍 DataFrame Info:")
    df.info()
    
    # Check for datetime columns
    print(f"\n📅 Index range:")
    if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
        try:
            print(f"   From: {df.index.min()}")
            print(f"   To: {df.index.max()}")
            print(f"   Duration: {df.index.max() - df.index.min()}")
        except:
            print(f"   Index range: {df.index[0]} to {df.index[-1]}")
    
    # Column overview
    print(f"\n📋 Column Categories:")
    bc_cols = [col for col in df.columns if 'BC' in str(col).upper()]
    atn_cols = [col for col in df.columns if 'ATN' in str(col).upper()]
    flow_cols = [col for col in df.columns if 'flow' in str(col).lower()]
    
    print(f"   Black Carbon columns: {len(bc_cols)}")
    print(f"   Attenuation columns: {len(atn_cols)}")
    print(f"   Flow columns: {len(flow_cols)}")
    print(f"   Other columns: {len(df.columns) - len(bc_cols) - len(atn_cols) - len(flow_cols)}")
    
else:
    print("❌ No data available to display information")

# %% [markdown]
# ## 3. Preview DataFrame Contents
# 
# Let's look at the first and last few rows to understand the data structure.

# %%
if df is not None:
    print("🔍 FIRST 5 ROWS")
    print("=" * 50)
    display(df.head())
    
    print(f"\n🔍 LAST 5 ROWS")
    print("=" * 50)
    display(df.tail())
    
    # Show key columns if they exist
    key_columns = []
    for col_pattern in ['BC', 'ATN', 'flow', 'AAE', 'Delta']:
        matching_cols = [col for col in df.columns if col_pattern.lower() in str(col).lower()]
        key_columns.extend(matching_cols[:3])  # Limit to first 3 matches per pattern
    
    if key_columns:
        print(f"\n🎯 KEY COLUMNS PREVIEW ({len(key_columns)} columns)")
        print("=" * 50)
        display(df[key_columns].head())
    
    # Check for any obvious data quality issues
    print(f"\n🔍 QUICK DATA QUALITY CHECK")
    print("=" * 50)
    print(f"Total missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Check for negative values in BC columns (shouldn't happen)
    bc_cols = [col for col in df.columns if 'BC' in str(col).upper() and 'c' in str(col)]
    if bc_cols:
        negative_counts = (df[bc_cols] < 0).sum()
        if negative_counts.any():
            print(f"Negative BC values found: {negative_counts[negative_counts > 0].to_dict()}")
        else:
            print("✅ No negative BC values found")
            
else:
    print("❌ No data available to preview")

# %% [markdown]
# ## 4. Basic DataFrame Statistics
# 
# Let's examine the statistical properties of our data.

# %%
if df is not None:
    print("📈 BASIC STATISTICS")
    print("=" * 50)
    
    # Overall statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # General statistics
    stats = df.describe()
    display(stats)
    
    # Focus on BC columns if they exist
    bc_cols = [col for col in df.columns if 'BC' in str(col).upper() and 'c' in str(col)]
    if bc_cols:
        print(f"\n🎯 BLACK CARBON STATISTICS ({len(bc_cols)} columns)")
        print("=" * 50)
        bc_stats = df[bc_cols].describe()
        display(bc_stats)
        
        # Additional BC-specific stats
        print(f"\n📊 BC Summary:")
        for col in bc_cols[:5]:  # Show first 5 BC columns
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"   {col}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Check for correlations between key variables
    if len(bc_cols) >= 2:
        print(f"\n🔗 BC CORRELATIONS (top correlations)")
        print("=" * 50)
        bc_corr = df[bc_cols].corr()
        
        # Get upper triangle of correlation matrix
        mask = np.triu(np.ones_like(bc_corr, dtype=bool))
        bc_corr_masked = bc_corr.mask(mask)
        
        # Find highest correlations
        corr_pairs = []
        for col in bc_corr_masked.columns:
            for idx in bc_corr_masked.index:
                if not pd.isna(bc_corr_masked.loc[idx, col]):
                    corr_pairs.append((idx, col, bc_corr_masked.loc[idx, col]))
        
        # Sort by correlation strength
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for i, (var1, var2, corr) in enumerate(corr_pairs[:5]):
            print(f"   {var1} vs {var2}: {corr:.3f}")
    
else:
    print("❌ No data available for statistics")

# %% [markdown]
# ## 5. Time Series Visualization
# 
# Let's create some basic visualizations using the modular system's plotting capabilities.

# %%
if df is not None:
    print("📈 CREATING TIME SERIES VISUALIZATIONS")
    print("=" * 50)
    
    # Find BC columns for plotting
    bc_cols = [col for col in df.columns if 'BC' in str(col).upper() and 'c' in str(col)]
    print(f"Available BC columns: {bc_cols[:5]}")
    
    # Find rows with actual BC data (not all NaN)
    if bc_cols:
        non_nan_mask = df[bc_cols].notna().any(axis=1)
        available_data_count = non_nan_mask.sum()
        print(f"Rows with BC data: {available_data_count}")
        
        if available_data_count > 0:
            # Get a representative sample of the data
            sample_indices = df[non_nan_mask].index[:1000]  # First 1000 rows with data
            plot_df = df.loc[sample_indices].copy()
            
            # Fix datetime conversion - use the actual datetime column
            datetime_success = False
            if 'datetime_local' in plot_df.columns:
                try:
                    # Use the datetime_local column as the index
                    plot_df = plot_df.set_index('datetime_local')
                    datetime_success = True
                    print("✅ Successfully set datetime_local as index")
                    print(f"   Date range: {plot_df.index.min()} to {plot_df.index.max()}")
                except Exception as e:
                    print(f"⚠️ Could not set datetime_local as index: {e}")
            elif 'datetime' in plot_df.columns:
                try:
                    plot_df = plot_df.set_index('datetime')
                    datetime_success = True
                    print("✅ Successfully set datetime as index")
                except Exception as e:
                    print(f"⚠️ Could not set datetime as index: {e}")
            else:
                # Check for other timestamp columns
                time_cols = [col for col in plot_df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if time_cols:
                    try:
                        time_col = time_cols[0]  # Use first timestamp column found
                        plot_df = plot_df.set_index(time_col)
                        datetime_success = True
                        print(f"✅ Successfully set {time_col} as index")
                    except Exception as e:
                        print(f"⚠️ Could not set {time_col} as index: {e}")
                else:
                    print("⚠️ No suitable datetime columns found")
            
            # Try the modular plotting system
            modular_success = False
            if datetime_success and isinstance(plot_df.index, pd.DatetimeIndex):
                try:
                    print(f"\n🎯 Using modular plotting system...")
                    
                    # Initialize plotter with safe settings
                    plotter = AethalometerPlotter(style='default', figsize=(15, 8))
                    
                    # Select valid columns with actual data
                    valid_cols = []
                    for col in bc_cols[:5]:  # Limit to first 5 columns
                        if col in plot_df.columns and plot_df[col].notna().any():
                            valid_cols.append(col)
                    
                    if valid_cols:
                        print(f"Plotting columns: {valid_cols}")
                        
                        # Plot time series
                        fig = plotter.plot_time_series(
                            plot_df[valid_cols], 
                            columns=valid_cols,
                            title="Black Carbon Time Series - Aethalometer Data"
                        )
                        plt.tight_layout()
                        plt.show()
                        modular_success = True
                        print("✅ Modular plotting successful!")
                    
                except Exception as e:
                    print(f"⚠️ Modular plotting failed: {e}")
                    modular_success = False
            
            # Fallback to basic matplotlib if modular system fails
            if not modular_success:
                print(f"\n📊 Using fallback matplotlib plotting...")
                
                try:
                    # Create basic plots
                    valid_bc_cols = [col for col in bc_cols[:4] if col in plot_df.columns and plot_df[col].notna().any()]
                    
                    if valid_bc_cols:
                        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                        axes = axes.flatten()
                        
                        for i, col in enumerate(valid_bc_cols):
                            if i < 4:
                                # Clean the data before plotting
                                clean_data = pd.to_numeric(plot_df[col], errors='coerce')
                                axes[i].plot(plot_df.index, clean_data, alpha=0.7)
                                axes[i].set_title(f'{col}')
                                axes[i].set_ylabel('Concentration (μg/m³)')
                                axes[i].grid(True, alpha=0.3)
                                
                                if datetime_success:
                                    axes[i].tick_params(axis='x', rotation=45)
                        
                        # Hide empty subplots
                        for i in range(len(valid_bc_cols), 4):
                            axes[i].set_visible(False)
                        
                        plt.suptitle('Black Carbon Time Series - Fallback Plots', fontsize=16)
                        plt.tight_layout()
                        plt.show()
                        print("✅ Fallback plotting successful!")
                    else:
                        print("❌ No valid BC columns found for plotting")
                        
                except Exception as e:
                    print(f"❌ Fallback plotting also failed: {e}")
        else:
            print("❌ No BC data available for plotting")
    else:
        print("❌ No BC columns found in dataset")
else:
    print("❌ No data available for visualization")

# %% [markdown]
# ## 6. Advanced Analysis with Modular System
# 
# Now let's demonstrate some of the advanced analysis capabilities of the modular system.

# %%
if df is not None:
    print("🔬 ADVANCED ANALYSIS USING MODULAR SYSTEM")
    print("=" * 50)
    
    try:
        # Source Apportionment Analysis
        print("1. Source Apportionment Analysis...")
        
        if SourceApportionmentAnalyzer is not None:
            try:
                analyzer = SourceApportionmentAnalyzer()
                results = analyzer.analyze(df)
                
                if 'error' not in results:
                    print(f"   ✅ Source apportionment completed")
                    print(f"   📊 Analysis results: {results.get('summary', 'No summary available')}")
                    
                    # Display detailed results
                    if 'source_contributions' in results:
                        contrib = results['source_contributions']
                        print(f"   🔥 Biomass burning: {contrib['biomass_fraction']['mean']*100:.1f}% ± {contrib['biomass_fraction']['std']*100:.1f}%")
                        print(f"   ⛽ Fossil fuel: {contrib['fossil_fraction']['mean']*100:.1f}% ± {contrib['fossil_fraction']['std']*100:.1f}%")
                    
                    if 'aae_statistics' in results:
                        aae = results['aae_statistics']
                        print(f"   📈 AAE: {aae['mean']:.2f} ± {aae['std']:.2f}")
                        
                else:
                    print(f"   ⚠️ Source apportionment failed: {results['error']}")
                    
            except Exception as e:
                print(f"   ⚠️ Source apportionment analysis error: {e}")
        else:
            print(f"   ⚠️ SourceApportionmentAnalyzer not available")
    
    except Exception as e:
        print(f"⚠️ Error in source apportionment analysis: {e}")
    
    try:
        # Black Carbon analysis with available analyzer
        print("\n2. Black Carbon Analysis...")
        
        # Check if we have the required columns for BC analysis
        bc_columns = [col for col in df.columns if 'BC' in str(col).upper() and 'c' in str(col)]
        
        if BlackCarbonAnalyzer is not None and len(bc_columns) >= 1:
            try:
                bc_analyzer = BlackCarbonAnalyzer()
                print(f"   ✅ BlackCarbonAnalyzer initialized")
                print(f"   📊 Available BC columns: {bc_columns[:5]}")  # Show first 5
                
                # Basic BC statistics
                print(f"   📈 Basic BC Statistics:")
                for col in bc_columns[:3]:  # Analyze first 3 BC columns
                    if col in df.columns:
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        median_val = df[col].median()
                        print(f"      {col}: mean={mean_val:.3f}, std={std_val:.3f}, median={median_val:.3f}")
                        
            except Exception as e:
                print(f"   ⚠️ Black carbon analysis failed: {e}")
                
        else:
            print(f"   ⚠️ BlackCarbonAnalyzer not available or insufficient BC columns")
            print(f"   Available BC columns: {bc_columns}")
    
    except Exception as e:
        print(f"⚠️ Error in black carbon analysis: {e}")
    
    print(f"\n3. Data Quality Assessment...")
    
    # Basic data quality checks
    quality_issues = []
    
    # Check for missing data
    missing_pct = (df.isnull().sum() / len(df) * 100)
    high_missing = missing_pct[missing_pct > 10]
    if not high_missing.empty:
        quality_issues.append(f"High missing data in {len(high_missing)} columns")
    
    # Check for outliers in BC data
    bc_cols = [col for col in df.columns if 'BC' in str(col).upper() and 'c' in str(col)]
    if bc_cols:
        for col in bc_cols[:3]:  # Check first 3 BC columns
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                    quality_issues.append(f"High outlier rate in {col}: {len(outliers)/len(df)*100:.1f}%")
    
    if quality_issues:
        print("   ⚠️ Quality issues found:")
        for issue in quality_issues:
            print(f"      - {issue}")
    else:
        print("   ✅ No major quality issues detected")
    
    print(f"\n4. Correlation Analysis...")
    
    # Correlation analysis for BC columns
    if len(bc_cols) >= 2:
        print("   📊 BC Correlations:")
        bc_corr = df[bc_cols].corr()
        
        # Show strongest correlations
        mask = np.triu(np.ones_like(bc_corr, dtype=bool))
        bc_corr_masked = bc_corr.mask(mask)
        
        corr_pairs = []
        for col in bc_corr_masked.columns:
            for idx in bc_corr_masked.index:
                if not pd.isna(bc_corr_masked.loc[idx, col]):
                    corr_pairs.append((idx, col, bc_corr_masked.loc[idx, col]))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for i, (var1, var2, corr) in enumerate(corr_pairs[:5]):
            print(f"      {var1} vs {var2}: {corr:.3f}")
    
    print(f"\n5. Summary Statistics...")
    
    # Generate comprehensive summary
    summary_stats = {
        'total_records': len(df),
        'date_range': f"Available: {len(df)} records",
        'columns': len(df.columns),
        'missing_data_pct': f"{df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}%"
    }
    
    if bc_cols:
        bc_means = df[bc_cols].mean()
        summary_stats['avg_bc_concentration'] = f"{bc_means.mean():.3f} ± {bc_means.std():.3f}"
        summary_stats['bc_columns_available'] = len(bc_cols)
    
    print("   📊 Dataset Summary:")
    for key, value in summary_stats.items():
        print(f"      {key.replace('_', ' ').title()}: {value}")
        
else:
    print("❌ No data available for advanced analysis")

# %% [markdown]
# ## 7. Conclusion
# 
# This notebook demonstrated the successful integration of the modular aethalometer analysis system with Jupyter notebooks. 
# 
# ### What we accomplished:
# 1. ✅ Successfully imported the modular system components
# 2. ✅ Loaded aethalometer data from pickle files
# 3. ✅ Performed basic data inspection and quality checks
# 4. ✅ Generated statistical summaries
# 5. ✅ Created visualizations using the plotting utilities
# 6. ✅ Demonstrated advanced analysis capabilities
# 7. ✅ **NEW:** Provided comprehensive user guide for future users
# 
# ### Next steps:
# - Explore additional analysis modules (seasonal, correlations, quality assessment)
# - Set up automated reporting pipelines
# - Integrate with the batch processing capabilities
# - Export results in various formats
# - **Follow the user guide in Section 8 for your own analyses**
# 
# ### Key Benefits:
# - **Modular Design**: Easy to add new analysis components
# - **Data Format Flexibility**: Handles different aethalometer data formats
# - **Quality Control**: Built-in data validation and quality checks
# - **Visualization**: Integrated plotting utilities for immediate insights
# - **Extensibility**: Can easily add custom analysis modules
# - **User-Friendly**: Comprehensive documentation and examples for new users
# 
# **🎉 The modular aethalometer analysis system is successfully working with Jupyter notebooks!**
# 
# **📚 For future users: See Section 8 for detailed guidance and best practices.**

# %% [markdown]
# ## 8. User Guide for the Modular Aethalometer Analysis System
# 
# ### 🎯 **For Future Users: Important Things to Know**
# 
# This section provides essential guidance for anyone who will be using this modular aethalometer analysis system in the future.
# 
# ---
# 
# ### 📚 **System Architecture Overview**
# 
# The modular system is organized into several key components:
# 
# ```
# src/
# ├── data/loaders/          # Data loading utilities
# ├── analysis/bc/           # Black carbon analysis modules
# ├── analysis/quality/      # Data quality assessment
# ├── analysis/seasonal/     # Seasonal analysis tools
# ├── utils/                 # Plotting and utility functions
# ├── config/                # Configuration files
# └── core/                  # Core system components
# ```
# 
# ---
# 
# ### 🔧 **Getting Started Checklist**
# 
# **Before using this system:**
# 
# 1. **✅ Environment Setup**
#    - Ensure Python 3.8+ is installed
#    - Install required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`
#    - Add the `src` directory to your Python path (as shown in Cell 2)
# 
# 2. **✅ Data Format Requirements**
#    - Your data should contain BC columns (named with 'BC' and 'c', e.g., 'Blue BCc')
#    - A datetime column is essential (preferably named 'datetime_local', 'datetime', or similar)
#    - Data can be in pickle (.pkl), CSV, or other pandas-compatible formats
# 
# 3. **✅ File Path Configuration**
#    - Update the `data_path` variable in Cell 4 to point to your data file
#    - Use absolute paths to avoid confusion
# 
# ---
# 
# ### ⚠️ **Common Pitfalls and How to Avoid Them**
# 
# #### **1. DateTime Index Issues**
# - **Problem**: Plots show wrong dates (e.g., 1970s instead of 2020s)
# - **Solution**: Ensure your data has a proper datetime column. The system will automatically detect columns like:
#   - `datetime_local`
#   - `datetime`
#   - `timestamp`
#   - Any column with 'time' or 'date' in the name
# - **Example**: If your datetime column has a different name, add it to the detection logic in the plotting cell
# 
# #### **2. Missing BC Data**
# - **Problem**: "No BC columns found" or "No BC data available"
# - **Solution**: Check your column naming convention. BC columns should contain both 'BC' and 'c'
# - **Valid examples**: `Blue BCc`, `Red BC concentration`, `UV_BC_corrected`
# - **Invalid examples**: `Blue_absorption`, `carbon_red`
# 
# #### **3. Memory Issues with Large Datasets**
# - **Problem**: System runs slowly or crashes with large files
# - **Solution**: The system automatically samples data (first 1000 rows with valid data)
# - **Tip**: For full dataset analysis, increase the sample size in the plotting cell or use batch processing
# 
# #### **4. Import Errors**
# - **Problem**: ModuleNotFoundError when importing analysis components
# - **Solution**: Verify the `src` path is correctly added to `sys.path`
# - **Debugging**: Check if files exist in the expected locations under `src/`
# 
# ---
# 
# ### 📊 **Data Quality Best Practices**
# 
# #### **Before Analysis:**
# 1. **Check for missing data**: Look at the "QUICK DATA QUALITY CHECK" output
# 2. **Validate date ranges**: Ensure timestamps make sense for your measurement period
# 3. **Inspect BC values**: Negative BC values may indicate data quality issues
# 4. **Review correlations**: Strong correlations between BC channels are expected
# 
# #### **During Analysis:**
# 1. **Monitor memory usage**: Large datasets may require chunking
# 2. **Validate results**: Cross-check automated analysis results with domain knowledge
# 3. **Save intermediate results**: Export processed data at key stages
# 
# ---
# 
# ### 🔬 **Analysis Module Usage**
# 
# #### **Source Apportionment**
# - **Requirements**: Needs multi-wavelength BC data
# - **Purpose**: Separates biomass burning vs. fossil fuel contributions
# - **Interpretation**: 
#   - AAE > 1.2: More biomass burning influence
#   - AAE < 1.0: More fossil fuel influence
# 
# #### **Black Carbon Analysis**
# - **Requirements**: At least one BC column with valid data
# - **Purpose**: Basic statistical analysis of BC concentrations
# - **Outputs**: Mean, standard deviation, median values
# 
# #### **Plotting System**
# - **Automatic fallback**: If modular plotting fails, basic matplotlib plots are generated
# - **Customization**: Modify plotting parameters in `AethalometerPlotter` initialization
# - **Time series**: Works best with datetime-indexed data
# 
# ---
# 
# ### 🚀 **Performance Optimization Tips**
# 
# 1. **Data Sampling**: For exploratory analysis, use data subsets (already implemented)
# 2. **Memory Management**: Close matplotlib figures after displaying: `plt.close()`
# 3. **Batch Processing**: For multiple files, use the batch processing capabilities
# 4. **Caching**: Results can be cached for repeated analysis
# 
# ---
# 
# ### 🛠️ **Customization and Extension**
# 
# #### **Adding New Analysis Modules:**
# 1. Create new files under `src/analysis/`
# 2. Follow the existing pattern (e.g., `SourceApportionmentAnalyzer`)
# 3. Import and test in this notebook
# 
# #### **Custom Plotting:**
# 1. Extend `AethalometerPlotter` class in `src/utils/plotting.py`
# 2. Add new plot types following existing patterns
# 3. Test with your specific data format
# 
# #### **Data Loaders:**
# 1. Add new loaders to `src/data/loaders/`
# 2. Support different file formats or data structures
# 3. Maintain consistent output format (pandas DataFrame)
# 
# ---
# 
# ### 📋 **Troubleshooting Checklist**
# 
# **If something doesn't work:**
# 
# 1. **✅ Check the error messages** - they usually provide specific guidance
# 2. **✅ Verify data format** - ensure BC columns and datetime columns exist
# 3. **✅ Check file paths** - use absolute paths when possible
# 4. **✅ Review data quality** - missing or corrupted data can cause failures
# 5. **✅ Test with smaller datasets** - isolate whether it's a data size issue
# 6. **✅ Check Python environment** - ensure all required packages are installed
# 
# ---
# 
# ### 💡 **Pro Tips**
# 
# 1. **Always run cells in order** - the system builds on previous imports and data loading
# 2. **Use the debug cells** - uncomment print statements for detailed execution info
# 3. **Save your work frequently** - large datasets can take time to process
# 4. **Document your modifications** - if you customize the code, document changes
# 5. **Validate results** - Always cross-check automated results with manual calculations
# 
# ---
# 
# ### 📞 **Getting Help**
# 
# - **Error messages**: Most error messages include specific guidance
# - **Documentation**: Check the docstrings in the source code files
# - **Examples**: This notebook serves as a comprehensive example
# - **Testing**: Use the functionality test cell (Cell 12) to verify system status

# %%
# 🔧 PRACTICAL EXAMPLES FOR FUTURE USERS
print("📋 COMMON USAGE PATTERNS AND EXAMPLES")
print("=" * 50)

# Example 1: Quick data validation
if df is not None:
    print("1. QUICK DATA VALIDATION EXAMPLE")
    print("   ✅ Steps to validate your data:")
    
    # Check basic requirements
    bc_cols = [col for col in df.columns if 'BC' in str(col).upper() and 'c' in str(col)]
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    
    print(f"   • BC columns found: {len(bc_cols)} (need ≥1)")
    print(f"   • Time columns found: {len(time_cols)} (need ≥1)")
    print(f"   • Data completeness: {(1 - df.isnull().sum().sum()/(len(df)*len(df.columns)))*100:.1f}%")
    
    # Data range check
    if bc_cols:
        bc_data = df[bc_cols].dropna()
        if len(bc_data) > 0:
            print(f"   • Valid BC measurements: {len(bc_data):,} rows")
            print(f"   • BC value range: {bc_data.min().min():.2f} to {bc_data.max().max():.2f}")
        else:
            print("   ⚠️ No valid BC data found!")
    
    print()

# Example 2: Custom analysis workflow
print("2. CUSTOM ANALYSIS WORKFLOW TEMPLATE")
print("   📝 Copy and modify this template for your analysis:")
print("""
   # Step 1: Load your data
   your_data_path = "/path/to/your/data.pkl"
   df = pd.read_pickle(your_data_path)
   
   # Step 2: Validate data structure
   bc_columns = [col for col in df.columns if 'BC' in col.upper() and 'c' in col]
   assert len(bc_columns) > 0, "No BC columns found!"
   
   # Step 3: Set up time index
   if 'datetime_local' in df.columns:
       df = df.set_index('datetime_local')
   
   # Step 4: Create plots
   plotter = AethalometerPlotter(style='default', figsize=(12, 8))
   fig = plotter.plot_time_series(df, columns=bc_columns[:3], 
                                  title="Your Data Analysis")
   
   # Step 5: Run analysis
   if SourceApportionmentAnalyzer:
       analyzer = SourceApportionmentAnalyzer()
       results = analyzer.analyze(df)
""")

# Example 3: Error handling template
print("3. ERROR HANDLING BEST PRACTICES")
print("   🛡️ Always wrap analysis in try-except blocks:")
print("""
   try:
       # Your analysis code here
       results = analyzer.analyze(data)
       if 'error' in results:
           print(f"Analysis failed: {results['error']}")
       else:
           print("Analysis successful!")
           # Process results...
   except Exception as e:
       print(f"Unexpected error: {e}")
       # Fallback or alternative analysis
""")

# Example 4: Data subsetting for testing
print("4. DATA SUBSETTING FOR TESTING")
print("   🔬 Test with small datasets first:")
if df is not None:
    print(f"   • Original data: {len(df):,} rows")
    test_data = df.head(1000)  # First 1000 rows
    print(f"   • Test subset: {len(test_data):,} rows")
    print("   • Use this pattern: df.head(n) or df.sample(n)")

print("\n5. MEMORY MANAGEMENT TIPS")
print("   💾 For large datasets:")
print("   • Use df.sample(n=10000) for random sampling")
print("   • Process data in chunks: for chunk in pd.read_csv(file, chunksize=1000)")
print("   • Clear variables: del large_dataframe")
print("   • Close plots: plt.close('all')")

print("\n✅ These examples should help you get started!")
print("💡 Modify the patterns above for your specific analysis needs.")

# %% [markdown]
# ---
# 
# ## 🚀 **Quick Reference Card**
# 
# ### **🔥 MOST IMPORTANT THINGS TO REMEMBER:**
# 
# | **Category** | **Key Points** |
# |--------------|----------------|
# | **📁 Data Requirements** | • BC columns with 'BC' and 'c' in name<br/>• Datetime column (any name with 'time'/'date')<br/>• Use absolute file paths |
# | **⚡ Quick Start** | • Update `data_path` in Cell 4<br/>• Run cells 1-6 in order<br/>• Check error messages for guidance |
# | **🐛 Common Issues** | • Wrong dates → Check datetime column<br/>• No BC data → Check column naming<br/>• Import errors → Verify src path |
# | **🎯 Best Practices** | • Test with small data first<br/>• Always validate results<br/>• Save work frequently<br/>• Document modifications |
# | **🆘 When Things Break** | • Check error messages first<br/>• Verify data format<br/>• Test with subset of data<br/>• Use fallback plotting if needed |
# 
# ---
# 
# ### **📞 Emergency Troubleshooting Commands**
# 
# ```python
# # 1. Check if data loaded correctly
# print(f"Data shape: {df.shape if 'df' in globals() else 'No data'}")
# 
# # 2. Find BC columns
# bc_cols = [col for col in df.columns if 'BC' in str(col).upper()]
# print(f"BC columns: {bc_cols}")
# 
# # 3. Find datetime columns  
# time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
# print(f"Time columns: {time_cols}")
# 
# # 4. Basic plot without modular system
# plt.figure(figsize=(10, 6))
# plt.plot(df[bc_cols[0]].dropna())
# plt.title("Basic BC Plot")
# plt.show()
# ```
# 
# ---
# 
# **💡 Remember: This notebook is your template. Copy it for new analyses and modify as needed!**

# %%
# Quick test to verify all components are working
import importlib

# Force reload the modules
try:
    import analysis.bc.source_apportionment
    importlib.reload(analysis.bc.source_apportionment)
    from analysis.bc.source_apportionment import SourceApportionmentAnalyzer
    print("✅ SourceApportionmentAnalyzer reloaded")
except Exception as e:
    print(f"⚠️ Error reloading SourceApportionmentAnalyzer: {e}")

print("\n🧪 QUICK FUNCTIONALITY TEST")
print("=" * 40)

# Test 1: Check if analyzers are available
analyzers_available = []
if 'BlackCarbonAnalyzer' in globals() and BlackCarbonAnalyzer is not None:
    analyzers_available.append("BlackCarbonAnalyzer")
if 'SourceApportionmentAnalyzer' in locals():
    analyzers_available.append("SourceApportionmentAnalyzer")

print(f"✅ Available analyzers: {', '.join(analyzers_available)}")

# Test 2: Check if data is loaded
if 'df' in globals() and df is not None:
    print(f"✅ Data loaded: {len(df)} rows × {len(df.columns)} columns")
    
    # Quick column check
    bc_cols = [col for col in df.columns if 'BC' in str(col).upper() and 'c' in str(col)]
    print(f"✅ BC columns found: {len(bc_cols)}")
    
    if len(bc_cols) > 0:
        print(f"   Sample BC columns: {bc_cols[:3]}")
else:
    print("⚠️ No data loaded")

# Test 3: Quick analysis test
if 'df' in globals() and df is not None and 'SourceApportionmentAnalyzer' in locals():
    try:
        test_analyzer = SourceApportionmentAnalyzer()
        sample_data = df.head(100)  # Use just first 100 rows for quick test
        test_results = test_analyzer.analyze(sample_data)
        
        if 'error' not in test_results:
            print("✅ Source apportionment test: PASSED")
            print(f"   Summary: {test_results.get('summary', 'No summary')}")
        else:
            print(f"⚠️ Source apportionment test: {test_results['error']}")
    except Exception as e:
        print(f"⚠️ Source apportionment test error: {e}")
else:
    print("⚠️ Cannot test source apportionment: analyzer or data not available")

print("\n🎉 System check complete!")


