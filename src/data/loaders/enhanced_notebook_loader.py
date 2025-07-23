# src/data/loaders/enhanced_notebook_loader.py
"""
Enhanced data loader that simplifies notebook usage
Incorporates all the complex loading logic with fallbacks
"""

import os
import sys
import pickle
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
import warnings

from config.notebook_config import NotebookConfig

class EnhancedNotebookLoader:
    """
    Simplified loader for notebook usage with intelligent fallbacks
    """
    
    def __init__(self, config: NotebookConfig):
        """
        Initialize with configuration
        
        Parameters:
        -----------
        config : NotebookConfig
            Configuration object
        """
        self.config = config
        self.modular_available = False
        self.modular_components = {}
        
        # Try to setup modular system
        self._setup_modular_system()
    
    def _setup_modular_system(self):
        """Setup modular system with intelligent path detection"""
        
        print("📦 Setting up modular system...")
        
        # Find and add src directory to path
        current_dir = Path.cwd()
        
        # Look for src directory in current, parent, or grandparent directories
        src_candidates = [
            current_dir / 'src',
            current_dir.parent / 'src',
            current_dir.parent.parent / 'src'
        ]
        
        src_path = None
        for candidate in src_candidates:
            if candidate.exists():
                src_path = str(candidate.resolve())
                break
        
        if src_path and src_path not in sys.path:
            sys.path.insert(0, src_path)
            print(f"✅ Added {src_path} to Python path")
        
        # Dictionary to store successfully imported components
        imported_components = {
            'loaders': {},
            'analysis': {},
            'utils': {}
        }
        
        # Try importing components with graceful fallbacks
        try:
            from data.loaders.aethalometer import (
                AethalometerPKLLoader, 
                AethalometerCSVLoader,
                load_aethalometer_data
            )
            imported_components['loaders'].update({
                'AethalometerPKLLoader': AethalometerPKLLoader,
                'AethalometerCSVLoader': AethalometerCSVLoader,
                'load_aethalometer_data': load_aethalometer_data
            })
            print("✅ Aethalometer loaders imported")
        except ImportError as e:
            print(f"⚠️ Aethalometer loaders not available: {e}")
        
        try:
            from data.loaders.database import FTIRHIPSLoader
            imported_components['loaders']['FTIRHIPSLoader'] = FTIRHIPSLoader
            print("✅ Database loader imported")
        except ImportError as e:
            print(f"⚠️ Database loader not available: {e}")
        
        try:
            from utils.plotting import AethalometerPlotter
            imported_components['utils']['AethalometerPlotter'] = AethalometerPlotter
            print("✅ Plotting utilities imported")
        except ImportError as e:
            print(f"⚠️ Plotting utilities not available: {e}")
        
        try:
            from config.plotting import setup_plotting_style
            setup_plotting_style()
            print("✅ Plotting style configured")
        except ImportError as e:
            print(f"⚠️ Plotting style config not available: {e}")
        
        # Store results
        if any(imported_components.values()):
            self.modular_available = True
            self.modular_components = imported_components
            success_count = sum(len(v) for v in imported_components.values())
            print(f"✅ Successfully imported {success_count} modular components")
        else:
            print("⚠️ No modular components available, using fallback methods")
    
    def load_aethalometer_data(self, 
                             dataset_name: str = 'pkl_data',
                             site_filter: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Load aethalometer data with intelligent fallbacks
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset in config.aethalometer_files
        site_filter : str, optional
            Filter data by site
            
        Returns:
        --------
        tuple
            (DataFrame, summary_dict)
        """
        
        file_path = self.config.aethalometer_files.get(dataset_name)
        if not file_path:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"📁 Loading {dataset_name}: {Path(file_path).name}")
        
        # Try modular system first
        if self.modular_available and 'load_aethalometer_data' in self.modular_components['loaders']:
            try:
                return self._load_with_modular_system(file_path, site_filter)
            except Exception as e:
                print(f"⚠️ Modular loading failed: {e}")
                print("🔄 Falling back to direct loading...")
        
        # Fallback to direct loading
        return self._load_with_fallback(file_path)
    
    def _load_with_modular_system(self, 
                                file_path: str, 
                                site_filter: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Load using modular system"""
        
        load_function = self.modular_components['loaders']['load_aethalometer_data']
        
        df = load_function(
            file_path,
            output_format=self.config.output_format,
            site_filter=site_filter,
            set_datetime_index=True
        )
        
        if df is None or len(df) == 0:
            raise ValueError("No data loaded from modular system")
        
        # Generate comprehensive summary
        summary = self._generate_summary(df, file_path, 'modular')
        
        print(f"✅ Modular load: {df.shape[0]:,} rows × {df.shape[1]} columns")
        self._print_summary(summary)
        
        return df, summary
    
    def _load_with_fallback(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Load using fallback methods"""
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pkl':
                df = self._load_pkl_fallback(file_path)
            elif file_ext == '.csv':
                df = self._load_csv_fallback(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            if df is None or len(df) == 0:
                return None, {}
            
            # Fix datetime index if needed
            df = self._ensure_datetime_index(df)
            
            # Apply format conversion if needed
            if self.config.output_format == 'jpl':
                df = self._convert_to_jpl_format(df)
            
            summary = self._generate_summary(df, file_path, 'fallback')
            
            print(f"✅ Fallback load: {df.shape[0]:,} rows × {df.shape[1]} columns")
            self._print_summary(summary)
            
            return df, summary
            
        except Exception as e:
            print(f"❌ Fallback loading failed: {e}")
            return None, {}
    
    def _load_pkl_fallback(self, file_path: str) -> pd.DataFrame:
        """Load PKL file with fallback"""
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        return df
    
    def _load_csv_fallback(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with enhanced datetime handling"""
        df = pd.read_csv(file_path)
        
        # Handle timezone conversion for ETAD data
        if 'Time (UTC)' in df.columns:
            df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], utc=True)
            df['Time (Local)'] = df['Time (UTC)'].dt.tz_convert('Africa/Addis_Ababa')
            df.set_index('Time (Local)', inplace=True)
        else:
            # Try to find and set datetime index using other common column names
            datetime_candidates = ['datetime', 'timestamp', 'Time', 'Date', 'datetime_local', 'datetime_utc']
            for col in datetime_candidates:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        if df[col].notna().sum() > len(df) * 0.8:  # At least 80% valid
                            df.set_index(col, inplace=True)
                            break
                    except:
                        continue
        
        return df
    
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has proper datetime index"""
        
        if isinstance(df.index, pd.DatetimeIndex):
            return df.sort_index()
        
        # Find datetime columns
        datetime_candidates = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime']):
                datetime_candidates.append(col)
        
        print(f"🔍 Found datetime candidates: {datetime_candidates}")
        
        # Try to create datetime index
        for col in datetime_candidates:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].notna().sum() > len(df) * 0.8:  # At least 80% valid
                    df = df.set_index(col).sort_index()
                    print(f"✅ Using {col} as datetime index")
                    return df
            except Exception:
                continue
        
        # Try combining date and time columns
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        time_cols = [col for col in df.columns if 'time' in col.lower() and 'date' not in col.lower()]
        
        if date_cols and time_cols:
            try:
                date_col, time_col = date_cols[0], time_cols[0]
                datetime_combined = pd.to_datetime(
                    df[date_col].astype(str) + ' ' + df[time_col].astype(str), 
                    errors='coerce'
                )
                if datetime_combined.notna().sum() > len(df) * 0.8:
                    df['datetime_combined'] = datetime_combined
                    df = df.set_index('datetime_combined').sort_index()
                    print(f"✅ Combined {date_col} + {time_col} for datetime index")
                    return df
            except Exception:
                pass
        
        print("⚠️ Could not create datetime index")
        return df
    
    def _convert_to_jpl_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert column names to JPL format"""
        
        column_mapping = {
            'IR BCc': 'IR.BCc',
            'Blue BCc': 'Blue.BCc', 
            'Green BCc': 'Green.BCc',
            'Red BCc': 'Red.BCc',
            'UV BCc': 'UV.BCc',
            'Biomass BCc': 'Biomass.BCc',
            'Fossil fuel BCc': 'Fossil.fuel.BCc',
        }
        
        rename_dict = {}
        for std_col, jpl_col in column_mapping.items():
            if std_col in df.columns:
                rename_dict[std_col] = jpl_col
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
            print(f"🔧 Converted {len(rename_dict)} columns to JPL format")
        
        return df
    
    def _generate_summary(self, df: pd.DataFrame, file_path: str, method: str) -> Dict:
        """Generate data summary"""
        
        return {
            'file_name': Path(file_path).name,
            'file_type': Path(file_path).suffix,
            'method': method,
            'format': self.config.output_format,
            'shape': df.shape,
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'bc_columns': [col for col in df.columns if '.BCc' in col or 'BC' in col.upper()],
            'atn_columns': [col for col in df.columns if '.ATN' in col or 'ATN' in col.upper()],
            'time_range': (df.index.min(), df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None,
            'missing_data_pct': (df.isnull().sum().sum() / df.size) * 100,
            'has_datetime_index': isinstance(df.index, pd.DatetimeIndex)
        }
    
    def _print_summary(self, summary: Dict):
        """Print formatted summary"""
        
        print(f"📊 Method: {summary['method']}")
        print(f"📊 Format: {summary['format']}")
        print(f"📊 Memory: {summary['memory_mb']:.2f} MB")
        print(f"🧮 BC columns: {len(summary['bc_columns'])}")
        print(f"📈 ATN columns: {len(summary['atn_columns'])}")
        
        if summary['time_range']:
            print(f"📅 Time range: {summary['time_range'][0]} to {summary['time_range'][1]}")
    
    def load_ftir_hips_data(self) -> Optional[pd.DataFrame]:
        """
        Load FTIR/HIPS data with intelligent fallbacks
        
        Returns:
        --------
        pd.DataFrame or None
            Loaded FTIR/HIPS data
        """
        
        print(f"🗃️ Loading FTIR/HIPS data for site {self.config.site_code}...")
        
        if not os.path.exists(self.config.ftir_db_path):
            print(f"❌ FTIR database not found: {self.config.ftir_db_path}")
            return None
        
        # Try modular system first
        if self.modular_available and 'FTIRHIPSLoader' in self.modular_components['loaders']:
            try:
                return self._load_ftir_with_modular_system()
            except Exception as e:
                print(f"⚠️ Modular FTIR loading failed: {e}")
                print("🔄 Falling back to direct database query...")
        
        # Fallback to direct database access
        return self._load_ftir_with_fallback()
    
    def _load_ftir_with_modular_system(self) -> pd.DataFrame:
        """Load FTIR data using modular system"""
        
        FTIRHIPSLoader = self.modular_components['loaders']['FTIRHIPSLoader']
        loader = FTIRHIPSLoader(self.config.ftir_db_path)
        
        # Check available sites
        available_sites = loader.get_available_sites()
        print(f"📊 Available sites: {available_sites}")
        
        if self.config.site_code not in available_sites:
            print(f"⚠️ Site '{self.config.site_code}' not found in database")
            return None
        
        df = loader.load(self.config.site_code)
        
        if len(df) > 0:
            print(f"✅ Modular FTIR load: {len(df)} samples")
            print(f"📅 Date range: {df['sample_date'].min()} to {df['sample_date'].max()}")
        
        return df
    
    def _load_ftir_with_fallback(self) -> Optional[pd.DataFrame]:
        """Load FTIR data using direct database query"""
        
        try:
            conn = sqlite3.connect(self.config.ftir_db_path)
            
            query = """
            SELECT 
                f.filter_id, f.sample_date, f.site_code, f.filter_type,
                m.volume_m3, m.ec_ftir, m.ec_ftir_mdl, m.oc_ftir, m.oc_ftir_mdl,
                m.fabs, m.fabs_mdl, m.fabs_uncertainty, m.ftir_batch_id
            FROM filters f
            JOIN ftir_sample_measurements m ON f.filter_id = m.filter_id
            WHERE f.site_code = ?
            ORDER BY f.sample_date
            """
            
            df = pd.read_sql_query(query, conn, params=(self.config.site_code,))
            conn.close()
            
            if len(df) > 0:
                df['sample_date'] = pd.to_datetime(df['sample_date'])
                print(f"✅ Fallback FTIR load: {len(df)} samples")
                return df
            else:
                print("❌ No FTIR data found")
                return None
                
        except Exception as e:
            print(f"❌ Fallback FTIR loading failed: {e}")
            return None
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all configured datasets
        
        Returns:
        --------
        dict
            Dictionary of dataset_name -> DataFrame
        """
        
        print("📁 Loading all datasets...")
        datasets = {}
        
        # Load aethalometer datasets
        for dataset_name in self.config.aethalometer_files.keys():
            print(f"\n{'='*50}")
            print(f"📊 Loading {dataset_name}")
            print("="*50)
            
            df, summary = self.load_aethalometer_data(dataset_name)
            if df is not None:
                datasets[dataset_name] = df
                print(f"✅ {dataset_name} loaded successfully")
            else:
                print(f"❌ Failed to load {dataset_name}")
        
        # Load FTIR data
        print(f"\n{'='*50}")
        print("🗃️ Loading FTIR/HIPS data")
        print("="*50)
        
        ftir_data = self.load_ftir_hips_data()
        if ftir_data is not None:
            datasets['ftir_hips'] = ftir_data
            print("✅ FTIR/HIPS data loaded successfully")
        else:
            print("❌ Failed to load FTIR/HIPS data")
        
        print(f"\n📊 Loading summary: {len(datasets)} datasets loaded")
        return datasets