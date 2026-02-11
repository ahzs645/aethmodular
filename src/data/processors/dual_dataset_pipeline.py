# Enhanced Dual-Dataset Processing Pipeline
# Creates two complementary datasets: high-resolution and FTIR-matched

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os

class DualDatasetProcessor:
    """
    Enhanced processor that creates two complementary datasets:
    1. High-resolution aethalometer data (all periods, full temporal resolution)
    2. FTIR-matched dataset (matched periods only, 9am-to-9am averaged)
    """
    
    def __init__(self, config, setup=None):
        self.config = config
        self.setup = setup
        self.datasets = {}
        self.processing_log = []
    
    def process_dual_datasets(self, 
                            apply_ethiopia_fix: bool = True,
                            save_outputs: bool = True,
                            output_dir: str = 'processed_data') -> Dict[str, pd.DataFrame]:
        """
        Main processing pipeline that creates both datasets
        
        Returns:
            Dict containing 'high_resolution' and 'ftir_matched' datasets
        """
        
        print("🚀 DUAL-DATASET PROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load and clean raw aethalometer data
        print("\n📁 Step 1: Loading and cleaning raw aethalometer data...")
        raw_data = self._load_raw_aethalometer_data()
        
        # Step 2: Apply enhanced PKL processing (cleaning + Ethiopia corrections)
        print("\n🔧 Step 2: Applying enhanced processing and Ethiopia corrections...")
        cleaned_data = self._apply_enhanced_processing(raw_data, apply_ethiopia_fix)
        
        # Step 3: Load FTIR data and identify matching periods
        print("\n🧪 Step 3: Loading FTIR data and identifying matching periods...")
        ftir_data = self._load_ftir_data()
        matching_periods = self._identify_matching_periods(cleaned_data, ftir_data)
        
        # Step 4: Create high-resolution dataset (all data + DEMA)
        print("\n📈 Step 4: Creating high-resolution dataset...")
        high_res_data = self._create_high_resolution_dataset(cleaned_data)
        
        # Step 5: Create FTIR-matched dataset (matched periods + 9am-to-9am + DEMA)
        print("\n🔗 Step 5: Creating FTIR-matched dataset...")
        ftir_matched_data = self._create_ftir_matched_dataset(
            cleaned_data, ftir_data, matching_periods
        )
        
        # Step 6: Save outputs
        if save_outputs:
            print("\n💾 Step 6: Saving processed datasets...")
            self._save_datasets(high_res_data, ftir_matched_data, output_dir)
        
        # Store in instance for further analysis
        self.datasets = {
            'high_resolution': high_res_data,
            'ftir_matched': ftir_matched_data,
            'raw_data': raw_data,
            'cleaned_data': cleaned_data,
            'ftir_data': ftir_data
        }
        
        self._print_summary()
        
        return self.datasets
    
    def _load_raw_aethalometer_data(self) -> pd.DataFrame:
        """Load raw aethalometer data"""
        
        # Use existing setup if available
        if self.setup:
            datasets = self.setup.load_all_data()
            raw_data = self.setup.get_dataset('pkl_data')
        else:
            # Fallback to direct loading
            file_path = self.config.aethalometer_files['pkl_data']
            raw_data = pd.read_pickle(file_path)
        
        # Ensure datetime_local column exists
        if 'datetime_local' not in raw_data.columns:
            if raw_data.index.name == 'datetime_local':
                raw_data = raw_data.reset_index()
            elif hasattr(raw_data.index, 'tz'):
                raw_data['datetime_local'] = raw_data.index
                raw_data = raw_data.reset_index(drop=True)
        
        print(f"✅ Loaded raw data: {raw_data.shape}")
        print(f"📅 Date range: {raw_data['datetime_local'].min()} to {raw_data['datetime_local'].max()}")
        
        return raw_data
    
    def _apply_enhanced_processing(self, raw_data: pd.DataFrame, apply_ethiopia_fix: bool) -> pd.DataFrame:
        """Apply enhanced PKL processing with Ethiopia corrections"""
        
        # Import the enhanced processing function
        try:
            from src.data.qc.enhanced_pkl_processing import process_pkl_data_enhanced
        except ImportError:
            print("⚠️ Enhanced PKL processing not available, using basic cleaning...")
            return self._basic_cleaning(raw_data)
        
        # Apply enhanced processing
        cleaned_data = process_pkl_data_enhanced(
            raw_data,
            wavelengths_to_filter=['IR', 'Blue'],  # Adjust as needed
            apply_ethiopia_fix=apply_ethiopia_fix,
            site_code='ETAD' if apply_ethiopia_fix else None,
            verbose=True
        )
        
        print(f"✅ Enhanced processing complete: {cleaned_data.shape}")
        
        # Log what corrections were applied
        ethiopia_cols = [col for col in cleaned_data.columns 
                        if any(x in col for x in ['corrected', 'manual', 'optimized'])]
        if ethiopia_cols:
            print(f"🔧 Ethiopia corrections applied: {len(ethiopia_cols)} columns")
        
        return cleaned_data
    
    def _basic_cleaning(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning fallback if enhanced processing unavailable"""
        
        df = raw_data.copy()
        
        # Basic datetime handling
        df['datetime_local'] = pd.to_datetime(df['datetime_local'])
        
        # Filter to 2022+
        df = df.loc[df['datetime_local'].dt.year >= 2022]
        
        # Set serial number if not present
        if 'Serial number' not in df.columns:
            df['Serial number'] = "MA350-0238"
        
        print(f"✅ Basic cleaning complete: {df.shape}")
        return df
    
    def _load_ftir_data(self) -> pd.DataFrame:
        """Load FTIR data"""
        
        # Use existing setup if available
        if self.setup:
            ftir_data = self.setup.get_dataset('ftir_data')
            if ftir_data is not None:
                return ftir_data
        
        # Fallback to direct database loading
        try:
            import sqlite3
            conn = sqlite3.connect(self.config.ftir_db_path)
            
            query = """
            SELECT 
                f.filter_id, f.sample_date, f.site_code,
                m.ec_ftir, m.oc_ftir
            FROM filters f
            JOIN ftir_sample_measurements m ON f.filter_id = m.filter_id
            WHERE f.site_code = ?
            """
            
            ftir_data = pd.read_sql_query(query, conn, params=[self.config.site_code])
            ftir_data['sample_date'] = pd.to_datetime(ftir_data['sample_date'])
            
            print(f"✅ Loaded FTIR data: {len(ftir_data)} samples")
            print(f"📅 FTIR date range: {ftir_data['sample_date'].min()} to {ftir_data['sample_date'].max()}")
            
            return ftir_data
            
        except Exception as e:
            print(f"❌ Could not load FTIR data: {e}")
            return pd.DataFrame()
    
    def _identify_matching_periods(self, aeth_data: pd.DataFrame, ftir_data: pd.DataFrame) -> pd.DatetimeIndex:
        """Identify periods where both aethalometer and FTIR data exist"""
        
        if len(ftir_data) == 0:
            print("⚠️ No FTIR data available - no matching periods identified")
            return pd.DatetimeIndex([])
        
        # Get FTIR sample dates
        ftir_dates = pd.to_datetime(ftir_data['sample_date']).dt.date
        
        # Get aethalometer dates  
        aeth_dates = pd.to_datetime(aeth_data['datetime_local']).dt.date
        
        # Find overlapping dates
        matching_dates = set(ftir_dates).intersection(set(aeth_dates))
        matching_periods = pd.to_datetime(list(matching_dates))
        
        print(f"🔗 Found {len(matching_periods)} matching periods")
        print(f"📅 Matching range: {matching_periods.min()} to {matching_periods.max()}")
        
        return matching_periods
    
    def _create_high_resolution_dataset(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """Create high-resolution dataset with DEMA smoothing applied to all data"""
        
        print("  📈 Processing high-resolution dataset...")
        
        # Start with cleaned data
        high_res_data = cleaned_data.copy()
        
        # Apply DEMA smoothing to all data
        high_res_data = self._apply_dema_smoothing(
            high_res_data, 
            dataset_type="high_resolution"
        )
        
        print(f"  ✅ High-resolution dataset created: {high_res_data.shape}")
        print(f"  📅 Full date range: {high_res_data['datetime_local'].min()} to {high_res_data['datetime_local'].max()}")
        
        return high_res_data
    
    def _create_ftir_matched_dataset(self, 
                                   cleaned_data: pd.DataFrame, 
                                   ftir_data: pd.DataFrame,
                                   matching_periods: pd.DatetimeIndex) -> pd.DataFrame:
        """Create FTIR-matched dataset with 9am-to-9am averaging and DEMA smoothing"""
        
        print("  🔗 Processing FTIR-matched dataset...")
        
        if len(matching_periods) == 0:
            print("  ⚠️ No matching periods - returning empty dataset")
            return pd.DataFrame()
        
        # Step 1: Filter aethalometer data to matching periods only
        aeth_matched = self._filter_to_matching_periods(cleaned_data, matching_periods)
        
        # Step 2: Apply 9am-to-9am resampling
        aeth_daily = self._resample_9am_to_9am(aeth_matched)
        
        # Step 3: Apply DEMA smoothing to the resampled data
        aeth_daily_smoothed = self._apply_dema_smoothing(
            aeth_daily, 
            dataset_type="ftir_matched"
        )
        
        # Step 4: Merge with FTIR data
        ftir_matched_data = self._merge_with_ftir(aeth_daily_smoothed, ftir_data)
        
        print(f"  ✅ FTIR-matched dataset created: {ftir_matched_data.shape}")
        if len(ftir_matched_data) > 0:
            print(f"  📅 Matched date range: {ftir_matched_data.index.min()} to {ftir_matched_data.index.max()}")
        
        return ftir_matched_data
    
    def _filter_to_matching_periods(self, data: pd.DataFrame, matching_periods: pd.DatetimeIndex) -> pd.DataFrame:
        """Filter aethalometer data to periods that have FTIR measurements"""
        
        # Convert matching periods to date range with some tolerance
        matching_dates = set(matching_periods.date)
        
        # Filter data to matching dates
        data_dates = pd.to_datetime(data['datetime_local']).dt.date
        mask = data_dates.isin(matching_dates)
        
        filtered_data = data.loc[mask].copy()
        
        print(f"    📊 Filtered to matching periods: {len(data):,} -> {len(filtered_data):,} rows")
        
        return filtered_data
    
    def _resample_9am_to_9am(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply 9am-to-9am resampling"""
        
        df_work = data.copy()
        
        # Set datetime as index
        df_work['datetime_local'] = pd.to_datetime(df_work['datetime_local'])
        df_work = df_work.set_index('datetime_local')
        
        # Localize timezone if needed
        if df_work.index.tz is None:
            timezone = getattr(self.config, 'timezone', 'Africa/Addis_Ababa')
            df_work.index = df_work.index.tz_localize(timezone)
        
        # Shift time back by 9 hours so 9am becomes start of day
        df_shifted = df_work.copy()
        df_shifted.index = df_shifted.index - pd.Timedelta(hours=9)
        
        # Get numeric columns only
        numeric_cols = df_shifted.select_dtypes(include=[np.number]).columns
        
        # Resample to daily, calculating mean and count
        daily_means = df_shifted[numeric_cols].resample('D').mean()
        daily_counts = df_shifted[numeric_cols].resample('D').count()
        
        # Filter out days with insufficient data (require at least 4 hours of data)
        min_hours = 4
        for col in numeric_cols:
            insufficient_data = daily_counts[col] < min_hours
            daily_means.loc[insufficient_data, col] = np.nan
        
        # Shift index forward by 9 hours to get 9am timestamps
        daily_means.index = daily_means.index + pd.Timedelta(hours=9)
        daily_means.index.name = 'datetime_local'
        
        print(f"    ⏰ 9am-to-9am resampling: {len(df_work):,} -> {len(daily_means)} daily averages")
        
        return daily_means
    
    def _apply_dema_smoothing(self, data: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Apply DEMA smoothing appropriate for the dataset type"""
        
        print(f"    🔄 Applying DEMA smoothing ({dataset_type})...")
        
        # Try to use modular DEMA implementation
        try:
            from src.data.qc.enhanced_pkl_processing import EnhancedPKLProcessor
            
            processor = EnhancedPKLProcessor(verbose=False)
            smoothed_data = processor.apply_dema_smoothing(data)
            
            print(f"    ✅ DEMA smoothing applied using enhanced processor")
            
        except ImportError:
            print(f"    ⚠️ Enhanced DEMA not available, using basic smoothing...")
            smoothed_data = self._basic_dema_smoothing(data)
        
        return smoothed_data
    
    def _basic_dema_smoothing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic DEMA smoothing fallback"""
        
        df = data.copy()
        
        # Apply simple exponential smoothing to BC columns
        alpha = 0.125  # Standard DEMA alpha
        bc_cols = [col for col in df.columns if 'BC' in col and 'smoothed' not in col]
        
        for col in bc_cols:
            if col in df.columns and df[col].notna().sum() > 10:
                # Simple exponential smoothing
                smoothed = df[col].ewm(alpha=alpha).mean()
                df[f"{col}_smoothed"] = smoothed
        
        print(f"    ✅ Basic DEMA applied to {len(bc_cols)} BC columns")
        
        return df
    
    def _merge_with_ftir(self, aeth_data: pd.DataFrame, ftir_data: pd.DataFrame) -> pd.DataFrame:
        """Merge aethalometer daily averages with FTIR data"""
        
        if len(ftir_data) == 0:
            print("    ⚠️ No FTIR data to merge")
            return aeth_data
        
        # Prepare FTIR data for merging
        ftir_for_merge = ftir_data.copy()
        
        # Set FTIR timestamps to 9am on sample dates
        ftir_timestamps = (pd.to_datetime(ftir_for_merge['sample_date']).dt.normalize() + 
                          pd.Timedelta(hours=9))
        
        # Localize timezone to match aethalometer data
        timezone = getattr(self.config, 'timezone', 'Africa/Addis_Ababa')
        ftir_timestamps = ftir_timestamps.dt.tz_localize(timezone)
        
        ftir_for_merge.index = ftir_timestamps
        ftir_for_merge.index.name = 'datetime_local'
        
        # Merge datasets
        merged_data = pd.merge(
            aeth_data,
            ftir_for_merge,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        print(f"    🔗 Merged with FTIR: {len(aeth_data)} aeth + {len(ftir_for_merge)} FTIR -> {len(merged_data)} merged")
        
        return merged_data
    
    def _save_datasets(self, 
                      high_res_data: pd.DataFrame, 
                      ftir_matched_data: pd.DataFrame,
                      output_dir: str):
        """Save both datasets as pkl files"""
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save high-resolution dataset
        high_res_path = Path(output_dir) / f"aethalometer_high_resolution_{self.config.site_code}.pkl"
        high_res_data.to_pickle(high_res_path)
        print(f"  💾 Saved high-resolution data: {high_res_path}")
        
        # Save FTIR-matched dataset
        if len(ftir_matched_data) > 0:
            ftir_matched_path = Path(output_dir) / f"aethalometer_ftir_matched_{self.config.site_code}.pkl"
            ftir_matched_data.to_pickle(ftir_matched_path)
            print(f"  💾 Saved FTIR-matched data: {ftir_matched_path}")
        
        # Save as CSV for easy inspection
        high_res_csv = Path(output_dir) / f"aethalometer_high_resolution_{self.config.site_code}.csv"
        high_res_data.to_csv(high_res_csv)
        
        if len(ftir_matched_data) > 0:
            ftir_csv = Path(output_dir) / f"aethalometer_ftir_matched_{self.config.site_code}.csv"
            ftir_matched_data.to_csv(ftir_csv)
        
        print(f"  📄 CSV versions also saved for inspection")
    
    def _print_summary(self):
        """Print processing summary"""
        
        print(f"\n🎯 DUAL-DATASET PROCESSING SUMMARY")
        print(f"=" * 50)
        
        if 'high_resolution' in self.datasets:
            hr_data = self.datasets['high_resolution']
            print(f"📈 High-Resolution Dataset:")
            print(f"   Shape: {hr_data.shape}")
            print(f"   Date range: {hr_data['datetime_local'].min()} to {hr_data['datetime_local'].max()}")
            
            # Count smoothed columns
            smoothed_cols = [col for col in hr_data.columns if 'smoothed' in col]
            print(f"   DEMA columns: {len(smoothed_cols)}")
        
        if 'ftir_matched' in self.datasets and len(self.datasets['ftir_matched']) > 0:
            fm_data = self.datasets['ftir_matched']
            print(f"\n🔗 FTIR-Matched Dataset:")
            print(f"   Shape: {fm_data.shape}")
            print(f"   Date range: {fm_data.index.min()} to {fm_data.index.max()}")
            
            # Count FTIR columns
            ftir_cols = [col for col in fm_data.columns if any(x in col.lower() for x in ['ec_ftir', 'oc_ftir'])]
            print(f"   FTIR columns: {len(ftir_cols)}")
            
            # Match efficiency
            if 'high_resolution' in self.datasets:
                hr_days = len(pd.to_datetime(self.datasets['high_resolution']['datetime_local']).dt.date.unique())
                fm_days = len(fm_data)
                match_rate = (fm_days / hr_days) * 100
                print(f"   Match rate: {match_rate:.1f}% ({fm_days}/{hr_days} days)")
        
        print(f"\n✅ Both datasets ready for analysis!")
        print(f"   📈 Use high-resolution data for time series analysis")
        print(f"   🔗 Use FTIR-matched data for method comparison")


# Usage example function
def run_dual_dataset_processing(config, setup=None, ethiopia_fix=True):
    """
    Convenience function to run the dual dataset processing
    
    Args:
        config: Your existing NotebookConfig
        setup: Your existing setup object (optional)
        ethiopia_fix: Whether to apply Ethiopia corrections
    
    Returns:
        Dict with both datasets
    """
    
    processor = DualDatasetProcessor(config, setup)
    
    datasets = processor.process_dual_datasets(
        apply_ethiopia_fix=ethiopia_fix,
        save_outputs=True,
        output_dir='processed_data'
    )
    
    return datasets
