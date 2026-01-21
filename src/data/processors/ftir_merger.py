"""FTIR data merger for aethalometer analysis"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

try:
    from core.base import BaseAnalyzer
    from core.exceptions import DataValidationError
except ImportError:
    # Fallback for when running from different locations
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.base import BaseAnalyzer
    from core.exceptions import DataValidationError


class FTIRMerger(BaseAnalyzer):
    """Merge FTIR data with aethalometer measurements"""
    
    def __init__(self, 
                 time_tolerance: str = '5min',
                 interpolation_method: str = 'linear'):
        """
        Initialize FTIR merger
        
        Args:
            time_tolerance: Maximum time difference for matching data points
            interpolation_method: Method for interpolating missing values
        """
        self.time_tolerance = time_tolerance
        self.interpolation_method = interpolation_method
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        This is a placeholder - actual merging should be done with load_and_merge_ftir
        
        Args:
            data: Aethalometer DataFrame
            
        Returns:
            Dict with original data (no FTIR data to merge)
        """
        return {
            'merged_data': data,
            'ftir_files_merged': [],
            'merge_summary': "No FTIR files provided for merging"
        }
    
    def load_and_merge_ftir(self, 
                           aethalometer_data: pd.DataFrame,
                           ftir_files: Union[str, List[str], Path, List[Path]]) -> Dict[str, Any]:
        """
        Load FTIR data and merge with aethalometer data
        
        Args:
            aethalometer_data: Main aethalometer DataFrame
            ftir_files: Path(s) to FTIR CSV files
            
        Returns:
            Dict containing merged data and merge statistics
        """
        if isinstance(ftir_files, (str, Path)):
            ftir_files = [ftir_files]
        
        # Convert to Path objects
        ftir_files = [Path(f) for f in ftir_files]
        
        merged_data = aethalometer_data.copy()
        merge_stats = {
            'files_processed': [],
            'columns_added': [],
            'merge_success': [],
            'total_ftir_points': 0,
            'matched_points': 0
        }
        
        for ftir_file in ftir_files:
            try:
                ftir_df = self._load_ftir_file(ftir_file)
                merged_data, file_stats = self._merge_single_ftir(merged_data, ftir_df, ftir_file.name)
                
                merge_stats['files_processed'].append(ftir_file.name)
                merge_stats['columns_added'].extend(file_stats['new_columns'])
                merge_stats['merge_success'].append(True)
                merge_stats['total_ftir_points'] += file_stats['ftir_points']
                merge_stats['matched_points'] += file_stats['matched_points']
                
            except Exception as e:
                print(f"Failed to merge {ftir_file.name}: {e}")
                merge_stats['files_processed'].append(ftir_file.name)
                merge_stats['merge_success'].append(False)
        
        return {
            'merged_data': merged_data,
            'merge_stats': merge_stats,
            'merge_summary': self._generate_merge_summary(merge_stats)
        }
    
    def _load_ftir_file(self, ftir_file: Path) -> pd.DataFrame:
        """Load a single FTIR CSV file"""
        if not ftir_file.exists():
            raise FileNotFoundError(f"FTIR file not found: {ftir_file}")
        
        try:
            # Try different common CSV formats
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(ftir_file, sep=sep)
                    if len(df.columns) > 1:  # Successfully parsed
                        break
                except:
                    continue
            else:
                raise ValueError("Could not parse CSV file with common separators")
            
            # Try to identify datetime column
            datetime_column = self._identify_datetime_column(df)
            if datetime_column:
                df[datetime_column] = pd.to_datetime(df[datetime_column])
                df.set_index(datetime_column, inplace=True)
            else:
                raise DataValidationError("No datetime column found in FTIR data")
            
            return df
            
        except Exception as e:
            raise DataValidationError(f"Failed to load FTIR file {ftir_file}: {e}")
    
    def _identify_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the datetime column in FTIR data"""
        datetime_candidates = ['datetime', 'timestamp', 'time', 'date']
        
        # First check for exact matches
        for col in df.columns:
            if col.lower() in datetime_candidates:
                return col
        
        # Then check for partial matches
        for col in df.columns:
            for candidate in datetime_candidates:
                if candidate in col.lower():
                    return col
        
        # Try to find any column that looks like datetime
        for col in df.columns:
            try:
                pd.to_datetime(df[col].iloc[0])
                return col
            except:
                continue
        
        return None
    
    def _merge_single_ftir(self, 
                          aethalometer_data: pd.DataFrame, 
                          ftir_data: pd.DataFrame, 
                          filename: str) -> tuple:
        """Merge a single FTIR dataset with aethalometer data"""
        
        # Ensure aethalometer data has datetime index
        if not isinstance(aethalometer_data.index, pd.DatetimeIndex):
            if 'datetime' in aethalometer_data.columns:
                aethalometer_data = aethalometer_data.set_index('datetime')
            else:
                raise DataValidationError("Aethalometer data must have datetime index or column")
        
        # Resample FTIR data to match aethalometer frequency if needed
        ftir_resampled = self._resample_ftir_data(ftir_data, aethalometer_data.index)
        
        # Add prefix to FTIR columns to avoid conflicts
        prefix = f"FTIR_{filename.split('.')[0]}_"
        ftir_resampled.columns = [f"{prefix}{col}" for col in ftir_resampled.columns]
        
        # Merge using pandas merge_asof for time-based joining
        merged = pd.merge_asof(
            aethalometer_data.sort_index(),
            ftir_resampled.sort_index(),
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(self.time_tolerance),
            direction='nearest'
        )
        
        # Calculate merge statistics
        new_columns = list(ftir_resampled.columns)
        matched_points = merged[new_columns].notna().any(axis=1).sum()
        
        file_stats = {
            'new_columns': new_columns,
            'ftir_points': len(ftir_data),
            'matched_points': matched_points
        }
        
        return merged, file_stats
    
    def _resample_ftir_data(self, ftir_data: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Resample FTIR data to match target frequency"""
        # Determine target frequency
        if len(target_index) > 1:
            freq = pd.infer_freq(target_index)
            if freq is None:
                # Estimate frequency from median time difference
                time_diffs = target_index.to_series().diff().dropna()
                median_diff = time_diffs.median()
                freq = f"{int(median_diff.total_seconds())}S"
        else:
            freq = '1min'  # Default frequency
        
        # Resample FTIR data
        if self.interpolation_method == 'linear':
            resampled = ftir_data.resample(freq).interpolate(method='linear')
        elif self.interpolation_method == 'nearest':
            resampled = ftir_data.resample(freq).nearest()
        else:
            resampled = ftir_data.resample(freq).mean()
        
        return resampled
    
    def _generate_merge_summary(self, merge_stats: Dict[str, Any]) -> str:
        """Generate a summary of the merge operation"""
        successful_merges = sum(merge_stats['merge_success'])
        total_files = len(merge_stats['files_processed'])
        
        summary = f"""
FTIR Merge Summary:
==================
Files processed: {total_files}
Successful merges: {successful_merges}
Failed merges: {total_files - successful_merges}
Total FTIR data points: {merge_stats['total_ftir_points']}
Matched data points: {merge_stats['matched_points']}
New columns added: {len(merge_stats['columns_added'])}

Files processed:
{chr(10).join(f"  - {file}: {'✓' if success else '✗'}" 
              for file, success in zip(merge_stats['files_processed'], merge_stats['merge_success']))}
        """
        return summary.strip()
