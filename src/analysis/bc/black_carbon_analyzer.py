"""
Black Carbon analyzer for JPL format aethalometer data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src.core.base import BaseAnalyzer
from src.core.exceptions import DataValidationError

class BlackCarbonAnalyzer(BaseAnalyzer):
    """
    Analyzer for black carbon data in JPL repository format
    Handles IR.BCc, Biomass.BCc, Fossil.fuel.BCc columns
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        try:
            super().__init__("BlackCarbonAnalyzer")
        except TypeError:
            # Fallback if BaseAnalyzer doesn't require name parameter
            super().__init__()
        self.required_columns = ['IR.BCc']
        self.optional_columns = ['Biomass.BCc', 'Fossil.fuel.BCc', 'datetime_local']
        
    def analyze(self, data: pd.DataFrame, 
                time_resolution: str = 'hourly',
                include_trends: bool = True) -> Dict[str, Any]:
        """
        Analyze black carbon data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with JPL format columns
        time_resolution : str
            Time resolution for analysis ('hourly', 'daily', 'monthly')
        include_trends : bool
            Whether to include trend analysis
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results
        """
        
        # Validate input
        self._validate_input_data(data)
        
        # Clean data
        clean_data = self._clean_data(data.copy())
        
        if len(clean_data) == 0:
            raise DataValidationError("No valid data remaining after cleaning")
        
        results = {
            'data_info': self._get_data_info(clean_data),
            'basic_statistics': self._calculate_basic_statistics(clean_data),
            'source_apportionment': self._analyze_source_apportionment(clean_data),
            'temporal_patterns': self._analyze_temporal_patterns(clean_data, time_resolution),
        }
        
        if include_trends and 'datetime_local' in clean_data.columns:
            results['trends'] = self._analyze_trends(clean_data)
        
        return results
    
    def _validate_input_data(self, data: pd.DataFrame):
        """Validate input data has required columns"""
        missing_required = [col for col in self.required_columns if col not in data.columns]
        if missing_required:
            raise DataValidationError(f"Missing required columns: {missing_required}")
        
        if len(data) == 0:
            raise DataValidationError("Input data is empty")
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter the data"""
        # Remove negative values
        bc_columns = [col for col in data.columns if '.BCc' in col]
        for col in bc_columns:
            if col in data.columns:
                data = data[data[col] >= 0]
        
        # Remove extreme outliers (>99.9th percentile)
        for col in bc_columns:
            if col in data.columns and len(data[col].dropna()) > 0:
                threshold = data[col].quantile(0.999)
                data = data[data[col] <= threshold]
        
        # Ensure datetime is properly formatted
        if 'datetime_local' in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data['datetime_local']):
                data['datetime_local'] = pd.to_datetime(data['datetime_local'])
        
        return data
    
    def _get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic data information"""
        info = {
            'total_samples': len(data),
            'date_range': None,
            'available_bc_types': []
        }
        
        # Get date range
        if 'datetime_local' in data.columns:
            info['date_range'] = {
                'start': data['datetime_local'].min(),
                'end': data['datetime_local'].max(),
                'duration_days': (data['datetime_local'].max() - data['datetime_local'].min()).days
            }
        
        # Check available BC types
        bc_types = ['IR.BCc', 'Biomass.BCc', 'Fossil.fuel.BCc', 'Blue.BCc', 'Green.BCc', 'Red.BCc', 'UV.BCc']
        for bc_type in bc_types:
            if bc_type in data.columns:
                valid_count = data[bc_type].notna().sum()
                if valid_count > 0:
                    info['available_bc_types'].append({
                        'type': bc_type,
                        'valid_samples': valid_count,
                        'coverage': valid_count / len(data)
                    })
        
        return info
    
    def _calculate_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for BC measurements"""
        stats = {}
        
        bc_columns = [col for col in data.columns if '.BCc' in col]
        
        for col in bc_columns:
            if col in data.columns and data[col].notna().sum() > 0:
                values = data[col].dropna()
                stats[col] = {
                    'count': len(values),
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'percentiles': {
                        '25th': float(values.quantile(0.25)),
                        '75th': float(values.quantile(0.75)),
                        '90th': float(values.quantile(0.90)),
                        '95th': float(values.quantile(0.95))
                    }
                }
        
        return stats
    
    def _analyze_source_apportionment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze source apportionment if biomass and fossil fuel BC are available"""
        results = {
            'available': False,
            'biomass_fraction': None,
            'fossil_fraction': None,
            'statistics': {}
        }
        
        if 'Biomass.BCc' in data.columns and 'Fossil.fuel.BCc' in data.columns:
            # Filter for valid data
            valid_data = data[
                data['Biomass.BCc'].notna() & 
                data['Fossil.fuel.BCc'].notna() &
                (data['Biomass.BCc'] >= 0) &
                (data['Fossil.fuel.BCc'] >= 0)
            ].copy()
            
            if len(valid_data) > 0:
                valid_data['Total.BCc'] = valid_data['Biomass.BCc'] + valid_data['Fossil.fuel.BCc']
                
                # Calculate fractions
                valid_data['Biomass.fraction'] = valid_data['Biomass.BCc'] / valid_data['Total.BCc']
                valid_data['Fossil.fraction'] = valid_data['Fossil.fuel.BCc'] / valid_data['Total.BCc']
                
                # Remove infinite/NaN values
                valid_data = valid_data[
                    np.isfinite(valid_data['Biomass.fraction']) & 
                    np.isfinite(valid_data['Fossil.fraction'])
                ]
                
                if len(valid_data) > 0:
                    results['available'] = True
                    results['statistics'] = {
                        'sample_count': len(valid_data),
                        'mean_biomass_fraction': float(valid_data['Biomass.fraction'].mean()),
                        'mean_fossil_fraction': float(valid_data['Fossil.fraction'].mean()),
                        'biomass_contribution': {
                            'mean_ug_m3': float(valid_data['Biomass.BCc'].mean()),
                            'median_ug_m3': float(valid_data['Biomass.BCc'].median()),
                            'std_ug_m3': float(valid_data['Biomass.BCc'].std())
                        },
                        'fossil_contribution': {
                            'mean_ug_m3': float(valid_data['Fossil.fuel.BCc'].mean()),
                            'median_ug_m3': float(valid_data['Fossil.fuel.BCc'].median()),
                            'std_ug_m3': float(valid_data['Fossil.fuel.BCc'].std())
                        }
                    }
        
        return results
    
    def _analyze_temporal_patterns(self, data: pd.DataFrame, resolution: str) -> Dict[str, Any]:
        """Analyze temporal patterns in BC data"""
        results = {
            'resolution': resolution,
            'patterns': {}
        }
        
        if 'datetime_local' not in data.columns:
            results['error'] = 'No datetime column available'
            return results
        
        # Set datetime as index for resampling
        data_time = data.set_index('datetime_local')
        
        # Choose resampling frequency
        freq_map = {
            'hourly': 'H',
            'daily': 'D',
            'monthly': 'M'
        }
        freq = freq_map.get(resolution, 'H')
        
        bc_columns = [col for col in data_time.columns if '.BCc' in col]
        
        for col in bc_columns:
            if col in data_time.columns:
                # Resample data
                resampled = data_time[col].resample(freq).mean()
                resampled = resampled.dropna()
                
                if len(resampled) > 0:
                    results['patterns'][col] = {
                        'mean_by_period': float(resampled.mean()),
                        'std_by_period': float(resampled.std()),
                        'periods_with_data': len(resampled),
                        'temporal_coverage': len(resampled) / len(data_time) if len(data_time) > 0 else 0
                    }
                    
                    # Add hourly patterns if resolution is hourly
                    if resolution == 'hourly' and len(resampled) >= 24:
                        hourly_means = resampled.groupby(resampled.index.hour).mean()
                        results['patterns'][col]['hourly_cycle'] = {
                            'peak_hour': int(hourly_means.idxmax()),
                            'min_hour': int(hourly_means.idxmin()),
                            'peak_value': float(hourly_means.max()),
                            'min_value': float(hourly_means.min())
                        }
        
        return results
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze long-term trends in BC data"""
        results = {}
        
        if 'datetime_local' not in data.columns:
            return {'error': 'No datetime column available'}
        
        # Set datetime as index
        data_time = data.set_index('datetime_local')
        bc_columns = [col for col in data_time.columns if '.BCc' in col]
        
        for col in bc_columns:
            if col in data_time.columns and data_time[col].notna().sum() > 30:  # Need at least 30 points
                try:
                    # Simple linear trend analysis
                    values = data_time[col].dropna()
                    
                    if len(values) > 30:
                        # Convert datetime to numeric for trend calculation
                        time_numeric = (values.index - values.index[0]).total_seconds() / (24 * 3600)  # Days
                        
                        # Linear regression
                        z = np.polyfit(time_numeric, values.values, 1)
                        trend_slope = z[0]  # Change per day
                        
                        # Calculate trend statistics
                        results[col] = {
                            'trend_slope_per_day': float(trend_slope),
                            'trend_slope_per_year': float(trend_slope * 365),
                            'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                            'relative_trend_percent_per_year': float((trend_slope * 365 / values.mean()) * 100),
                            'data_points': len(values),
                            'time_span_days': float(time_numeric.max())
                        }
                        
                except Exception as e:
                    results[col] = {'error': f'Trend analysis failed: {str(e)}'}
        
        return results


class MultiWavelengthBCAnalyzer(BaseAnalyzer):
    """
    Analyzer for multi-wavelength BC data (IR, Blue, Green, Red, UV)
    """
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze multi-wavelength BC data to calculate spectral properties
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with wavelength-specific BC columns
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including AAE calculations
        """
        
        # Define wavelengths (nm) for each channel
        wavelengths = {
            'UV.BCc': 370,
            'Blue.BCc': 470,
            'Green.BCc': 520,
            'Red.BCc': 660,
            'IR.BCc': 880
        }
        
        available_wavelengths = {k: v for k, v in wavelengths.items() if k in data.columns}
        
        if len(available_wavelengths) < 2:
            return {'error': 'Need at least 2 wavelengths for spectral analysis'}
        
        results = {
            'available_wavelengths': available_wavelengths,
            'spectral_analysis': self._calculate_aae(data, available_wavelengths)
        }
        
        return results
    
    def _calculate_aae(self, data: pd.DataFrame, wavelengths: Dict[str, int]) -> Dict[str, Any]:
        """Calculate Absorption Angstrom Exponent (AAE)"""
        
        if len(wavelengths) < 2:
            return {'error': 'Need at least 2 wavelengths'}
        
        # Get pairs for AAE calculation
        wl_pairs = list(wavelengths.items())
        
        results = {}
        
        # Calculate AAE for each pair
        for i in range(len(wl_pairs)):
            for j in range(i + 1, len(wl_pairs)):
                col1, wl1 = wl_pairs[i]
                col2, wl2 = wl_pairs[j]
                
                if col1 in data.columns and col2 in data.columns:
                    # Filter valid data
                    valid_mask = (data[col1] > 0) & (data[col2] > 0) & data[col1].notna() & data[col2].notna()
                    valid_data = data[valid_mask]
                    
                    if len(valid_data) > 0:
                        # Calculate AAE
                        aae_values = -np.log(valid_data[col1] / valid_data[col2]) / np.log(wl1 / wl2)
                        aae_values = aae_values[np.isfinite(aae_values)]
                        
                        if len(aae_values) > 0:
                            pair_name = f"{col1}_{col2}"
                            results[pair_name] = {
                                'wavelength_pair': [wl1, wl2],
                                'aae_mean': float(aae_values.mean()),
                                'aae_median': float(aae_values.median()),
                                'aae_std': float(aae_values.std()),
                                'valid_samples': len(aae_values)
                            }
        
        return results
