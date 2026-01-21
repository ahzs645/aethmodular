"""Ethiopian seasonal analysis for FTIR and Aethalometer data"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, date
from ...core.base import BaseAnalyzer
from ...data.processors.validation import validate_columns_exist

class EthiopianSeasonAnalyzer(BaseAnalyzer):
    """
    Analyzer for Ethiopian climate-specific seasonal patterns
    
    Ethiopian seasons:
    - Dry Season (Bega): October - May
    - Belg Rainy Season: March - May (overlaps with dry season end)
    - Kiremt Rainy Season: June - September
    """
    
    def __init__(self):
        super().__init__("EthiopianSeasonAnalyzer")
        
        # Ethiopian season definitions
        self.seasons = {
            'dry_season_bega': {
                'name': 'Dry Season (Bega)',
                'months': [10, 11, 12, 1, 2, 3, 4, 5],  # Oct-May
                'primary_months': [11, 12, 1, 2],  # Peak dry
                'description': 'Main dry season with minimal rainfall'
            },
            'belg_rainy': {
                'name': 'Belg Rainy Season',
                'months': [3, 4, 5],  # Mar-May
                'primary_months': [4, 5],  # Peak Belg
                'description': 'Short rainy season (overlaps with dry season end)'
            },
            'kiremt_rainy': {
                'name': 'Kiremt Rainy Season', 
                'months': [6, 7, 8, 9],  # Jun-Sep
                'primary_months': [7, 8],  # Peak Kiremt
                'description': 'Main rainy season with heavy rainfall'
            }
        }
    
    def analyze(self, data: pd.DataFrame, date_column: str = 'timestamp',
                target_columns: List[str] = None) -> Dict[str, Any]:
        """
        Perform seasonal analysis on Ethiopian climate patterns
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with datetime column
        date_column : str
            Name of the datetime column
        target_columns : List[str]
            Columns to analyze seasonally (if None, analyzes all numeric columns)
            
        Returns:
        --------
        Dict[str, Any]
            Seasonal analysis results
        """
        validate_columns_exist(data, [date_column])
        
        # Ensure datetime column
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
        
        # Get target columns
        if target_columns is None:
            target_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            validate_columns_exist(data, target_columns)
        
        # Add seasonal classifications
        data_with_seasons = self._classify_seasons(data, date_column)
        
        # Perform seasonal statistics
        seasonal_stats = self._calculate_seasonal_statistics(
            data_with_seasons, target_columns
        )
        
        # Seasonal MAC analysis (if applicable)
        mac_analysis = self._seasonal_mac_analysis(data_with_seasons)
        
        # Seasonal patterns and trends
        seasonal_patterns = self._analyze_seasonal_patterns(
            data_with_seasons, target_columns, date_column
        )
        
        # Climate-specific analytics
        climate_analytics = self._climate_specific_analytics(
            data_with_seasons, target_columns
        )
        
        results = {
            'season_definitions': self.seasons,
            'data_info': {
                'total_samples': len(data),
                'date_range': {
                    'start': data[date_column].min().strftime('%Y-%m-%d'),
                    'end': data[date_column].max().strftime('%Y-%m-%d')
                },
                'years_covered': sorted(data[date_column].dt.year.unique().tolist()),
                'target_columns': target_columns
            },
            'seasonal_statistics': seasonal_stats,
            'mac_analysis': mac_analysis,
            'seasonal_patterns': seasonal_patterns,
            'climate_analytics': climate_analytics,
            'classified_data': data_with_seasons
        }
        
        return results
    
    def _classify_seasons(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Classify each data point by Ethiopian season"""
        data_copy = data.copy()
        
        # Extract month from datetime
        data_copy['month'] = data_copy[date_column].dt.month
        data_copy['year'] = data_copy[date_column].dt.year
        
        # Classify seasons
        def classify_season(month):
            if month in self.seasons['kiremt_rainy']['months']:
                return 'kiremt_rainy'
            elif month in self.seasons['belg_rainy']['months']:
                # Belg overlaps with dry season, but we prioritize Belg
                return 'belg_rainy'
            else:
                return 'dry_season_bega'
        
        data_copy['season'] = data_copy['month'].apply(classify_season)
        data_copy['season_name'] = data_copy['season'].map(
            {k: v['name'] for k, v in self.seasons.items()}
        )
        
        # Additional classifications
        data_copy['is_rainy_season'] = data_copy['season'].isin(['kiremt_rainy', 'belg_rainy'])
        data_copy['is_dry_season'] = data_copy['season'] == 'dry_season_bega'
        
        # Peak season indicators
        def is_peak_season(row):
            season = row['season']
            month = row['month']
            return month in self.seasons[season]['primary_months']
        
        data_copy['is_peak_season'] = data_copy.apply(is_peak_season, axis=1)
        
        return data_copy
    
    def _calculate_seasonal_statistics(self, data: pd.DataFrame, 
                                     target_columns: List[str]) -> Dict[str, Any]:
        """Calculate statistics for each season"""
        seasonal_stats = {}
        
        for season_key, season_info in self.seasons.items():
            season_data = data[data['season'] == season_key]
            
            if len(season_data) == 0:
                seasonal_stats[season_key] = {
                    'name': season_info['name'],
                    'sample_count': 0,
                    'error': 'No data for this season'
                }
                continue
            
            season_stats = {
                'name': season_info['name'],
                'sample_count': len(season_data),
                'months': season_info['months'],
                'date_range': {
                    'start': season_data.index.min() if hasattr(season_data.index, 'min') else 'N/A',
                    'end': season_data.index.max() if hasattr(season_data.index, 'max') else 'N/A'
                },
                'statistics': {}
            }
            
            # Calculate statistics for each target column
            for col in target_columns:
                if col in season_data.columns:
                    col_data = season_data[col].dropna()
                    
                    if len(col_data) > 0:
                        season_stats['statistics'][col] = {
                            'count': int(len(col_data)),
                            'mean': float(col_data.mean()),
                            'median': float(col_data.median()),
                            'std': float(col_data.std()),
                            'min': float(col_data.min()),
                            'max': float(col_data.max()),
                            'percentiles': {
                                '25th': float(col_data.quantile(0.25)),
                                '75th': float(col_data.quantile(0.75))
                            }
                        }
                    else:
                        season_stats['statistics'][col] = {
                            'count': 0,
                            'error': 'No valid data'
                        }
            
            seasonal_stats[season_key] = season_stats
        
        return seasonal_stats
    
    def _seasonal_mac_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform season-specific MAC calculations if FTIR data is present"""
        mac_analysis = {}
        
        # Check if we have FTIR data for MAC calculation
        ftir_columns = ['fabs', 'ec_ftir']
        has_ftir = all(col in data.columns for col in ftir_columns)
        
        if not has_ftir:
            return {'error': 'FTIR data not available for MAC analysis'}
        
        for season_key, season_info in self.seasons.items():
            season_data = data[data['season'] == season_key]
            
            if len(season_data) < 5:  # Need minimum samples for MAC
                mac_analysis[season_key] = {
                    'name': season_info['name'],
                    'error': f'Insufficient data: {len(season_data)} samples'
                }
                continue
            
            # Calculate seasonal MAC
            valid_mask = (season_data['fabs'] > 0) & (season_data['ec_ftir'] > 0)
            valid_data = season_data[valid_mask]
            
            if len(valid_data) < 3:
                mac_analysis[season_key] = {
                    'name': season_info['name'],
                    'error': f'Insufficient valid FTIR data: {len(valid_data)} samples'
                }
                continue
            
            fabs = valid_data['fabs']
            ec = valid_data['ec_ftir']
            
            # Calculate MAC using multiple methods
            individual_mac = fabs / ec
            ratio_mac = fabs.sum() / ec.sum()
            
            # Seasonal regression
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(ec, fabs)
            
            mac_analysis[season_key] = {
                'name': season_info['name'],
                'sample_count': len(valid_data),
                'individual_mac': {
                    'mean': float(individual_mac.mean()),
                    'std': float(individual_mac.std()),
                    'median': float(individual_mac.median())
                },
                'ratio_mac': float(ratio_mac),
                'regression_mac': {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'std_error': float(std_err)
                },
                'fabs_stats': {
                    'mean': float(fabs.mean()),
                    'std': float(fabs.std())
                },
                'ec_stats': {
                    'mean': float(ec.mean()),
                    'std': float(ec.std())
                }
            }
        
        # Compare MAC values across seasons
        mac_comparison = self._compare_seasonal_mac(mac_analysis)
        mac_analysis['seasonal_comparison'] = mac_comparison
        
        return mac_analysis
    
    def _compare_seasonal_mac(self, mac_analysis: Dict) -> Dict[str, Any]:
        """Compare MAC values across seasons"""
        if not mac_analysis or all('error' in v for v in mac_analysis.values()):
            return {'error': 'No valid seasonal MAC data for comparison'}
        
        # Extract MAC values
        individual_macs = {}
        ratio_macs = {}
        regression_macs = {}
        
        for season, data in mac_analysis.items():
            if 'error' not in data:
                individual_macs[season] = data['individual_mac']['mean']
                ratio_macs[season] = data['ratio_mac']
                regression_macs[season] = data['regression_mac']['slope']
        
        comparison = {}
        
        # Compare individual MAC means
        if individual_macs:
            values = list(individual_macs.values())
            comparison['individual_mac_comparison'] = {
                'values': individual_macs,
                'statistics': {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'cv_percent': float(np.std(values) / np.mean(values) * 100),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values))
                }
            }
        
        # Similar comparisons for other MAC methods
        for mac_type, mac_dict in [('ratio_mac', ratio_macs), ('regression_mac', regression_macs)]:
            if mac_dict:
                values = list(mac_dict.values())
                comparison[f'{mac_type}_comparison'] = {
                    'values': mac_dict,
                    'statistics': {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'cv_percent': float(np.std(values) / np.mean(values) * 100),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'range': float(np.max(values) - np.min(values))
                    }
                }
        
        return comparison
    
    def _analyze_seasonal_patterns(self, data: pd.DataFrame, target_columns: List[str],
                                 date_column: str) -> Dict[str, Any]:
        """Analyze patterns and trends within seasons"""
        patterns = {}
        
        # Monthly patterns within seasons
        monthly_patterns = {}
        for month in range(1, 13):
            month_data = data[data['month'] == month]
            if len(month_data) > 0:
                month_stats = {}
                for col in target_columns:
                    if col in month_data.columns:
                        col_data = month_data[col].dropna()
                        if len(col_data) > 0:
                            month_stats[col] = {
                                'mean': float(col_data.mean()),
                                'count': int(len(col_data))
                            }
                
                monthly_patterns[month] = {
                    'season': month_data['season'].iloc[0] if len(month_data) > 0 else None,
                    'sample_count': len(month_data),
                    'statistics': month_stats
                }
        
        patterns['monthly_patterns'] = monthly_patterns
        
        # Yearly trends within seasons
        yearly_trends = {}
        for season_key in self.seasons.keys():
            season_data = data[data['season'] == season_key]
            
            if len(season_data) == 0:
                continue
            
            yearly_stats = {}
            for year in sorted(season_data['year'].unique()):
                year_season_data = season_data[season_data['year'] == year]
                
                year_stats = {}
                for col in target_columns:
                    if col in year_season_data.columns:
                        col_data = year_season_data[col].dropna()
                        if len(col_data) > 0:
                            year_stats[col] = {
                                'mean': float(col_data.mean()),
                                'count': int(len(col_data))
                            }
                
                yearly_stats[year] = {
                    'sample_count': len(year_season_data),
                    'statistics': year_stats
                }
            
            yearly_trends[season_key] = yearly_stats
        
        patterns['yearly_trends'] = yearly_trends
        
        return patterns
    
    def _climate_specific_analytics(self, data: pd.DataFrame, 
                                  target_columns: List[str]) -> Dict[str, Any]:
        """Climate-specific analytics for Ethiopian conditions"""
        analytics = {}
        
        # Rainy vs Dry season comparison
        rainy_data = data[data['is_rainy_season']]
        dry_data = data[data['is_dry_season']]
        
        rainy_dry_comparison = {}
        
        for col in target_columns:
            if col in data.columns:
                rainy_values = rainy_data[col].dropna()
                dry_values = dry_data[col].dropna()
                
                if len(rainy_values) > 0 and len(dry_values) > 0:
                    # Statistical comparison
                    from scipy import stats
                    t_stat, p_value = stats.ttest_ind(rainy_values, dry_values)
                    
                    rainy_dry_comparison[col] = {
                        'rainy_season': {
                            'mean': float(rainy_values.mean()),
                            'std': float(rainy_values.std()),
                            'count': int(len(rainy_values))
                        },
                        'dry_season': {
                            'mean': float(dry_values.mean()),
                            'std': float(dry_values.std()),
                            'count': int(len(dry_values))
                        },
                        'comparison': {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant_difference': bool(p_value < 0.05),
                            'rainy_vs_dry_ratio': float(rainy_values.mean() / dry_values.mean()) if dry_values.mean() != 0 else np.inf
                        }
                    }
        
        analytics['rainy_dry_comparison'] = rainy_dry_comparison
        
        # Peak season analysis
        peak_data = data[data['is_peak_season']]
        if len(peak_data) > 0:
            peak_analysis = {}
            for col in target_columns:
                if col in peak_data.columns:
                    col_data = peak_data[col].dropna()
                    if len(col_data) > 0:
                        peak_analysis[col] = {
                            'mean': float(col_data.mean()),
                            'std': float(col_data.std()),
                            'count': int(len(col_data))
                        }
            
            analytics['peak_season_analysis'] = peak_analysis
        
        # Ethiopian climate-specific insights
        insights = self._generate_climate_insights(rainy_dry_comparison)
        analytics['climate_insights'] = insights
        
        return analytics
    
    def _generate_climate_insights(self, rainy_dry_comparison: Dict) -> List[str]:
        """Generate insights specific to Ethiopian climate"""
        insights = []
        
        for col, comparison in rainy_dry_comparison.items():
            if 'comparison' in comparison:
                ratio = comparison['comparison']['rainy_vs_dry_ratio']
                significant = comparison['comparison']['significant_difference']
                
                if significant:
                    if ratio > 1.2:
                        insights.append(
                            f"{col} is significantly higher during rainy seasons "
                            f"({ratio:.1f}x higher) - suggests rain-related sources/effects"
                        )
                    elif ratio < 0.8:
                        insights.append(
                            f"{col} is significantly lower during rainy seasons "
                            f"({1/ratio:.1f}x lower) - suggests rain washout or dry season sources"
                        )
                    else:
                        insights.append(
                            f"{col} shows significant but moderate seasonal variation "
                            f"(ratio: {ratio:.2f})"
                        )
                else:
                    insights.append(
                        f"{col} shows no significant seasonal variation "
                        f"(p-value: {comparison['comparison']['p_value']:.3f})"
                    )
        
        return insights
