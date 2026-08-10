"""
Filter Sample Mapping Module

This module provides tools for mapping filter sampling periods to aethalometer
data quality periods. It supports loading filter data from databases and
analyzing overlaps with high-quality aethalometer data.

Author: AethModular Team
Created: 2025-01-12
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FilterSampleMapper:
    """
    Mapper for filter sampling periods to aethalometer quality periods.
    
    This class provides methods to:
    - Load filter sample data from databases
    - Map filter sampling periods to quality classifications
    - Analyze overlaps between filter samples and high-quality periods
    - Export filtered datasets for analysis
    """
    
    def __init__(self):
        """Initialize the filter sample mapper."""
        self.filter_data = None
        self.mapping_results = None
        
    def load_filter_data_from_db(self, db_path: str, 
                                site_code: str = 'ETAD') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load filter sample data from SQLite database.
        
        Parameters:
        -----------
        db_path : str
            Path to the SQLite database
        site_code : str, default 'ETAD'
            Site code to filter data
            
        Returns:
        --------
        tuple
            (etad_data, ftir_data) DataFrames with filter sample information
        """
        logger.info(f"Loading filter sample data from database: {db_path}")
        
        if not Path(db_path).exists():
            logger.error(f"Database file not found: {db_path}")
            return self._create_empty_dataframes()
        
        try:
            conn = sqlite3.connect(db_path)
            
            query = """
            SELECT f.filter_id, 
                   f.sample_date AS SampleDate, 
                   m.ec_ftir AS EC_FTIR,
                   m.oc_ftir AS OC_FTIR,
                   m.fabs AS Fabs,
                   f.site_code AS Site
            FROM filters f
            JOIN ftir_sample_measurements m USING(filter_id)
            WHERE f.site_code = ?
            ORDER BY f.sample_date;
            """
            
            combined_data = pd.read_sql_query(query, conn, params=[site_code])
            
            if len(combined_data) == 0:
                logger.warning(f"No data found for site {site_code}")
                return self._create_empty_dataframes()
            
            # Convert date column to datetime
            combined_data['SampleDate'] = pd.to_datetime(combined_data['SampleDate'])
            
            # Create separate dataframes for compatibility
            etad_data = combined_data.copy()
            ftir_data = combined_data.copy()
            ftir_data.rename(columns={'SampleDate': 'date'}, inplace=True)
            
            conn.close()
            
            logger.info(f"Loaded {len(etad_data)} {site_code} samples from database")
            
            # Store for later use
            self.filter_data = {
                'etad': etad_data,
                'ftir': ftir_data,
                'site_code': site_code,
                'db_path': db_path
            }
            
            return etad_data, ftir_data
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return self._create_empty_dataframes()
    
    def _create_empty_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create empty dataframes as fallback."""
        etad_data = pd.DataFrame(columns=['filter_id', 'SampleDate', 'Fabs', 'Site'])
        ftir_data = pd.DataFrame(columns=['filter_id', 'date', 'EC_FTIR', 'OC_FTIR'])
        return etad_data, ftir_data
    
    def load_filter_data_from_csv(self, hips_path: Optional[str] = None,
                                 ftir_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load filter sample data from CSV files.
        
        Parameters:
        -----------
        hips_path : str, optional
            Path to HIPS/ETAD CSV file
        ftir_path : str, optional
            Path to FTIR CSV file
            
        Returns:
        --------
        tuple
            (etad_data, ftir_data) DataFrames
        """
        logger.info("Loading filter sample data from CSV files...")
        
        etad_data = pd.DataFrame()
        ftir_data = pd.DataFrame()
        
        if hips_path and Path(hips_path).exists():
            etad_data = pd.read_csv(hips_path)
            etad_data['SampleDate'] = pd.to_datetime(etad_data['SampleDate'])
            logger.info(f"Loaded {len(etad_data)} HIPS samples from {hips_path}")
        
        if ftir_path and Path(ftir_path).exists():
            ftir_data = pd.read_csv(ftir_path)
            ftir_data['date'] = pd.to_datetime(ftir_data['date'])
            logger.info(f"Loaded {len(ftir_data)} FTIR samples from {ftir_path}")
        
        # Store for later use
        self.filter_data = {
            'etad': etad_data,
            'ftir': ftir_data,
            'hips_path': hips_path,
            'ftir_path': ftir_path
        }
        
        return etad_data, ftir_data
    
    def map_to_quality_periods(self, quality_series: pd.Series, 
                              etad_data: pd.DataFrame, 
                              ftir_data: pd.DataFrame) -> Dict:
        """
        Map filter sampling periods to aethalometer data quality periods.
        
        Parameters:
        -----------
        quality_series : pd.Series
            Series with datetime index and quality labels (from QualityClassifier)
        etad_data : pd.DataFrame
            ETAD/HIPS filter data with 'SampleDate' column
        ftir_data : pd.DataFrame
            FTIR filter data with 'date' column
            
        Returns:
        --------
        dict
            Comprehensive mapping results including overlaps and statistics
        """
        logger.info("Mapping filter samples to aethalometer quality periods...")
        
        # Get quality period classifications
        quality_periods = self._extract_quality_periods(quality_series)
        
        # Convert filter sample dates to analysis periods (9am start times)
        filter_periods = self._convert_filter_dates_to_periods(etad_data, ftir_data)
        
        # Find overlaps between filter periods and quality periods
        overlaps = self._calculate_overlaps(filter_periods, quality_periods)
        
        # Calculate quality distribution for filter periods
        filter_quality_analysis = self._analyze_filter_quality_distribution(
            filter_periods, quality_series
        )
        
        # Compile results
        results = {
            **quality_periods,
            **filter_periods,
            **overlaps,
            'filter_quality_table': filter_quality_analysis['quality_table'],
            'filter_quality_stats': filter_quality_analysis['stats'],
            'quality_series': quality_series
        }
        
        # Store results
        self.mapping_results = results
        
        # Log summary
        self._log_mapping_summary(results)
        
        return results
    
    def _extract_quality_periods(self, quality_series: pd.Series) -> Dict:
        """Extract different quality period classifications."""
        high_quality_mask = quality_series.isin(['Excellent', 'Good'])
        
        return {
            'high_quality_periods': quality_series[high_quality_mask].index,
            'excellent_periods': quality_series[quality_series == 'Excellent'].index,
            'good_periods': quality_series[quality_series == 'Good'].index,
            'moderate_periods': quality_series[quality_series == 'Moderate'].index,
            'poor_periods': quality_series[quality_series == 'Poor'].index
        }
    
    def _convert_filter_dates_to_periods(self, etad_data: pd.DataFrame, 
                                       ftir_data: pd.DataFrame) -> Dict:
        """Convert filter sample dates to 9am-to-9am analysis periods."""
        # Get HIPS/ETAD periods
        hips_dates = pd.DatetimeIndex([])
        if len(etad_data) > 0 and 'SampleDate' in etad_data.columns:
            valid_hips_dates = etad_data['SampleDate'].dropna()
            hips_dates = pd.DatetimeIndex([
                d.normalize() + pd.Timedelta(hours=9)
                for d in valid_hips_dates
            ])
        
        # Get FTIR periods
        ftir_dates = pd.DatetimeIndex([])
        if len(ftir_data) > 0 and 'date' in ftir_data.columns:
            valid_ftir_dates = ftir_data['date'].dropna()
            ftir_dates = pd.DatetimeIndex([
                d.normalize() + pd.Timedelta(hours=9)
                for d in valid_ftir_dates
            ])
        
        # Combine unique filter periods
        all_filter_periods = hips_dates.union(ftir_dates).unique()
        
        # Find overlap between HIPS and FTIR
        hips_ftir_overlap = hips_dates.intersection(ftir_dates).unique()
        
        return {
            'filter_periods': all_filter_periods,
            'hips_dates': hips_dates,
            'ftir_dates': ftir_dates,
            'hips_ftir_overlap': hips_ftir_overlap
        }
    
    def _calculate_overlaps(self, filter_periods: Dict, quality_periods: Dict) -> Dict:
        """Calculate overlaps between filter periods and quality periods."""
        all_filter_periods = filter_periods['filter_periods']
        
        overlaps = {}
        
        # Calculate overlaps for each quality level
        for quality_type, periods in quality_periods.items():
            if quality_type.endswith('_periods'):
                overlap_key = quality_type.replace('_periods', '_overlaps')
                overlaps[overlap_key] = pd.DatetimeIndex(all_filter_periods).intersection(periods)
        
        # Combined high-quality overlaps
        overlaps['overlap_periods'] = overlaps['excellent_overlaps'].union(overlaps['good_overlaps'])
        
        return overlaps
    
    def _analyze_filter_quality_distribution(self, filter_periods: Dict, 
                                           quality_series: pd.Series) -> Dict:
        """Analyze quality distribution for filter periods."""
        all_filter_periods = filter_periods['filter_periods']
        
        # Get quality for each filter period
        filter_qualities = []
        for period in all_filter_periods:
            if period in quality_series.index:
                filter_qualities.append(quality_series.loc[period])
            else:
                filter_qualities.append('No Data')
        
        quality_counts = pd.Series(filter_qualities).value_counts()
        quality_pcts = quality_counts / len(all_filter_periods) * 100
        
        quality_table = pd.DataFrame({
            'Count': quality_counts,
            'Percentage': quality_pcts.round(1)
        })
        
        # Calculate statistics
        total_periods = len(all_filter_periods)
        high_quality_count = quality_counts.get('Excellent', 0) + quality_counts.get('Good', 0)
        usable_percentage = (high_quality_count / total_periods * 100) if total_periods > 0 else 0
        
        stats = {
            'total_filter_periods': total_periods,
            'high_quality_periods': high_quality_count,
            'usable_percentage': usable_percentage,
            'excellent_count': quality_counts.get('Excellent', 0),
            'good_count': quality_counts.get('Good', 0),
            'moderate_count': quality_counts.get('Moderate', 0),
            'poor_count': quality_counts.get('Poor', 0),
            'no_data_count': quality_counts.get('No Data', 0)
        }
        
        return {
            'quality_table': quality_table,
            'stats': stats
        }
    
    def _log_mapping_summary(self, results: Dict) -> None:
        """Log summary of mapping results."""
        total_filter = len(results['filter_periods'])
        high_quality = len(results['overlap_periods'])
        excellent = len(results['excellent_overlaps'])
        good = len(results['good_overlaps'])
        
        logger.info(f"Filter sample mapping summary:")
        logger.info(f"  Total filter periods: {total_filter}")
        logger.info(f"  High-quality overlaps: {high_quality} ({high_quality/total_filter*100:.1f}%)" if total_filter > 0 else "  High-quality overlaps: 0")
        logger.info(f"    - Excellent quality: {excellent}")
        logger.info(f"    - Good quality: {good}")
        
        if 'hips_ftir_overlap' in results:
            both_filters = len(results['hips_ftir_overlap'])
            logger.info(f"  Periods with both filter types: {both_filters}")
    
    def export_quality_filtered_periods(self, include_all_quality: bool = True,
                                      output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Export filter periods with quality information.
        
        Parameters:
        -----------
        include_all_quality : bool, default True
            If True, include all quality levels. If False, only high-quality periods.
        output_path : str, optional
            Path to save CSV file
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with filter periods and quality information
        """
        if self.mapping_results is None:
            raise ValueError("Must run map_to_quality_periods() first")
        
        results = self.mapping_results
        
        if include_all_quality:
            periods_to_include = results['filter_periods']
        else:
            periods_to_include = results['overlap_periods']
        
        if len(periods_to_include) == 0:
            logger.warning("No periods found for export")
            return pd.DataFrame()
        
        # Create export DataFrame
        export_df = pd.DataFrame(index=periods_to_include)
        export_df['start_time'] = export_df.index
        export_df['end_time'] = export_df['start_time'] + pd.Timedelta(days=1)
        
        # Add filter type information
        export_df['in_hips'] = export_df.index.isin(results['hips_dates'])
        export_df['in_ftir'] = export_df.index.isin(results['ftir_dates'])
        export_df['has_both_filters'] = export_df.index.isin(results['hips_ftir_overlap'])
        
        # Add quality information
        quality_series = results['quality_series']
        export_df['aethalometer_quality'] = 'Missing'
        
        for date in export_df.index:
            if date in quality_series.index:
                export_df.loc[date, 'aethalometer_quality'] = quality_series.loc[date]
        
        # Add convenience flags
        export_df['missing_aethalometer'] = export_df['aethalometer_quality'] == 'Missing'
        export_df['usable_for_comparison'] = (
            export_df['aethalometer_quality'].isin(['Excellent', 'Good']) & 
            export_df['has_both_filters']
        )
        
        # Sort by date
        export_df = export_df.sort_values('start_time')
        
        # Save to file if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            export_df.to_csv(output_path, index=False)
            logger.info(f"Data exported to: {output_path}")
        
        return export_df
    
    def get_usable_periods(self, min_quality: str = 'Good') -> pd.DatetimeIndex:
        """
        Get periods usable for comparison studies.
        
        Parameters:
        -----------
        min_quality : str, default 'Good'
            Minimum quality level required
            
        Returns:
        --------
        pd.DatetimeIndex
            Usable periods with both filter types and adequate quality
        """
        if self.mapping_results is None:
            raise ValueError("Must run map_to_quality_periods() first")
        
        results = self.mapping_results
        
        # Get quality series and filter periods with both filter types
        quality_series = results['quality_series']
        both_filter_periods = results['hips_ftir_overlap']
        
        # Filter by quality
        if min_quality == 'Excellent':
            quality_mask = quality_series == 'Excellent'
        elif min_quality == 'Good':
            quality_mask = quality_series.isin(['Excellent', 'Good'])
        elif min_quality == 'Moderate':
            quality_mask = quality_series.isin(['Excellent', 'Good', 'Moderate'])
        else:
            quality_mask = quality_series.notna()  # Any quality
        
        usable_quality_periods = quality_series[quality_mask].index
        
        # Find intersection with both-filter periods
        usable_periods = pd.DatetimeIndex(both_filter_periods).intersection(usable_quality_periods)
        
        logger.info(f"Found {len(usable_periods)} usable periods (both filters + {min_quality}+ quality)")
        
        return usable_periods


def map_filter_samples(quality_series: pd.Series, 
                      etad_data: pd.DataFrame, 
                      ftir_data: pd.DataFrame) -> Dict:
    """
    Convenience function for mapping filter samples to quality periods.
    
    Parameters:
    -----------
    quality_series : pd.Series
        Quality classification series
    etad_data : pd.DataFrame
        ETAD/HIPS filter data
    ftir_data : pd.DataFrame
        FTIR filter data
        
    Returns:
    --------
    dict
        Mapping results
    """
    mapper = FilterSampleMapper()
    return mapper.map_to_quality_periods(quality_series, etad_data, ftir_data)


def load_filter_database(db_path: str, site_code: str = 'ETAD') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for loading filter data from database.
    
    Parameters:
    -----------
    db_path : str
        Path to database
    site_code : str, default 'ETAD'
        Site code
        
    Returns:
    --------
    tuple
        (etad_data, ftir_data)
    """
    mapper = FilterSampleMapper()
    return mapper.load_filter_data_from_db(db_path, site_code)


if __name__ == "__main__":
    # Example usage
    print("Filter Sample Mapping Module")
    print("Use FilterSampleMapper class or convenience functions")
