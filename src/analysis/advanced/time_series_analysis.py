"""Advanced time series analysis for ETAD data"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ...core.base import BaseAnalyzer
from ...core.monitoring import monitor_performance
from ...utils.memory_optimization import optimize_memory

@dataclass
class TrendResult:
    """Results from trend analysis"""
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1, strength of trend
    slope: float  # Linear trend slope
    p_value: float  # Statistical significance
    confidence_interval: Tuple[float, float]
    seasonal_component: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None

@dataclass 
class SeasonalResult:
    """Results from seasonal analysis"""
    seasonal_pattern: str  # 'daily', 'weekly', 'monthly', 'annual', 'none'
    seasonal_strength: float  # 0-1, strength of seasonality
    peak_periods: List[int]  # Indices of peak periods
    dominant_frequencies: List[float]  # Dominant frequency components
    seasonal_decomposition: Dict[str, np.ndarray]

class TimeSeriesAnalyzer(BaseAnalyzer):
    """Advanced time series analysis for ETAD data"""
    
    def __init__(self):
        super().__init__("TimeSeriesAnalyzer")
        
    @monitor_performance()
    @optimize_memory()
    def analyze_complete_timeseries(self, 
                                  data: pd.DataFrame, 
                                  value_columns: List[str],
                                  timestamp_column: str = 'timestamp',
                                  include_decomposition: bool = True,
                                  include_forecasting: bool = False) -> Dict[str, Any]:
        """
        Complete time series analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data
        value_columns : List[str]
            Columns to analyze
        timestamp_column : str
            Name of timestamp column
        include_decomposition : bool
            Include seasonal decomposition
        include_forecasting : bool
            Include forecasting analysis
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive analysis results
        """
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': self._get_data_summary(data, value_columns, timestamp_column),
            'trends': {},
            'seasonality': {},
            'anomalies': {},
            'correlations': {},
            'forecasts': {} if include_forecasting else None
        }
        
        # Ensure timestamp column is datetime
        if timestamp_column in data.columns:
            data = data.copy()
            data[timestamp_column] = pd.to_datetime(data[timestamp_column])
            data = data.sort_values(timestamp_column)
        
        # Analyze each column
        for column in value_columns:
            if column not in data.columns:
                self.logger.warning(f"Column {column} not found in data")
                continue
                
            column_data = data[column].dropna()
            if len(column_data) < 10:
                self.logger.warning(f"Insufficient data for analysis in column {column}")
                continue
            
            # Trend analysis
            trend_analyzer = TrendDetector()
            results['trends'][column] = trend_analyzer.detect_trend(
                column_data.values,
                timestamps=data[timestamp_column].values if timestamp_column in data.columns else None
            )
            
            # Seasonal analysis
            if include_decomposition and len(column_data) >= 24:  # Need sufficient data
                seasonal_analyzer = SeasonalAnalyzer()
                results['seasonality'][column] = seasonal_analyzer.analyze_seasonality(
                    column_data.values,
                    timestamps=data[timestamp_column].values if timestamp_column in data.columns else None
                )
            
            # Anomaly detection
            anomalies = self._detect_anomalies(column_data.values)
            results['anomalies'][column] = {
                'anomaly_indices': anomalies.tolist(),
                'anomaly_count': len(anomalies),
                'anomaly_percentage': len(anomalies) / len(column_data) * 100
            }
        
        # Cross-correlation analysis
        if len(value_columns) > 1:
            results['correlations'] = self._analyze_cross_correlations(data, value_columns)
        
        # Forecasting
        if include_forecasting:
            forecaster = ForecastAnalyzer()
            for column in value_columns:
                if column in data.columns:
                    forecast_result = forecaster.forecast_timeseries(
                        data[column].dropna().values,
                        forecast_periods=24
                    )
                    results['forecasts'][column] = forecast_result
        
        return results
    
    def _get_data_summary(self, data: pd.DataFrame, value_columns: List[str], timestamp_column: str) -> Dict[str, Any]:
        """Get summary statistics for the data"""
        summary = {
            'total_records': len(data),
            'time_range': {},
            'data_completeness': {},
            'basic_statistics': {}
        }
        
        # Time range
        if timestamp_column in data.columns:
            timestamps = pd.to_datetime(data[timestamp_column])
            summary['time_range'] = {
                'start': timestamps.min().isoformat(),
                'end': timestamps.max().isoformat(),
                'duration_days': (timestamps.max() - timestamps.min()).days,
                'frequency_estimate': self._estimate_frequency(timestamps)
            }
        
        # Data completeness and basic stats
        for column in value_columns:
            if column in data.columns:
                col_data = data[column]
                summary['data_completeness'][column] = {
                    'valid_count': col_data.notna().sum(),
                    'missing_count': col_data.isna().sum(),
                    'completeness_percentage': col_data.notna().sum() / len(col_data) * 100
                }
                
                valid_data = col_data.dropna()
                if len(valid_data) > 0:
                    summary['basic_statistics'][column] = {
                        'mean': float(valid_data.mean()),
                        'median': float(valid_data.median()),
                        'std': float(valid_data.std()),
                        'min': float(valid_data.min()),
                        'max': float(valid_data.max()),
                        'skewness': float(stats.skew(valid_data)),
                        'kurtosis': float(stats.kurtosis(valid_data))
                    }
        
        return summary
    
    def _estimate_frequency(self, timestamps: pd.Series) -> str:
        """Estimate the frequency of timestamps"""
        if len(timestamps) < 2:
            return "unknown"
        
        # Calculate time differences
        time_diffs = timestamps.diff().dropna()
        median_diff = time_diffs.median()
        
        # Convert to common frequencies
        if median_diff <= pd.Timedelta(seconds=30):
            return "high-frequency (<=30s)"
        elif median_diff <= pd.Timedelta(minutes=1):
            return "1-minute"
        elif median_diff <= pd.Timedelta(minutes=5):
            return "5-minute"
        elif median_diff <= pd.Timedelta(minutes=15):
            return "15-minute"
        elif median_diff <= pd.Timedelta(hours=1):
            return "hourly"
        elif median_diff <= pd.Timedelta(days=1):
            return "daily"
        else:
            return "low-frequency (>1 day)"
    
    def _detect_anomalies(self, values: np.ndarray, method: str = 'iqr') -> np.ndarray:
        """Detect anomalies in time series data"""
        if method == 'iqr':
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies = np.where((values < lower_bound) | (values > upper_bound))[0]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(values))
            anomalies = np.where(z_scores > 3)[0]
            
        else:  # modified z-score
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad
            anomalies = np.where(np.abs(modified_z_scores) > 3.5)[0]
        
        return anomalies
    
    def _analyze_cross_correlations(self, data: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Analyze cross-correlations between variables"""
        correlations = {}
        
        # Pearson correlations
        correlation_matrix = data[columns].corr()
        correlations['pearson_matrix'] = correlation_matrix.to_dict()
        
        # Find strongest correlations
        strong_correlations = []
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                corr_value = correlation_matrix.loc[col1, col2]
                if abs(corr_value) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                    })
        
        correlations['strong_correlations'] = strong_correlations
        
        # Lag correlations (if enough data)
        if len(data) >= 50:
            lag_correlations = {}
            for col1 in columns:
                for col2 in columns:
                    if col1 != col2:
                        max_lag = min(24, len(data) // 4)  # Up to 24 periods or 1/4 of data
                        lags, corrs = self._calculate_lag_correlation(
                            data[col1].values, data[col2].values, max_lag
                        )
                        
                        # Find best lag
                        best_lag_idx = np.argmax(np.abs(corrs))
                        lag_correlations[f"{col1}_vs_{col2}"] = {
                            'best_lag': int(lags[best_lag_idx]),
                            'best_correlation': float(corrs[best_lag_idx]),
                            'all_lags': lags.tolist(),
                            'all_correlations': corrs.tolist()
                        }
            
            correlations['lag_correlations'] = lag_correlations
        
        return correlations
    
    def _calculate_lag_correlation(self, x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate lag correlation between two series"""
        lags = np.arange(-max_lag, max_lag + 1)
        correlations = np.zeros(len(lags))
        
        for i, lag in enumerate(lags):
            if lag == 0:
                correlations[i] = np.corrcoef(x, y)[0, 1]
            elif lag > 0:
                # y leads x
                if len(x) > lag:
                    correlations[i] = np.corrcoef(x[:-lag], y[lag:])[0, 1]
            else:
                # x leads y
                lag = abs(lag)
                if len(y) > lag:
                    correlations[i] = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        
        return lags, correlations

class TrendDetector(BaseAnalyzer):
    """Detect and analyze trends in time series data"""
    
    def __init__(self):
        super().__init__("TrendDetector")
    
    @monitor_performance()
    def detect_trend(self, 
                    values: np.ndarray, 
                    timestamps: Optional[np.ndarray] = None,
                    method: str = 'linear') -> TrendResult:
        """
        Detect trend in time series
        
        Parameters:
        -----------
        values : np.ndarray
            Time series values
        timestamps : np.ndarray, optional
            Timestamps for the values
        method : str
            Trend detection method ('linear', 'polynomial', 'seasonal')
            
        Returns:
        --------
        TrendResult
            Trend analysis results
        """
        if len(values) < 3:
            return TrendResult(
                trend_direction='unknown',
                trend_strength=0.0,
                slope=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0)
            )
        
        # Create time index if not provided
        if timestamps is None:
            x = np.arange(len(values))
        else:
            x = np.arange(len(timestamps))
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        x_clean = x[valid_mask]
        y_clean = values[valid_mask]
        
        if len(y_clean) < 3:
            return TrendResult(
                trend_direction='unknown',
                trend_strength=0.0,
                slope=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0)
            )
        
        if method == 'linear':
            return self._linear_trend(x_clean, y_clean)
        elif method == 'polynomial':
            return self._polynomial_trend(x_clean, y_clean)
        else:  # seasonal
            return self._seasonal_trend(x_clean, y_clean)
    
    def _linear_trend(self, x: np.ndarray, y: np.ndarray) -> TrendResult:
        """Detect linear trend using linear regression"""
        try:
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine trend direction
            if p_value < 0.05:  # Statistically significant
                if slope > 0:
                    direction = 'increasing'
                elif slope < 0:
                    direction = 'decreasing'
                else:
                    direction = 'stable'
            else:
                direction = 'stable'
            
            # Trend strength (R-squared)
            trend_strength = r_value ** 2
            
            # Confidence interval for slope
            t_val = stats.t.ppf(0.975, len(x) - 2)  # 95% confidence
            margin_error = t_val * std_err
            confidence_interval = (slope - margin_error, slope + margin_error)
            
            return TrendResult(
                trend_direction=direction,
                trend_strength=trend_strength,
                slope=slope,
                p_value=p_value,
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            self.logger.error(f"Error in linear trend detection: {e}")
            return TrendResult(
                trend_direction='error',
                trend_strength=0.0,
                slope=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0)
            )
    
    def _polynomial_trend(self, x: np.ndarray, y: np.ndarray, degree: int = 2) -> TrendResult:
        """Detect polynomial trend"""
        try:
            # Fit polynomial
            coefficients = np.polyfit(x, y, degree)
            poly_func = np.poly1d(coefficients)
            y_pred = poly_func(x)
            
            # Calculate R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Determine trend direction from first derivative
            derivative = np.polyder(poly_func)
            trend_at_end = derivative(x[-1])
            
            if trend_at_end > 0:
                direction = 'increasing'
            elif trend_at_end < 0:
                direction = 'decreasing'
            else:
                direction = 'stable'
            
            return TrendResult(
                trend_direction=direction,
                trend_strength=r_squared,
                slope=float(trend_at_end),
                p_value=0.0,  # P-value calculation for polynomial is complex
                confidence_interval=(0.0, 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in polynomial trend detection: {e}")
            return self._linear_trend(x, y)  # Fallback to linear
    
    def _seasonal_trend(self, x: np.ndarray, y: np.ndarray) -> TrendResult:
        """Detect trend with seasonal decomposition"""
        try:
            # Simple seasonal decomposition using moving averages
            # This is a simplified version - for production use, consider statsmodels
            window_size = min(max(7, len(y) // 10), len(y) // 3)
            
            if len(y) < window_size:
                return self._linear_trend(x, y)
            
            # Calculate trend component using centered moving average
            trend_component = np.full(len(y), np.nan)
            half_window = window_size // 2
            
            for i in range(half_window, len(y) - half_window):
                trend_component[i] = np.mean(y[i - half_window:i + half_window + 1])
            
            # Remove NaN values from trend
            valid_trend = ~np.isnan(trend_component)
            if valid_trend.sum() < 3:
                return self._linear_trend(x, y)
            
            # Calculate trend on the trend component
            trend_result = self._linear_trend(x[valid_trend], trend_component[valid_trend])
            
            # Calculate residuals for seasonal analysis
            residuals = y - trend_component
            
            trend_result.seasonal_component = trend_component
            trend_result.residuals = residuals
            
            return trend_result
            
        except Exception as e:
            self.logger.error(f"Error in seasonal trend detection: {e}")
            return self._linear_trend(x, y)

class SeasonalAnalyzer(BaseAnalyzer):
    """Analyze seasonal patterns in time series data"""
    
    def __init__(self):
        super().__init__("SeasonalAnalyzer")
    
    @monitor_performance()
    def analyze_seasonality(self, 
                          values: np.ndarray, 
                          timestamps: Optional[np.ndarray] = None,
                          detect_frequency: bool = True) -> SeasonalResult:
        """
        Analyze seasonal patterns
        
        Parameters:
        -----------
        values : np.ndarray
            Time series values
        timestamps : np.ndarray, optional
            Timestamps for the values
        detect_frequency : bool
            Whether to detect dominant frequencies
            
        Returns:
        --------
        SeasonalResult
            Seasonal analysis results
        """
        if len(values) < 12:  # Need minimum data for seasonal analysis
            return SeasonalResult(
                seasonal_pattern='none',
                seasonal_strength=0.0,
                peak_periods=[],
                dominant_frequencies=[],
                seasonal_decomposition={}
            )
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        clean_values = values[valid_mask]
        
        if len(clean_values) < 12:
            return SeasonalResult(
                seasonal_pattern='none',
                seasonal_strength=0.0,
                peak_periods=[],
                dominant_frequencies=[],
                seasonal_decomposition={}
            )
        
        # Frequency analysis using FFT
        fft_result = self._frequency_analysis(clean_values)
        
        # Detect seasonal patterns
        seasonal_pattern = self._detect_seasonal_pattern(clean_values, timestamps)
        
        # Calculate seasonal strength
        seasonal_strength = self._calculate_seasonal_strength(clean_values)
        
        # Find peak periods
        peak_periods = self._find_peak_periods(clean_values)
        
        # Seasonal decomposition
        decomposition = self._simple_seasonal_decomposition(clean_values)
        
        return SeasonalResult(
            seasonal_pattern=seasonal_pattern,
            seasonal_strength=seasonal_strength,
            peak_periods=peak_periods.tolist(),
            dominant_frequencies=fft_result['dominant_frequencies'],
            seasonal_decomposition=decomposition
        )
    
    def _frequency_analysis(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze frequency components using FFT"""
        try:
            # Apply FFT
            fft = np.fft.fft(values)
            frequencies = np.fft.fftfreq(len(values))
            
            # Get power spectrum
            power = np.abs(fft) ** 2
            
            # Find dominant frequencies (excluding DC component)
            valid_indices = frequencies > 0
            valid_freqs = frequencies[valid_indices]
            valid_power = power[valid_indices]
            
            # Sort by power and get top frequencies
            top_indices = np.argsort(valid_power)[-5:]  # Top 5 frequencies
            dominant_frequencies = valid_freqs[top_indices].tolist()
            
            return {
                'dominant_frequencies': dominant_frequencies,
                'power_spectrum': valid_power.tolist(),
                'frequencies': valid_freqs.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error in frequency analysis: {e}")
            return {
                'dominant_frequencies': [],
                'power_spectrum': [],
                'frequencies': []
            }
    
    def _detect_seasonal_pattern(self, values: np.ndarray, timestamps: Optional[np.ndarray]) -> str:
        """Detect the type of seasonal pattern"""
        
        # Try different seasonal periods
        periods_to_test = [7, 24, 30, 365]  # Weekly, daily, monthly, annual patterns
        
        if timestamps is not None:
            # Adjust periods based on data frequency
            # This is simplified - in practice, you'd analyze timestamp intervals
            pass
        
        best_period = None
        best_autocorr = 0
        
        for period in periods_to_test:
            if period < len(values):
                # Calculate autocorrelation at this lag
                autocorr = self._calculate_autocorrelation(values, period)
                if autocorr > best_autocorr:
                    best_autocorr = autocorr
                    best_period = period
        
        # Classify seasonal pattern
        if best_autocorr > 0.3:  # Threshold for meaningful seasonality
            if best_period == 7:
                return 'weekly'
            elif best_period == 24:
                return 'daily'
            elif best_period == 30:
                return 'monthly'
            elif best_period == 365:
                return 'annual'
            else:
                return 'custom'
        
        return 'none'
    
    def _calculate_autocorrelation(self, values: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at specific lag"""
        if lag >= len(values):
            return 0.0
        
        try:
            # Normalize values
            normalized = (values - np.mean(values)) / np.std(values)
            
            # Calculate autocorrelation
            autocorr = np.corrcoef(normalized[:-lag], normalized[lag:])[0, 1]
            
            return autocorr if not np.isnan(autocorr) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_seasonal_strength(self, values: np.ndarray) -> float:
        """Calculate the strength of seasonal pattern"""
        try:
            # Use coefficient of variation as a proxy for seasonal strength
            # This is simplified - more sophisticated methods exist
            
            # Detrend the data first
            detrended = savgol_filter(values, min(len(values)//4, 51), 3)
            residuals = values - detrended
            
            # Calculate seasonal strength as ratio of seasonal variance to total variance
            seasonal_var = np.var(residuals)
            total_var = np.var(values)
            
            seasonal_strength = seasonal_var / total_var if total_var > 0 else 0.0
            
            return min(seasonal_strength, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating seasonal strength: {e}")
            return 0.0
    
    def _find_peak_periods(self, values: np.ndarray) -> np.ndarray:
        """Find periods with peak values"""
        try:
            # Find peaks using scipy
            peaks, _ = find_peaks(values, height=np.percentile(values, 75))
            return peaks
            
        except Exception as e:
            self.logger.error(f"Error finding peak periods: {e}")
            return np.array([])
    
    def _simple_seasonal_decomposition(self, values: np.ndarray) -> Dict[str, np.ndarray]:
        """Simple seasonal decomposition"""
        try:
            # This is a simplified decomposition
            # For production use, consider statsmodels.seasonal_decompose
            
            # Trend component (moving average)
            window_size = min(max(7, len(values) // 10), len(values) // 3)
            trend = np.convolve(values, np.ones(window_size)/window_size, mode='same')
            
            # Seasonal component (residuals from trend)
            seasonal = values - trend
            
            # Residual component (noise)
            seasonal_smooth = savgol_filter(seasonal, min(len(seasonal)//4, 15), 2)
            residual = seasonal - seasonal_smooth
            
            return {
                'trend': trend,
                'seasonal': seasonal_smooth,
                'residual': residual,
                'original': values
            }
            
        except Exception as e:
            self.logger.error(f"Error in seasonal decomposition: {e}")
            return {
                'trend': values,
                'seasonal': np.zeros_like(values),
                'residual': np.zeros_like(values),
                'original': values
            }

class ForecastAnalyzer(BaseAnalyzer):
    """Simple forecasting for time series data"""
    
    def __init__(self):
        super().__init__("ForecastAnalyzer")
    
    @monitor_performance()
    def forecast_timeseries(self, 
                          values: np.ndarray, 
                          forecast_periods: int = 24,
                          method: str = 'linear') -> Dict[str, Any]:
        """
        Simple time series forecasting
        
        Parameters:
        -----------
        values : np.ndarray
            Historical values
        forecast_periods : int
            Number of periods to forecast
        method : str
            Forecasting method ('linear', 'seasonal', 'exponential')
            
        Returns:
        --------
        Dict[str, Any]
            Forecast results
        """
        if len(values) < 5:
            return {
                'method': method,
                'forecast': np.full(forecast_periods, np.nan).tolist(),
                'confidence_lower': np.full(forecast_periods, np.nan).tolist(),
                'confidence_upper': np.full(forecast_periods, np.nan).tolist(),
                'error': 'Insufficient data for forecasting'
            }
        
        # Remove NaN values
        clean_values = values[~np.isnan(values)]
        
        if method == 'linear':
            return self._linear_forecast(clean_values, forecast_periods)
        elif method == 'exponential':
            return self._exponential_forecast(clean_values, forecast_periods)
        else:  # seasonal
            return self._seasonal_forecast(clean_values, forecast_periods)
    
    def _linear_forecast(self, values: np.ndarray, periods: int) -> Dict[str, Any]:
        """Linear trend forecasting"""
        try:
            x = np.arange(len(values))
            
            # Fit linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Generate forecast
            future_x = np.arange(len(values), len(values) + periods)
            forecast = slope * future_x + intercept
            
            # Calculate confidence intervals (simplified)
            residuals = values - (slope * x + intercept)
            residual_std = np.std(residuals)
            
            confidence_lower = forecast - 1.96 * residual_std
            confidence_upper = forecast + 1.96 * residual_std
            
            return {
                'method': 'linear',
                'forecast': forecast.tolist(),
                'confidence_lower': confidence_lower.tolist(),
                'confidence_upper': confidence_upper.tolist(),
                'model_params': {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in linear forecasting: {e}")
            return {
                'method': 'linear',
                'forecast': np.full(periods, values[-1]).tolist(),
                'confidence_lower': np.full(periods, values[-1] * 0.8).tolist(),
                'confidence_upper': np.full(periods, values[-1] * 1.2).tolist(),
                'error': str(e)
            }
    
    def _exponential_forecast(self, values: np.ndarray, periods: int) -> Dict[str, Any]:
        """Exponential smoothing forecast"""
        try:
            # Simple exponential smoothing
            alpha = 0.3  # Smoothing parameter
            
            # Calculate smoothed values
            smoothed = np.zeros_like(values)
            smoothed[0] = values[0]
            
            for i in range(1, len(values)):
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
            
            # Forecast (constant level)
            last_smoothed = smoothed[-1]
            forecast = np.full(periods, last_smoothed)
            
            # Estimate prediction intervals
            residuals = values - smoothed
            residual_std = np.std(residuals)
            
            confidence_lower = forecast - 1.96 * residual_std
            confidence_upper = forecast + 1.96 * residual_std
            
            return {
                'method': 'exponential',
                'forecast': forecast.tolist(),
                'confidence_lower': confidence_lower.tolist(),
                'confidence_upper': confidence_upper.tolist(),
                'model_params': {
                    'alpha': alpha,
                    'final_level': last_smoothed
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in exponential forecasting: {e}")
            return self._linear_forecast(values, periods)  # Fallback
    
    def _seasonal_forecast(self, values: np.ndarray, periods: int) -> Dict[str, Any]:
        """Seasonal naive forecast"""
        try:
            # Detect seasonal period (simplified)
            seasonal_period = min(24, len(values) // 3)  # Assume daily seasonality or use 1/3 of data
            
            if seasonal_period < 2:
                return self._linear_forecast(values, periods)
            
            # Use seasonal naive method: forecast = value from same season last year
            forecast = []
            for i in range(periods):
                seasonal_index = (len(values) + i) % seasonal_period
                if seasonal_index < len(values):
                    # Find the most recent value at this seasonal index
                    seasonal_values = values[seasonal_index::seasonal_period]
                    if len(seasonal_values) > 0:
                        forecast.append(seasonal_values[-1])
                    else:
                        forecast.append(values[-1])
                else:
                    forecast.append(values[-1])
            
            forecast = np.array(forecast)
            
            # Estimate confidence intervals
            residual_std = np.std(values) * 0.1  # Simplified
            confidence_lower = forecast - 1.96 * residual_std
            confidence_upper = forecast + 1.96 * residual_std
            
            return {
                'method': 'seasonal',
                'forecast': forecast.tolist(),
                'confidence_lower': confidence_lower.tolist(),
                'confidence_upper': confidence_upper.tolist(),
                'model_params': {
                    'seasonal_period': seasonal_period
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in seasonal forecasting: {e}")
            return self._linear_forecast(values, periods)  # Fallback
