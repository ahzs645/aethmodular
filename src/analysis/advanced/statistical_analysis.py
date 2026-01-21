"""Statistical analysis module for aethalometer data"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ks_2samp, mannwhitneyu
from sklearn.preprocessing import StandardScaler
import warnings
from ...core.base import BaseAnalyzer
from ...core.monitoring import monitor_performance, handle_errors
from ...utils.logging.logger import get_logger


class StatisticalComparator(BaseAnalyzer):
    """
    Compare statistical properties between different datasets or time periods
    """
    
    def __init__(self):
        super().__init__("StatisticalComparator")
        self.logger = get_logger(__name__)
        
    @monitor_performance
    @handle_errors
    def compare_periods(self, data1: pd.DataFrame, data2: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       comparison_tests: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare statistical properties between two datasets
        
        Parameters:
        -----------
        data1, data2 : pd.DataFrame
            Datasets to compare
        columns : List[str], optional
            Columns to analyze. If None, uses all numeric columns
        comparison_tests : List[str], optional
            Statistical tests to perform. Options: ['ttest', 'kstest', 'mannwhitney', 'correlation']
            
        Returns:
        --------
        Dict[str, Any]
            Comparison results
        """
        if columns is None:
            columns = [col for col in data1.columns if pd.api.types.is_numeric_dtype(data1[col])]
            
        if comparison_tests is None:
            comparison_tests = ['ttest', 'kstest', 'mannwhitney']
            
        results = {
            'datasets_info': {
                'dataset1': {
                    'shape': data1.shape,
                    'period': f"{data1.index.min()} to {data1.index.max()}"
                },
                'dataset2': {
                    'shape': data2.shape,
                    'period': f"{data2.index.min()} to {data2.index.max()}"
                }
            },
            'columns_analyzed': columns,
            'statistical_tests': {}
        }
        
        for col in columns:
            if col not in data1.columns or col not in data2.columns:
                self.logger.warning(f"Column {col} not found in both datasets, skipping")
                continue
                
            # Remove NaN values for comparison
            series1 = data1[col].dropna()
            series2 = data2[col].dropna()
            
            if len(series1) == 0 or len(series2) == 0:
                self.logger.warning(f"No valid data for column {col}, skipping")
                continue
                
            col_results = {
                'descriptive_stats': {
                    'dataset1': self._calculate_descriptive_stats(series1),
                    'dataset2': self._calculate_descriptive_stats(series2)
                }
            }
            
            # Perform statistical tests
            if 'ttest' in comparison_tests:
                col_results['t_test'] = self._perform_ttest(series1, series2)
                
            if 'kstest' in comparison_tests:
                col_results['ks_test'] = self._perform_kstest(series1, series2)
                
            if 'mannwhitney' in comparison_tests:
                col_results['mannwhitney_test'] = self._perform_mannwhitney(series1, series2)
                
            if 'correlation' in comparison_tests:
                col_results['correlation'] = self._calculate_correlation(series1, series2)
                
            results['statistical_tests'][col] = col_results
            
        return results
    
    def _calculate_descriptive_stats(self, series: pd.Series) -> Dict[str, float]:
        """Calculate descriptive statistics for a series"""
        return {
            'count': len(series),
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'q25': float(series.quantile(0.25)),
            'median': float(series.median()),
            'q75': float(series.quantile(0.75)),
            'max': float(series.max()),
            'skewness': float(stats.skew(series)),
            'kurtosis': float(stats.kurtosis(series))
        }
    
    def _perform_ttest(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """Perform independent t-test"""
        try:
            statistic, p_value = stats.ttest_ind(series1, series2)
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': 'Means are significantly different' if p_value < 0.05 else 'No significant difference in means'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _perform_kstest(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test"""
        try:
            statistic, p_value = ks_2samp(series1, series2)
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': 'Distributions are significantly different' if p_value < 0.05 else 'No significant difference in distributions'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _perform_mannwhitney(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """Perform Mann-Whitney U test"""
        try:
            statistic, p_value = mannwhitneyu(series1, series2, alternative='two-sided')
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': 'Medians are significantly different' if p_value < 0.05 else 'No significant difference in medians'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_correlation(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """Calculate correlation between two series"""
        try:
            # Align series by index
            aligned_data = pd.DataFrame({'series1': series1, 'series2': series2}).dropna()
            
            if len(aligned_data) < 3:
                return {'error': 'Insufficient data for correlation analysis'}
            
            pearson_corr, pearson_p = pearsonr(aligned_data['series1'], aligned_data['series2'])
            spearman_corr, spearman_p = spearmanr(aligned_data['series1'], aligned_data['series2'])
            
            return {
                'pearson': {
                    'correlation': float(pearson_corr),
                    'p_value': float(pearson_p),
                    'significant': pearson_p < 0.05
                },
                'spearman': {
                    'correlation': float(spearman_corr),
                    'p_value': float(spearman_p),
                    'significant': spearman_p < 0.05
                },
                'aligned_data_points': len(aligned_data)
            }
        except Exception as e:
            return {'error': str(e)}


class DistributionAnalyzer(BaseAnalyzer):
    """
    Analyze data distributions and perform distribution fitting
    """
    
    def __init__(self):
        super().__init__("DistributionAnalyzer")
        self.logger = get_logger(__name__)
        
    @monitor_performance
    @handle_errors
    def analyze_distribution(self, data: pd.Series, 
                           distributions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze data distribution and fit common distributions
        
        Parameters:
        -----------
        data : pd.Series
            Data to analyze
        distributions : List[str], optional
            Distributions to test. Options: ['normal', 'lognormal', 'exponential', 'gamma']
            
        Returns:
        --------
        Dict[str, Any]
            Distribution analysis results
        """
        if distributions is None:
            distributions = ['normal', 'lognormal', 'exponential']
            
        # Clean data
        clean_data = data.dropna()
        if len(clean_data) < 10:
            raise ValueError("Insufficient data for distribution analysis (need at least 10 points)")
        
        results = {
            'data_summary': self._get_data_summary(clean_data),
            'normality_tests': self._test_normality(clean_data),
            'distribution_fits': {}
        }
        
        # Fit distributions
        for dist_name in distributions:
            try:
                fit_result = self._fit_distribution(clean_data, dist_name)
                results['distribution_fits'][dist_name] = fit_result
            except Exception as e:
                self.logger.warning(f"Failed to fit {dist_name} distribution: {e}")
                results['distribution_fits'][dist_name] = {'error': str(e)}
        
        # Find best fit
        best_fit = self._find_best_fit(results['distribution_fits'])
        results['best_fit'] = best_fit
        
        return results
    
    def _get_data_summary(self, data: pd.Series) -> Dict[str, Any]:
        """Get summary statistics for the data"""
        return {
            'count': len(data),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data)),
            'zeros_count': int((data == 0).sum()),
            'negative_count': int((data < 0).sum())
        }
    
    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Test for normality using multiple tests"""
        results = {}
        
        # Shapiro-Wilk test (good for small samples)
        if len(data) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                results['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            except Exception as e:
                results['shapiro_wilk'] = {'error': str(e)}
        
        # D'Agostino and Pearson's test
        try:
            dagostino_stat, dagostino_p = stats.normaltest(data)
            results['dagostino_pearson'] = {
                'statistic': float(dagostino_stat),
                'p_value': float(dagostino_p),
                'is_normal': dagostino_p > 0.05
            }
        except Exception as e:
            results['dagostino_pearson'] = {'error': str(e)}
        
        # Anderson-Darling test
        try:
            anderson_result = stats.anderson(data, dist='norm')
            # Use 5% significance level (index 2)
            critical_value = anderson_result.critical_values[2]
            is_normal = anderson_result.statistic < critical_value
            
            results['anderson_darling'] = {
                'statistic': float(anderson_result.statistic),
                'critical_value_5pct': float(critical_value),
                'is_normal': is_normal
            }
        except Exception as e:
            results['anderson_darling'] = {'error': str(e)}
        
        return results
    
    def _fit_distribution(self, data: pd.Series, dist_name: str) -> Dict[str, Any]:
        """Fit a specific distribution to the data"""
        if dist_name == 'normal':
            return self._fit_normal(data)
        elif dist_name == 'lognormal':
            return self._fit_lognormal(data)
        elif dist_name == 'exponential':
            return self._fit_exponential(data)
        elif dist_name == 'gamma':
            return self._fit_gamma(data)
        else:
            raise ValueError(f"Unknown distribution: {dist_name}")
    
    def _fit_normal(self, data: pd.Series) -> Dict[str, Any]:
        """Fit normal distribution"""
        mean, std = stats.norm.fit(data)
        ks_stat, ks_p = stats.kstest(data, lambda x: stats.norm.cdf(x, mean, std))
        
        return {
            'parameters': {'mean': float(mean), 'std': float(std)},
            'goodness_of_fit': {
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'aic': self._calculate_aic(data, stats.norm.logpdf, [mean, std])
            }
        }
    
    def _fit_lognormal(self, data: pd.Series) -> Dict[str, Any]:
        """Fit lognormal distribution"""
        # Only fit if all data is positive
        if (data <= 0).any():
            return {'error': 'Lognormal distribution requires all positive values'}
        
        shape, loc, scale = stats.lognorm.fit(data, floc=0)
        ks_stat, ks_p = stats.kstest(data, lambda x: stats.lognorm.cdf(x, shape, loc, scale))
        
        return {
            'parameters': {'shape': float(shape), 'loc': float(loc), 'scale': float(scale)},
            'goodness_of_fit': {
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'aic': self._calculate_aic(data, stats.lognorm.logpdf, [shape, loc, scale])
            }
        }
    
    def _fit_exponential(self, data: pd.Series) -> Dict[str, Any]:
        """Fit exponential distribution"""
        # Only fit if all data is non-negative
        if (data < 0).any():
            return {'error': 'Exponential distribution requires all non-negative values'}
        
        loc, scale = stats.expon.fit(data)
        ks_stat, ks_p = stats.kstest(data, lambda x: stats.expon.cdf(x, loc, scale))
        
        return {
            'parameters': {'loc': float(loc), 'scale': float(scale)},
            'goodness_of_fit': {
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'aic': self._calculate_aic(data, stats.expon.logpdf, [loc, scale])
            }
        }
    
    def _fit_gamma(self, data: pd.Series) -> Dict[str, Any]:
        """Fit gamma distribution"""
        # Only fit if all data is positive
        if (data <= 0).any():
            return {'error': 'Gamma distribution requires all positive values'}
        
        a, loc, scale = stats.gamma.fit(data, floc=0)
        ks_stat, ks_p = stats.kstest(data, lambda x: stats.gamma.cdf(x, a, loc, scale))
        
        return {
            'parameters': {'a': float(a), 'loc': float(loc), 'scale': float(scale)},
            'goodness_of_fit': {
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'aic': self._calculate_aic(data, stats.gamma.logpdf, [a, loc, scale])
            }
        }
    
    def _calculate_aic(self, data: pd.Series, logpdf_func, params: List[float]) -> float:
        """Calculate Akaike Information Criterion"""
        try:
            log_likelihood = np.sum(logpdf_func(data, *params))
            k = len(params)  # number of parameters
            n = len(data)    # number of observations
            aic = 2 * k - 2 * log_likelihood
            return float(aic)
        except:
            return float('inf')
    
    def _find_best_fit(self, distribution_fits: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best fitting distribution based on AIC"""
        valid_fits = {}
        
        for dist_name, fit_result in distribution_fits.items():
            if 'error' not in fit_result and 'goodness_of_fit' in fit_result:
                aic = fit_result['goodness_of_fit'].get('aic', float('inf'))
                if not np.isinf(aic):
                    valid_fits[dist_name] = aic
        
        if not valid_fits:
            return {'best_distribution': None, 'reason': 'No valid fits found'}
        
        best_dist = min(valid_fits, key=valid_fits.get)
        return {
            'best_distribution': best_dist,
            'aic_value': valid_fits[best_dist],
            'aic_comparison': valid_fits
        }


class OutlierDetector(BaseAnalyzer):
    """
    Detect outliers using multiple methods
    """
    
    def __init__(self):
        super().__init__("OutlierDetector")
        self.logger = get_logger(__name__)
        
    @monitor_performance
    @handle_errors
    def detect_outliers(self, data: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect outliers using multiple methods
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to analyze
        columns : List[str], optional
            Columns to analyze for outliers
        methods : List[str], optional
            Methods to use: ['iqr', 'zscore', 'modified_zscore', 'isolation_forest']
            
        Returns:
        --------
        Dict[str, Any]
            Outlier detection results
        """
        if columns is None:
            columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            
        if methods is None:
            methods = ['iqr', 'zscore', 'modified_zscore']
            
        results = {
            'columns_analyzed': columns,
            'methods_used': methods,
            'outlier_detection': {}
        }
        
        for col in columns:
            if col not in data.columns:
                continue
                
            series = data[col].dropna()
            if len(series) < 10:
                self.logger.warning(f"Insufficient data for outlier detection in column {col}")
                continue
            
            col_results = {}
            
            if 'iqr' in methods:
                col_results['iqr'] = self._detect_iqr_outliers(series)
                
            if 'zscore' in methods:
                col_results['zscore'] = self._detect_zscore_outliers(series)
                
            if 'modified_zscore' in methods:
                col_results['modified_zscore'] = self._detect_modified_zscore_outliers(series)
                
            if 'isolation_forest' in methods:
                try:
                    from sklearn.ensemble import IsolationForest
                    col_results['isolation_forest'] = self._detect_isolation_forest_outliers(series)
                except ImportError:
                    self.logger.warning("sklearn not available for isolation forest outlier detection")
            
            # Combine results
            col_results['summary'] = self._summarize_outlier_detection(col_results, series)
            results['outlier_detection'][col] = col_results
            
        return results
    
    def _detect_iqr_outliers(self, series: pd.Series, multiplier: float = 1.5) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'method': 'IQR',
            'parameters': {'multiplier': multiplier},
            'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(series)) * 100,
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.tolist()
        }
    
    def _detect_zscore_outliers(self, series: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(series))
        outliers = series[z_scores > threshold]
        
        return {
            'method': 'Z-Score',
            'parameters': {'threshold': threshold},
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(series)) * 100,
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.tolist(),
            'z_scores': z_scores[z_scores > threshold].tolist()
        }
    
    def _detect_modified_zscore_outliers(self, series: pd.Series, threshold: float = 3.5) -> Dict[str, Any]:
        """Detect outliers using modified Z-score method (using median)"""
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            return {
                'method': 'Modified Z-Score',
                'error': 'MAD is zero, cannot compute modified Z-scores'
            }
        
        modified_z_scores = 0.6745 * (series - median) / mad
        outliers = series[np.abs(modified_z_scores) > threshold]
        
        return {
            'method': 'Modified Z-Score',
            'parameters': {'threshold': threshold},
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(series)) * 100,
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.tolist(),
            'modified_z_scores': modified_z_scores[np.abs(modified_z_scores) > threshold].tolist()
        }
    
    def _detect_isolation_forest_outliers(self, series: pd.Series, 
                                         contamination: float = 0.1) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest"""
        from sklearn.ensemble import IsolationForest
        
        # Reshape for sklearn
        X = series.values.reshape(-1, 1)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        
        # Get outliers (labeled as -1)
        outlier_mask = outlier_labels == -1
        outliers = series[outlier_mask]
        
        return {
            'method': 'Isolation Forest',
            'parameters': {'contamination': contamination},
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(series)) * 100,
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.tolist()
        }
    
    def _summarize_outlier_detection(self, method_results: Dict[str, Any], 
                                   series: pd.Series) -> Dict[str, Any]:
        """Summarize outlier detection across methods"""
        all_outlier_indices = set()
        method_counts = {}
        
        for method, result in method_results.items():
            if isinstance(result, dict) and 'outlier_indices' in result:
                indices = set(result['outlier_indices'])
                all_outlier_indices.update(indices)
                method_counts[method] = len(indices)
        
        # Find consensus outliers (detected by multiple methods)
        consensus_outliers = {}
        if len(method_results) > 1:
            for idx in all_outlier_indices:
                count = sum(1 for method, result in method_results.items()
                           if isinstance(result, dict) and 
                           'outlier_indices' in result and 
                           idx in result['outlier_indices'])
                if count > 1:
                    consensus_outliers[idx] = count
        
        return {
            'total_unique_outliers': len(all_outlier_indices),
            'outlier_percentage': (len(all_outlier_indices) / len(series)) * 100,
            'method_counts': method_counts,
            'consensus_outliers': consensus_outliers,
            'consensus_count': len(consensus_outliers)
        }
