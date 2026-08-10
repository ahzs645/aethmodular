"""Statistical analysis utilities for aethalometer data"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple
from src.core.base import BaseAnalyzer


class StatisticalAnalyzer(BaseAnalyzer):
    """Statistical analysis for aethalometer data"""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis
        
        Args:
            data: DataFrame with aethalometer data
            
        Returns:
            Dict containing statistical analysis results
        """
        results = {
            'descriptive_stats': self.descriptive_statistics(data),
            'correlation_analysis': self.correlation_analysis(data),
            'normality_tests': self.test_normality(data),
            'trend_analysis': self.trend_analysis(data),
            'outlier_detection': self.detect_outliers(data)
        }
        
        return results
    
    def descriptive_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dict with descriptive statistics
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        stats_dict = {}
        for col in numeric_columns:
            if data[col].notna().sum() > 0:  # Skip empty columns
                col_data = data[col].dropna()
                
                stats_dict[col] = {
                    'count': len(col_data),
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'var': col_data.var(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q25': col_data.quantile(0.25),
                    'q75': col_data.quantile(0.75),
                    'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),
                    'skewness': stats.skew(col_data),
                    'kurtosis': stats.kurtosis(col_data),
                    'coefficient_of_variation': col_data.std() / col_data.mean() if col_data.mean() != 0 else np.nan
                }
        
        return stats_dict
    
    def correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform correlation analysis
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dict with correlation analysis results
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Pearson correlation
        pearson_corr = numeric_data.corr(method='pearson')
        
        # Spearman correlation (rank-based, good for non-linear relationships)
        spearman_corr = numeric_data.corr(method='spearman')
        
        # Find highly correlated pairs
        high_correlations = self._find_high_correlations(pearson_corr)
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'high_correlations': high_correlations,
            'correlation_summary': self._summarize_correlations(high_correlations)
        }
    
    def test_normality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test normality of data distributions
        
        Args:
            data: DataFrame to test
            
        Returns:
            Dict with normality test results
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        normality_results = {}
        for col in numeric_columns:
            col_data = data[col].dropna()
            
            if len(col_data) > 3:  # Minimum sample size for tests
                # Shapiro-Wilk test (good for small samples)
                shapiro_stat, shapiro_p = stats.shapiro(col_data)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(col_data, 'norm', 
                                           args=(col_data.mean(), col_data.std()))
                
                # Anderson-Darling test
                ad_stat, ad_critical, ad_significance = stats.anderson(col_data, dist='norm')
                
                normality_results[col] = {
                    'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p, 
                                   'is_normal': shapiro_p > self.alpha},
                    'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p,
                                         'is_normal': ks_p > self.alpha},
                    'anderson_darling': {'statistic': ad_stat, 'critical_values': ad_critical,
                                       'significance_levels': ad_significance}
                }
        
        return normality_results
    
    def trend_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trends in time series data
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            Dict with trend analysis results
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            return {'error': 'Data must have datetime index for trend analysis'}
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        trend_results = {}
        
        for col in numeric_columns:
            col_data = data[col].dropna()
            
            if len(col_data) > 10:  # Minimum data points for trend analysis
                # Convert datetime to numeric for regression
                x = np.arange(len(col_data))
                y = col_data.values
                
                # Linear regression for trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Mann-Kendall trend test (non-parametric)
                mk_trend, mk_p = self._mann_kendall_test(y)
                
                trend_results[col] = {
                    'linear_trend': {
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'std_error': std_err,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'no trend'
                    },
                    'mann_kendall': {
                        'trend': mk_trend,
                        'p_value': mk_p,
                        'is_significant': mk_p < self.alpha
                    }
                }
        
        return trend_results
    
    def detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers using multiple methods
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dict with outlier detection results
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_results = {}
        
        for col in numeric_columns:
            col_data = data[col].dropna()
            
            if len(col_data) > 4:  # Minimum sample size
                # IQR method
                q1, q3 = col_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                # Z-score method
                z_scores = np.abs(stats.zscore(col_data))
                z_outliers = col_data[z_scores > 3]  # Beyond 3 standard deviations
                
                # Modified Z-score method (more robust)
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                modified_z_scores = 0.6745 * (col_data - median) / mad
                modified_z_outliers = col_data[np.abs(modified_z_scores) > 3.5]
                
                outlier_results[col] = {
                    'iqr_method': {
                        'outliers': iqr_outliers.index.tolist(),
                        'count': len(iqr_outliers),
                        'percentage': len(iqr_outliers) / len(col_data) * 100,
                        'bounds': {'lower': lower_bound, 'upper': upper_bound}
                    },
                    'z_score_method': {
                        'outliers': z_outliers.index.tolist(),
                        'count': len(z_outliers),
                        'percentage': len(z_outliers) / len(col_data) * 100
                    },
                    'modified_z_score_method': {
                        'outliers': modified_z_outliers.index.tolist(),
                        'count': len(modified_z_outliers),
                        'percentage': len(modified_z_outliers) / len(col_data) * 100
                    }
                }
        
        return outlier_results
    
    def compare_groups(self, data: pd.DataFrame, group_column: str, 
                      value_column: str) -> Dict[str, Any]:
        """
        Compare groups using statistical tests
        
        Args:
            data: DataFrame with data
            group_column: Column defining groups
            value_column: Column with values to compare
            
        Returns:
            Dict with group comparison results
        """
        groups = data.groupby(group_column)[value_column].apply(list)
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for comparison'}
        
        results = {
            'group_statistics': {},
            'statistical_tests': {}
        }
        
        # Calculate group statistics
        for group_name, group_data in groups.items():
            group_array = np.array(group_data)
            results['group_statistics'][group_name] = {
                'count': len(group_array),
                'mean': np.mean(group_array),
                'std': np.std(group_array),
                'median': np.median(group_array)
            }
        
        # Perform statistical tests
        if len(groups) == 2:
            # Two-sample tests
            group1, group2 = list(groups.values())
            
            # t-test
            t_stat, t_p = stats.ttest_ind(group1, group2)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            
            results['statistical_tests'] = {
                'two_sample_t_test': {
                    'statistic': t_stat,
                    'p_value': t_p,
                    'significant': t_p < self.alpha
                },
                'mann_whitney_u': {
                    'statistic': u_stat,
                    'p_value': u_p,
                    'significant': u_p < self.alpha
                }
            }
        else:
            # Multiple group tests
            group_values = list(groups.values())
            
            # ANOVA
            f_stat, f_p = stats.f_oneway(*group_values)
            
            # Kruskal-Wallis test (non-parametric)
            h_stat, h_p = stats.kruskal(*group_values)
            
            results['statistical_tests'] = {
                'one_way_anova': {
                    'statistic': f_stat,
                    'p_value': f_p,
                    'significant': f_p < self.alpha
                },
                'kruskal_wallis': {
                    'statistic': h_stat,
                    'p_value': h_p,
                    'significant': h_p < self.alpha
                }
            }
        
        return results
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, 
                               threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find pairs with high correlation"""
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': self._correlation_strength(abs(corr_value))
                    })
        
        return sorted(high_corr, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _correlation_strength(self, corr_value: float) -> str:
        """Categorize correlation strength"""
        if corr_value >= 0.9:
            return 'very strong'
        elif corr_value >= 0.7:
            return 'strong'
        elif corr_value >= 0.5:
            return 'moderate'
        elif corr_value >= 0.3:
            return 'weak'
        else:
            return 'very weak'
    
    def _summarize_correlations(self, high_correlations: List[Dict[str, Any]]) -> str:
        """Generate correlation summary"""
        if not high_correlations:
            return "No high correlations found (threshold >= 0.7)"
        
        summary_lines = [f"Found {len(high_correlations)} high correlations:"]
        
        for corr in high_correlations[:10]:  # Show top 10
            summary_lines.append(
                f"  {corr['variable1']} ↔ {corr['variable2']}: "
                f"{corr['correlation']:.3f} ({corr['strength']})"
            )
        
        if len(high_correlations) > 10:
            summary_lines.append(f"  ... and {len(high_correlations) - 10} more")
        
        return '\n'.join(summary_lines)
    
    def _mann_kendall_test(self, data: np.ndarray) -> Tuple[str, float]:
        """
        Perform Mann-Kendall trend test
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (trend, p_value)
        """
        n = len(data)
        s = 0
        
        for i in range(n-1):
            for j in range(i+1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Determine trend
        if p_value < self.alpha:
            trend = 'increasing' if s > 0 else 'decreasing'
        else:
            trend = 'no trend'
        
        return trend, p_value
