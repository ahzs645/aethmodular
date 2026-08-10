"""Enhanced MAC calculation methods for FTIR analysis"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
from ...core.base import BaseAnalyzer
from ...data.processors.validation import validate_columns_exist, get_valid_data_mask

class EnhancedMACAnalyzer(BaseAnalyzer):
    """
    Enhanced MAC (Mass Absorption Cross-section) analyzer with multiple calculation methods
    
    Implements 4 different MAC calculation methods:
    1. Individual MAC mean: (∑fabs/ec)/n
    2. Ratio of means: ∑fabs/∑ec  
    3. Linear regression with intercept
    4. Linear regression through origin (forced zero intercept)
    
    Includes physical constraint validation (BC=0 when Fabs=0)
    """
    
    def __init__(self):
        super().__init__("EnhancedMACAnalyzer")
        self.required_columns = ['fabs', 'ec_ftir']
        self.mac_methods = {
            'method_1': 'Individual MAC mean',
            'method_2': 'Ratio of means', 
            'method_3': 'Linear regression with intercept',
            'method_4': 'Linear regression through origin'
        }
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive MAC analysis using all 4 methods
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with fabs and ec_ftir columns
            
        Returns:
        --------
        Dict[str, Any]
            Complete MAC analysis results with all methods
        """
        validate_columns_exist(data, self.required_columns)
        valid_mask = get_valid_data_mask(data, self.required_columns)
        
        if valid_mask.sum() < 5:
            raise ValueError(f"Insufficient valid data for MAC analysis: {valid_mask.sum()} samples")
        
        clean_data = data[valid_mask].copy()
        fabs = clean_data['fabs'].values
        ec = clean_data['ec_ftir'].values
        
        # Calculate MAC using all methods
        mac_results = {}
        
        # Method 1: Individual MAC mean
        mac_results['method_1'] = self._calculate_method_1(fabs, ec)
        
        # Method 2: Ratio of means
        mac_results['method_2'] = self._calculate_method_2(fabs, ec)
        
        # Method 3: Linear regression with intercept
        mac_results['method_3'] = self._calculate_method_3(fabs, ec)
        
        # Method 4: Linear regression through origin
        mac_results['method_4'] = self._calculate_method_4(fabs, ec)
        
        # Physical constraint validation
        constraint_analysis = self._validate_physical_constraints(mac_results, fabs, ec)
        
        # Performance metrics comparison
        performance_metrics = self._calculate_performance_metrics(mac_results, fabs, ec)
        
        # Method comparison and recommendations
        method_comparison = self._compare_methods(mac_results, performance_metrics, constraint_analysis)
        
        results = {
            'sample_info': {
                'total_samples': len(data),
                'valid_samples': len(clean_data),
                'data_completeness': len(clean_data) / len(data) * 100,
                'fabs_range': [float(fabs.min()), float(fabs.max())],
                'ec_range': [float(ec.min()), float(ec.max())]
            },
            'mac_methods': self.mac_methods,
            'mac_results': mac_results,
            'physical_constraints': constraint_analysis,
            'performance_metrics': performance_metrics,
            'method_comparison': method_comparison,
            'data_arrays': {
                'fabs': fabs,
                'ec_ftir': ec
            }
        }
        
        return results
    
    def _calculate_method_1(self, fabs: np.ndarray, ec: np.ndarray) -> Dict[str, Any]:
        """Method 1: Individual MAC mean (∑fabs/ec)/n"""
        # Calculate individual MAC values
        individual_mac = fabs / ec
        
        # Remove infinite and NaN values
        valid_mac = individual_mac[np.isfinite(individual_mac)]
        
        if len(valid_mac) == 0:
            return {'error': 'No valid MAC values calculated'}
        
        return {
            'mac_value': float(np.mean(valid_mac)),
            'mac_std': float(np.std(valid_mac)),
            'mac_median': float(np.median(valid_mac)),
            'mac_min': float(np.min(valid_mac)),
            'mac_max': float(np.max(valid_mac)),
            'valid_count': int(len(valid_mac)),
            'invalid_count': int(len(individual_mac) - len(valid_mac)),
            'individual_mac_values': valid_mac
        }
    
    def _calculate_method_2(self, fabs: np.ndarray, ec: np.ndarray) -> Dict[str, Any]:
        """Method 2: Ratio of means ∑fabs/∑ec"""
        fabs_sum = np.sum(fabs)
        ec_sum = np.sum(ec)
        
        if ec_sum == 0:
            return {'error': 'EC sum is zero, cannot calculate ratio'}
        
        mac_value = fabs_sum / ec_sum
        
        # Calculate uncertainty using error propagation
        fabs_std = np.std(fabs)
        ec_std = np.std(ec)
        fabs_mean = np.mean(fabs)
        ec_mean = np.mean(ec)
        
        # Relative uncertainty
        rel_uncertainty = np.sqrt((fabs_std/fabs_mean)**2 + (ec_std/ec_mean)**2)
        mac_uncertainty = mac_value * rel_uncertainty
        
        return {
            'mac_value': float(mac_value),
            'mac_uncertainty': float(mac_uncertainty),
            'relative_uncertainty': float(rel_uncertainty * 100),  # in percent
            'fabs_sum': float(fabs_sum),
            'ec_sum': float(ec_sum),
            'sample_count': int(len(fabs))
        }
    
    def _calculate_method_3(self, fabs: np.ndarray, ec: np.ndarray) -> Dict[str, Any]:
        """Method 3: Linear regression with intercept"""
        # Prepare data for regression
        X = ec.reshape(-1, 1)
        y = fabs
        
        # Perform linear regression
        reg = LinearRegression().fit(X, y)
        
        # Calculate predictions
        y_pred = reg.predict(X)
        
        # Calculate statistics
        r2_score = reg.score(X, y)
        correlation = np.corrcoef(ec, fabs)[0, 1]
        
        # Calculate residuals and RMSE
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Statistical significance test
        n = len(fabs)
        t_stat = correlation * np.sqrt((n-2)/(1-correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
        
        return {
            'mac_value': float(reg.coef_[0]),  # Slope is MAC
            'intercept': float(reg.intercept_),
            'r_squared': float(r2_score),
            'correlation': float(correlation),
            'rmse': float(rmse),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'predictions': y_pred,
            'residuals': residuals,
            'sample_count': int(n)
        }
    
    def _calculate_method_4(self, fabs: np.ndarray, ec: np.ndarray) -> Dict[str, Any]:
        """Method 4: Linear regression through origin (forced zero intercept)"""
        # Prepare data for regression through origin
        X = ec.reshape(-1, 1)
        y = fabs
        
        # Force intercept to be zero
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        
        # Calculate predictions
        y_pred = reg.predict(X)
        
        # Calculate R² for regression through origin
        ss_tot = np.sum(y**2)  # Total sum of squares (from zero)
        ss_res = np.sum((y - y_pred)**2)  # Residual sum of squares
        r2_score = 1 - (ss_res / ss_tot)
        
        # Calculate correlation
        correlation = np.corrcoef(ec, fabs)[0, 1]
        
        # Calculate residuals and RMSE
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Statistical significance test
        n = len(fabs)
        t_stat = correlation * np.sqrt((n-2)/(1-correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
        
        return {
            'mac_value': float(reg.coef_[0]),  # Slope is MAC
            'intercept': 0.0,  # Forced to zero
            'r_squared': float(r2_score),
            'correlation': float(correlation),
            'rmse': float(rmse),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'predictions': y_pred,
            'residuals': residuals,
            'sample_count': int(n)
        }
    
    def _validate_physical_constraints(self, mac_results: Dict, fabs: np.ndarray, 
                                     ec: np.ndarray) -> Dict[str, Any]:
        """Validate physical constraint: BC=0 when Fabs=0"""
        constraint_results = {}
        
        # Check if any methods violate the physical constraint
        # Physical constraint: when EC=0, Fabs should also be 0 (or very close)
        near_zero_ec = np.abs(ec) < 0.1  # EC values very close to zero
        
        if np.any(near_zero_ec):
            corresponding_fabs = fabs[near_zero_ec]
            violation_count = np.sum(np.abs(corresponding_fabs) > 0.5)  # Significant Fabs when EC~0
            
            constraint_results['near_zero_ec_count'] = int(np.sum(near_zero_ec))
            constraint_results['violation_count'] = int(violation_count)
            constraint_results['violation_percentage'] = float(violation_count / np.sum(near_zero_ec) * 100)
        else:
            constraint_results['near_zero_ec_count'] = 0
            constraint_results['violation_count'] = 0
            constraint_results['violation_percentage'] = 0.0
        
        # Check for negative predictions with each method
        for method_name, method_results in mac_results.items():
            if 'predictions' in method_results:
                predictions = method_results['predictions']
                negative_predictions = np.sum(predictions < 0)
                constraint_results[f'{method_name}_negative_predictions'] = int(negative_predictions)
                constraint_results[f'{method_name}_negative_percentage'] = float(
                    negative_predictions / len(predictions) * 100
                )
        
        # Overall constraint validation
        total_violations = sum(constraint_results.get(f'{method}_negative_predictions', 0) 
                             for method in mac_results.keys())
        
        constraint_results['overall_assessment'] = {
            'passes_constraint': constraint_results['violation_percentage'] < 5.0,  # <5% violations
            'total_negative_predictions': int(total_violations),
            'recommended_for_constraint': 'method_4'  # Through origin typically best for physical constraint
        }
        
        return constraint_results
    
    def _calculate_performance_metrics(self, mac_results: Dict, fabs: np.ndarray, 
                                     ec: np.ndarray) -> Dict[str, Any]:
        """Calculate performance metrics for method comparison"""
        metrics = {}
        
        for method_name, method_results in mac_results.items():
            if 'error' in method_results:
                metrics[method_name] = {'error': method_results['error']}
                continue
            
            method_metrics = {}
            
            # Get MAC value
            mac_value = method_results.get('mac_value', np.nan)
            method_metrics['mac_value'] = float(mac_value)
            
            # Bias calculation (for regression methods)
            if 'predictions' in method_results:
                predictions = method_results['predictions']
                bias = np.mean(predictions - fabs)
                method_metrics['bias'] = float(bias)
                method_metrics['absolute_bias'] = float(np.abs(bias))
                
                # Variance
                variance = np.var(predictions - fabs)
                method_metrics['variance'] = float(variance)
                
                # RMSE (already calculated for regression methods)
                method_metrics['rmse'] = method_results.get('rmse', np.nan)
                
                # R-squared
                method_metrics['r_squared'] = method_results.get('r_squared', np.nan)
            
            else:
                # For non-regression methods, calculate bias differently
                if method_name == 'method_1':
                    # Compare individual MAC approach
                    individual_mac = method_results.get('individual_mac_values', [])
                    if len(individual_mac) > 0:
                        predicted_fabs = mac_value * ec
                        bias = np.mean(predicted_fabs - fabs)
                        method_metrics['bias'] = float(bias)
                        method_metrics['absolute_bias'] = float(np.abs(bias))
                        method_metrics['variance'] = float(np.var(individual_mac))
                        method_metrics['rmse'] = float(np.sqrt(np.mean((predicted_fabs - fabs)**2)))
                
                elif method_name == 'method_2':
                    # For ratio of means
                    predicted_fabs = mac_value * ec
                    bias = np.mean(predicted_fabs - fabs)
                    method_metrics['bias'] = float(bias)
                    method_metrics['absolute_bias'] = float(np.abs(bias))
                    method_metrics['rmse'] = float(np.sqrt(np.mean((predicted_fabs - fabs)**2)))
            
            # Physical constraint penalty
            negative_predictions = method_results.get('negative_predictions', 0)
            if isinstance(negative_predictions, int):
                method_metrics['constraint_penalty'] = float(negative_predictions / len(fabs) * 100)
            else:
                method_metrics['constraint_penalty'] = 0.0
            
            metrics[method_name] = method_metrics
        
        return metrics
    
    def _compare_methods(self, mac_results: Dict, performance_metrics: Dict, 
                        constraint_analysis: Dict) -> Dict[str, Any]:
        """Compare methods and provide recommendations"""
        comparison = {
            'mac_values_comparison': {},
            'performance_ranking': {},
            'recommendations': {}
        }
        
        # Extract MAC values for comparison
        mac_values = {}
        for method, results in mac_results.items():
            if 'mac_value' in results:
                mac_values[method] = results['mac_value']
        
        comparison['mac_values_comparison'] = mac_values
        
        # Calculate MAC value statistics across methods
        if mac_values:
            values = list(mac_values.values())
            comparison['mac_statistics'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'coefficient_of_variation': float(np.std(values) / np.mean(values) * 100)
            }
        
        # Rank methods by performance criteria
        ranking_criteria = ['rmse', 'absolute_bias', 'constraint_penalty']
        method_scores = {}
        
        for method in mac_results.keys():
            if method in performance_metrics and 'error' not in performance_metrics[method]:
                score = 0
                # Lower RMSE is better
                rmse = performance_metrics[method].get('rmse', float('inf'))
                score += 1 / (1 + rmse) if rmse != float('inf') else 0
                
                # Lower absolute bias is better
                abs_bias = performance_metrics[method].get('absolute_bias', float('inf'))
                score += 1 / (1 + abs_bias) if abs_bias != float('inf') else 0
                
                # Lower constraint penalty is better
                penalty = performance_metrics[method].get('constraint_penalty', 100)
                score += (100 - penalty) / 100
                
                method_scores[method] = score
        
        # Sort by score (higher is better)
        ranked_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['performance_ranking'] = {
            f'rank_{i+1}': {'method': method, 'score': score} 
            for i, (method, score) in enumerate(ranked_methods)
        }
        
        # Generate recommendations
        if ranked_methods:
            best_method = ranked_methods[0][0]
            comparison['recommendations'] = {
                'best_overall': best_method,
                'best_for_constraints': 'method_4',  # Through origin typically best
                'most_conservative': 'method_2',     # Ratio of means
                'most_detailed': 'method_1',         # Individual MAC values
                'rationale': self._generate_recommendation_rationale(
                    best_method, mac_results, performance_metrics, constraint_analysis
                )
            }
        
        return comparison
    
    def _generate_recommendation_rationale(self, best_method: str, mac_results: Dict,
                                         performance_metrics: Dict, constraint_analysis: Dict) -> str:
        """Generate rationale for method recommendation"""
        method_names = {
            'method_1': 'Individual MAC mean',
            'method_2': 'Ratio of means',
            'method_3': 'Linear regression with intercept',
            'method_4': 'Linear regression through origin'
        }
        
        best_name = method_names.get(best_method, best_method)
        
        rationale = f"Method {best_method[-1]} ({best_name}) is recommended based on: "
        
        reasons = []
        
        # Check performance metrics
        if best_method in performance_metrics:
            metrics = performance_metrics[best_method]
            
            if metrics.get('rmse', float('inf')) < 2.0:
                reasons.append("low RMSE")
            
            if metrics.get('absolute_bias', float('inf')) < 1.0:
                reasons.append("minimal bias")
            
            if metrics.get('constraint_penalty', 100) < 5.0:
                reasons.append("good physical constraint compliance")
            
            if metrics.get('r_squared', 0) > 0.8:
                reasons.append("high correlation")
        
        # Check for physical constraints
        constraint_ok = constraint_analysis.get('overall_assessment', {}).get('passes_constraint', False)
        if constraint_ok:
            reasons.append("satisfies physical constraints")
        
        if not reasons:
            reasons = ["balanced performance across metrics"]
        
        rationale += ", ".join(reasons) + "."
        
        return rationale
