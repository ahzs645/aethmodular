"""Smoothening methods comparison and evaluation utilities"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .smoothening_factory import SmoothingFactory


class SmoothingComparison:
    """
    Utility class for comparing multiple smoothening methods
    
    Provides functionality to run multiple smoothening algorithms on the same data
    and compare their performance metrics.
    """
    
    def __init__(self, methods: Optional[List[str]] = None):
        """
        Initialize comparison with specified methods
        
        Parameters:
        -----------
        methods : List[str], optional
            List of smoothening methods to compare. If None, uses all available methods.
        """
        self.methods = methods or SmoothingFactory.get_available_methods()
        self.results = {}
    
    def compare_methods(self, data: pd.DataFrame, wavelength: str = 'IR', 
                       method_params: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Compare multiple smoothening methods on the same data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Aethalometer data
        wavelength : str
            Wavelength to analyze
        method_params : Dict[str, Dict], optional
            Method-specific parameters. Format: {'METHOD': {'param': value}}
            
        Returns:
        --------
        Dict[str, Any]
            Comparison results including individual method results and summary
        """
        method_params = method_params or {}
        results = {}
        
        print(f"Comparing {len(self.methods)} smoothening methods on {wavelength} wavelength...")
        
        for method in self.methods:
            try:
                # Get method-specific parameters
                params = method_params.get(method, {})
                
                # Create smoother and analyze
                smoother = SmoothingFactory.create_smoother(method, **params)
                result = smoother.analyze(data, wavelength)
                results[method] = result
                
                print(f"✓ {method}: {result['improvement_metrics']['noise_reduction_percent']:.1f}% noise reduction")
                
            except Exception as e:
                print(f"✗ {method}: Failed - {str(e)}")
                results[method] = {'error': str(e)}
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(results)
        
        self.results = {
            'individual_results': results,
            'comparison_summary': summary,
            'wavelength': wavelength,
            'data_info': {
                'total_samples': len(data),
                'analysis_timestamp': pd.Timestamp.now()
            }
        }
        
        return self.results
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary comparing all methods"""
        summary = {
            'best_noise_reduction': None,
            'best_correlation': None,
            'best_negative_reduction': None,
            'lowest_lag': None,
            'performance_ranking': [],
            'metrics_comparison': {}
        }
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return summary
        
        # Extract metrics for comparison
        metrics = ['noise_reduction_percent', 'correlation_with_original', 'negative_reduction']
        
        for metric in metrics:
            summary['metrics_comparison'][metric] = {}
            for method, result in valid_results.items():
                if 'improvement_metrics' in result:
                    value = result['improvement_metrics'].get(metric, 0)
                    summary['metrics_comparison'][metric][method] = value
        
        # Find best performers
        if 'noise_reduction_percent' in summary['metrics_comparison']:
            noise_reduction = summary['metrics_comparison']['noise_reduction_percent']
            if noise_reduction:
                best_noise = max(noise_reduction.items(), key=lambda x: x[1])
                summary['best_noise_reduction'] = {'method': best_noise[0], 'value': best_noise[1]}
        
        if 'correlation_with_original' in summary['metrics_comparison']:
            correlation = summary['metrics_comparison']['correlation_with_original']
            if correlation:
                best_corr = max(correlation.items(), key=lambda x: x[1])
                summary['best_correlation'] = {'method': best_corr[0], 'value': best_corr[1]}
        
        if 'negative_reduction' in summary['metrics_comparison']:
            neg_reduction = summary['metrics_comparison']['negative_reduction']
            if neg_reduction:
                best_neg = max(neg_reduction.items(), key=lambda x: x[1])
                summary['best_negative_reduction'] = {'method': best_neg[0], 'value': best_neg[1]}
        
        # Handle DEMA-specific lag metric
        lag_metrics = {}
        for method, result in valid_results.items():
            if 'improvement_metrics' in result and 'lag_metric' in result['improvement_metrics']:
                lag_metrics[method] = result['improvement_metrics']['lag_metric']
        
        if lag_metrics:
            best_lag = min(lag_metrics.items(), key=lambda x: x[1])
            summary['lowest_lag'] = {'method': best_lag[0], 'value': best_lag[1]}
        
        # Generate overall ranking (simple scoring system)
        summary['performance_ranking'] = self._rank_methods(valid_results)
        
        return summary
    
    def _rank_methods(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank methods based on multiple criteria"""
        scores = {}
        
        for method, result in results.items():
            if 'improvement_metrics' not in result:
                continue
                
            metrics = result['improvement_metrics']
            score = 0
            
            # Weight different metrics
            score += metrics.get('noise_reduction_percent', 0) * 0.4
            score += metrics.get('correlation_with_original', 0) * 100 * 0.3
            score += metrics.get('negative_reduction', 0) * 0.2
            
            # Penalize lag for DEMA
            if 'lag_metric' in metrics:
                score -= metrics['lag_metric'] * 0.1
            
            scores[method] = score
        
        # Sort by score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                'rank': i + 1,
                'method': method,
                'score': round(score, 2)
            }
            for i, (method, score) in enumerate(ranked)
        ]
    
    def get_best_method(self, criterion: str = 'overall') -> Optional[str]:
        """
        Get the best method based on specified criterion
        
        Parameters:
        -----------
        criterion : str
            Criterion for selection ('overall', 'noise_reduction', 'correlation', 'lag')
            
        Returns:
        --------
        str or None
            Name of the best method
        """
        if not self.results or 'comparison_summary' not in self.results:
            return None
        
        summary = self.results['comparison_summary']
        
        if criterion == 'overall':
            if summary['performance_ranking']:
                return summary['performance_ranking'][0]['method']
        elif criterion == 'noise_reduction':
            if summary['best_noise_reduction']:
                return summary['best_noise_reduction']['method']
        elif criterion == 'correlation':
            if summary['best_correlation']:
                return summary['best_correlation']['method']
        elif criterion == 'lag':
            if summary['lowest_lag']:
                return summary['lowest_lag']['method']
        
        return None
    
    def print_comparison_report(self):
        """Print a formatted comparison report"""
        if not self.results:
            print("No comparison results available. Run compare_methods() first.")
            return
        
        print("\n" + "="*80)
        print("SMOOTHENING METHODS COMPARISON REPORT")
        print("="*80)
        
        summary = self.results['comparison_summary']
        
        # Best performers
        print("\n🏆 BEST PERFORMERS:")
        if summary['best_noise_reduction']:
            best_noise = summary['best_noise_reduction']
            print(f"  • Noise Reduction: {best_noise['method']} ({best_noise['value']:.1f}%)")
        
        if summary['best_correlation']:
            best_corr = summary['best_correlation']
            print(f"  • Correlation: {best_corr['method']} ({best_corr['value']:.3f})")
        
        if summary['best_negative_reduction']:
            best_neg = summary['best_negative_reduction']
            print(f"  • Negative Value Reduction: {best_neg['method']} ({best_neg['value']} values)")
        
        if summary['lowest_lag']:
            best_lag = summary['lowest_lag']
            print(f"  • Lowest Lag: {best_lag['method']} ({best_lag['value']:.3f})")
        
        # Overall ranking
        if summary['performance_ranking']:
            print("\n📊 OVERALL RANKING:")
            for rank_info in summary['performance_ranking']:
                print(f"  {rank_info['rank']}. {rank_info['method']} (Score: {rank_info['score']})")
        
        # Individual method details
        print("\n📋 METHOD DETAILS:")
        for method, result in self.results['individual_results'].items():
            if 'error' in result:
                print(f"  ❌ {method}: {result['error']}")
            else:
                metrics = result['improvement_metrics']
                print(f"  ✅ {method}:")
                print(f"     - Noise reduction: {metrics['noise_reduction_percent']:.1f}%")
                print(f"     - Correlation: {metrics['correlation_with_original']:.3f}")
                print(f"     - Negative reduction: {metrics['negative_reduction']} values")
                if 'lag_metric' in metrics:
                    print(f"     - Lag metric: {metrics['lag_metric']:.3f}")
        
        print("\n" + "="*80)
