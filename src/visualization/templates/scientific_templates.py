"""
Scientific Analysis Visualization Templates

This module contains specialized templates for scientific analysis visualizations
including MAC analysis, correlation plots, and machine learning visualizations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from .base_template import BaseVisualizationTemplate

class MACAnalysisTemplate(BaseVisualizationTemplate):
    """Template for MAC (Mass Absorption Cross-section) analysis"""
    
    REQUIRED_PARAMS = ['fabs_data', 'ec_data']
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate MAC analysis parameters"""
        fabs_data = kwargs.get('fabs_data')
        ec_data = kwargs.get('ec_data')
        
        if fabs_data is None or ec_data is None:
            raise ValueError("Both 'fabs_data' and 'ec_data' are required")
        
        if len(fabs_data) != len(ec_data):
            raise ValueError("fabs_data and ec_data must have same length")
        
        return True
    
    def create_plot(self, **kwargs) -> plt.Figure:
        """Create MAC analysis plot with multiple methods"""
        self.validate_parameters(**kwargs)
        
        fabs_data = np.array(kwargs['fabs_data'])
        ec_data = np.array(kwargs['ec_data'])
        title = kwargs.get('title', 'MAC Analysis - Multiple Methods')
        
        # Filter valid data
        valid_mask = (fabs_data > 0) & (ec_data > 0) & np.isfinite(fabs_data) & np.isfinite(ec_data)
        fabs_clean = fabs_data[valid_mask]
        ec_clean = ec_data[valid_mask]
        
        if len(fabs_clean) < 5:
            raise ValueError("Insufficient valid data points for MAC analysis")
        
        # Calculate MAC using different methods
        mac_methods = self._calculate_mac_methods(fabs_clean, ec_clean)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (method_name, method_data) in enumerate(mac_methods.items()):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(ec_clean, fabs_clean, alpha=0.6, s=30)
            
            # Add regression line if available
            if 'predictions' in method_data:
                x_line = np.linspace(ec_clean.min(), ec_clean.max(), 100)
                if method_name == 'Linear Regression (Origin)':
                    y_line = method_data['mac_value'] * x_line
                else:
                    y_line = (method_data['mac_value'] * x_line + 
                             method_data.get('intercept', 0))
                ax.plot(x_line, y_line, color=colors[i], linewidth=2, 
                       label=f"MAC = {method_data['mac_value']:.2f}")
            
            ax.set_xlabel('EC (μg/m³)')
            ax.set_ylabel('Fabs (Mm⁻¹)')
            ax.set_title(f"{method_name}\nMAC = {method_data['mac_value']:.2f} m²/g")
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            if 'r_squared' in method_data:
                ax.text(0.05, 0.95, f"R² = {method_data['r_squared']:.3f}", 
                       transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def _calculate_mac_methods(self, fabs: np.ndarray, ec: np.ndarray) -> Dict[str, Dict]:
        """Calculate MAC using 4 different methods"""
        methods = {}
        
        # Method 1: Individual MAC mean
        individual_mac = fabs / ec
        methods['Individual MAC Mean'] = {
            'mac_value': np.mean(individual_mac),
            'std': np.std(individual_mac)
        }
        
        # Method 2: Ratio of means
        methods['Ratio of Means'] = {
            'mac_value': np.mean(fabs) / np.mean(ec)
        }
        
        # Try to use sklearn if available, otherwise use basic linear regression
        try:
            from sklearn.linear_model import LinearRegression
            
            # Method 3: Linear regression with intercept
            reg_intercept = LinearRegression().fit(ec.reshape(-1, 1), fabs)
            y_pred_intercept = reg_intercept.predict(ec.reshape(-1, 1))
            methods['Linear Regression (Intercept)'] = {
                'mac_value': reg_intercept.coef_[0],
                'intercept': reg_intercept.intercept_,
                'r_squared': reg_intercept.score(ec.reshape(-1, 1), fabs),
                'predictions': y_pred_intercept
            }
            
            # Method 4: Linear regression through origin
            reg_origin = LinearRegression(fit_intercept=False).fit(ec.reshape(-1, 1), fabs)
            y_pred_origin = reg_origin.predict(ec.reshape(-1, 1))
            methods['Linear Regression (Origin)'] = {
                'mac_value': reg_origin.coef_[0],
                'r_squared': reg_origin.score(ec.reshape(-1, 1), fabs),
                'predictions': y_pred_origin
            }
            
        except ImportError:
            # Fallback to basic numpy calculations
            # Method 3: Linear regression with intercept using numpy
            A = np.vstack([ec, np.ones(len(ec))]).T
            m, b = np.linalg.lstsq(A, fabs, rcond=None)[0]
            y_pred = m * ec + b
            r_squared = np.corrcoef(fabs, y_pred)[0, 1] ** 2
            
            methods['Linear Regression (Intercept)'] = {
                'mac_value': m,
                'intercept': b,
                'r_squared': r_squared,
                'predictions': y_pred
            }
            
            # Method 4: Linear regression through origin using numpy
            m_origin = np.linalg.lstsq(ec.reshape(-1, 1), fabs, rcond=None)[0][0]
            y_pred_origin = m_origin * ec
            r_squared_origin = np.corrcoef(fabs, y_pred_origin)[0, 1] ** 2
            
            methods['Linear Regression (Origin)'] = {
                'mac_value': m_origin,
                'r_squared': r_squared_origin,
                'predictions': y_pred_origin
            }
        
        return methods


class CorrelationAnalysisTemplate(BaseVisualizationTemplate):
    """Template for correlation analysis plots"""
    
    REQUIRED_PARAMS = ['data']
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate correlation analysis parameters"""
        data = kwargs.get('data')
        if not isinstance(data, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame")
        return True
    
    def create_plot(self, **kwargs) -> plt.Figure:
        """Create correlation analysis plot"""
        self.validate_parameters(**kwargs)
        
        data = kwargs['data']
        columns = kwargs.get('columns') or data.select_dtypes(include=[np.number]).columns.tolist()
        title = kwargs.get('title', 'Correlation Matrix')
        method = kwargs.get('method', 'pearson')
        
        # Calculate correlation matrix
        corr_matrix = data[columns].corr(method=method)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle if requested
        mask = kwargs.get('mask_upper', False)
        if mask:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        else:
            mask = None
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   mask=mask, ax=ax)
        
        ax.set_title(title, fontsize=16, pad=20)
        plt.tight_layout()
        return fig


class ScatterPlotTemplate(BaseVisualizationTemplate):
    """Template for scatter plot analysis"""
    
    REQUIRED_PARAMS = ['data', 'x_column', 'y_column']
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate scatter plot parameters"""
        data = kwargs.get('data')
        x_column = kwargs.get('x_column')
        y_column = kwargs.get('y_column')
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame")
        if x_column not in data.columns:
            raise ValueError(f"x_column '{x_column}' not found in data")
        if y_column not in data.columns:
            raise ValueError(f"y_column '{y_column}' not found in data")
        
        return True
    
    def create_plot(self, **kwargs) -> plt.Figure:
        """Create scatter plot with optional regression line"""
        self.validate_parameters(**kwargs)
        
        data = kwargs['data']
        x_column = kwargs['x_column']
        y_column = kwargs['y_column']
        color_column = kwargs.get('color_column')
        title = kwargs.get('title', f'{y_column} vs {x_column}')
        add_regression = kwargs.get('add_regression', False)
        
        fig, ax = plt.subplots(figsize=self.config['figsize'])
        
        # Create scatter plot
        if color_column and color_column in data.columns:
            scatter = ax.scatter(data[x_column], data[y_column], 
                               c=data[color_column], alpha=0.6, s=50)
            plt.colorbar(scatter, label=color_column)
        else:
            ax.scatter(data[x_column], data[y_column], alpha=0.6, s=50)
        
        # Add regression line if requested
        if add_regression:
            # Remove NaN values for regression
            clean_data = data[[x_column, y_column]].dropna()
            if len(clean_data) > 1:
                z = np.polyfit(clean_data[x_column], clean_data[y_column], 1)
                p = np.poly1d(z)
                ax.plot(clean_data[x_column], p(clean_data[x_column]), "r--", alpha=0.8)
                
                # Calculate R²
                correlation = np.corrcoef(clean_data[x_column], clean_data[y_column])[0, 1]
                r_squared = correlation ** 2
                ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                       transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
