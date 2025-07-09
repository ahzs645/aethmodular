"""Plotting utilities for aethalometer data"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path


class AethalometerPlotter:
    """Plotting utilities for aethalometer data visualization"""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter with style settings
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
        
    def plot_time_series(self, 
                        data: pd.DataFrame, 
                        columns: Optional[List[str]] = None,
                        title: str = "Aethalometer Time Series",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series of aethalometer data
        
        Args:
            data: DataFrame with datetime index
            columns: Columns to plot (if None, plots BC columns)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if columns is None:
            # Auto-detect BC columns
            columns = [col for col in data.columns if 'BC' in col and 'c' in col]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, col in enumerate(columns):
            if col in data.columns:
                ax.plot(data.index, data[col], 
                       label=col, color=self.colors[i % len(self.colors)], 
                       linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Date/Time')
        ax.set_ylabel('Black Carbon Concentration (μg/m³)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_spectral_dependence(self, 
                                data: pd.DataFrame,
                                wavelengths: Optional[List[str]] = None,
                                title: str = "Spectral Dependence",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot spectral dependence of black carbon
        
        Args:
            data: DataFrame with BC data
            wavelengths: Wavelength columns to plot
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if wavelengths is None:
            wavelengths = ['UV', 'Blue', 'Green', 'Red', 'IR']
        
        # Extract wavelength data
        wavelength_data = {}
        wl_values = {'UV': 370, 'Blue': 470, 'Green': 520, 'Red': 660, 'IR': 880}
        
        for wl in wavelengths:
            col_name = f'{wl} BCc'
            if col_name in data.columns:
                wavelength_data[wl_values.get(wl, 0)] = data[col_name].mean()
        
        if not wavelength_data:
            raise ValueError("No wavelength BC columns found in data")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        wavelengths_sorted = sorted(wavelength_data.keys())
        bc_values = [wavelength_data[wl] for wl in wavelengths_sorted]
        
        ax.loglog(wavelengths_sorted, bc_values, 'o-', 
                 markersize=8, linewidth=2, color='darkblue')
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Black Carbon Concentration (μg/m³)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add wavelength labels
        for wl, bc in zip(wavelengths_sorted, bc_values):
            ax.annotate(f'{wl}nm', (wl, bc), xytext=(5, 5), 
                       textcoords='offset points')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_source_apportionment(self, 
                                data: pd.DataFrame,
                                title: str = "Source Apportionment",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot biomass vs fossil fuel contributions
        
        Args:
            data: DataFrame with source apportionment data
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Time series plot
        if 'Biomass BCc' in data.columns:
            ax1.plot(data.index, data['Biomass BCc'], 
                    label='Biomass BC', color='green', linewidth=1.5)
        
        if 'Fossil fuel BCc' in data.columns:
            ax1.plot(data.index, data['Fossil fuel BCc'], 
                    label='Fossil Fuel BC', color='red', linewidth=1.5)
        
        ax1.set_ylabel('BC Concentration (μg/m³)')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Stacked area plot
        if all(col in data.columns for col in ['Biomass BCc', 'Fossil fuel BCc']):
            ax2.fill_between(data.index, 0, data['Biomass BCc'], 
                           label='Biomass BC', color='green', alpha=0.7)
            ax2.fill_between(data.index, data['Biomass BCc'], 
                           data['Biomass BCc'] + data['Fossil fuel BCc'],
                           label='Fossil Fuel BC', color='red', alpha=0.7)
        
        ax2.set_xlabel('Date/Time')
        ax2.set_ylabel('BC Concentration (μg/m³)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(self, 
                               data: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               title: str = "Correlation Matrix",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix of selected columns
        
        Args:
            data: DataFrame with data
            columns: Columns to include in correlation
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if columns is None:
            # Select numeric columns
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = data[columns].corr()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_diurnal_patterns(self, 
                             data: pd.DataFrame,
                             column: str = 'IR BCc',
                             title: str = "Diurnal Pattern",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot diurnal (hourly) patterns
        
        Args:
            data: DataFrame with datetime index
            column: Column to analyze
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        # Extract hour and calculate hourly means
        data_copy = data.copy()
        data_copy['hour'] = data_copy.index.hour
        hourly_stats = data_copy.groupby('hour')[column].agg(['mean', 'std'])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot mean with error bars
        ax.errorbar(hourly_stats.index, hourly_stats['mean'], 
                   yerr=hourly_stats['std'], marker='o', 
                   linewidth=2, markersize=6, capsize=5)
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel(f'{column} (μg/m³)')
        ax.set_title(title)
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(self, 
                        data: pd.DataFrame,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard
        
        Args:
            data: DataFrame with aethalometer data
            save_path: Path to save dashboard
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Time series
        ax1 = fig.add_subplot(gs[0, :])
        bc_columns = [col for col in data.columns if 'BC' in col and 'c' in col][:5]
        for i, col in enumerate(bc_columns):
            if col in data.columns:
                ax1.plot(data.index, data[col], label=col, 
                        color=self.colors[i % len(self.colors)])
        ax1.set_title('Black Carbon Time Series')
        ax1.set_ylabel('BC Concentration (μg/m³)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Source apportionment pie chart
        ax2 = fig.add_subplot(gs[1, 0])
        if all(col in data.columns for col in ['Biomass BCc', 'Fossil fuel BCc']):
            biomass_mean = data['Biomass BCc'].mean()
            fossil_mean = data['Fossil fuel BCc'].mean()
            ax2.pie([biomass_mean, fossil_mean], 
                   labels=['Biomass', 'Fossil Fuel'],
                   autopct='%1.1f%%', colors=['green', 'red'])
            ax2.set_title('Average Source Contribution')
        
        # Diurnal pattern
        ax3 = fig.add_subplot(gs[1, 1])
        if 'IR BCc' in data.columns:
            data_copy = data.copy()
            data_copy['hour'] = data_copy.index.hour
            hourly_mean = data_copy.groupby('hour')['IR BCc'].mean()
            ax3.plot(hourly_mean.index, hourly_mean.values, 'o-')
            ax3.set_title('Diurnal Pattern (IR BC)')
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('BC Concentration (μg/m³)')
            ax3.grid(True, alpha=0.3)
        
        # Statistics summary
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create statistics table
        stats_text = self._generate_stats_summary(data)
        ax4.text(0.1, 0.8, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Aethalometer Data Dashboard', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _generate_stats_summary(self, data: pd.DataFrame) -> str:
        """Generate statistics summary for dashboard"""
        bc_columns = [col for col in data.columns if 'BC' in col and 'c' in col]
        
        stats_lines = ['Data Summary:']
        stats_lines.append(f"Time period: {data.index.min()} to {data.index.max()}")
        stats_lines.append(f"Total data points: {len(data)}")
        stats_lines.append('')
        
        for col in bc_columns[:5]:  # Show first 5 BC columns
            if col in data.columns:
                mean_val = data[col].mean()
                std_val = data[col].std()
                max_val = data[col].max()
                stats_lines.append(f"{col}: Mean={mean_val:.2f}, Std={std_val:.2f}, Max={max_val:.2f} μg/m³")
        
        return '\n'.join(stats_lines)
