"""Machine learning analysis module for aethalometer data"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
from ...core.base import BaseAnalyzer
from ...core.monitoring import monitor_performance, handle_errors
from ...utils.logging.logger import get_logger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class MLModelTrainer(BaseAnalyzer):
    """
    Train machine learning models for aethalometer data analysis
    """
    
    def __init__(self):
        super().__init__("MLModelTrainer")
        self.logger = get_logger(__name__)
        self.models = {}
        self.scalers = {}
        
    @monitor_performance
    @handle_errors
    def train_regression_model(self, data: pd.DataFrame, 
                              target_column: str,
                              feature_columns: Optional[List[str]] = None,
                              model_type: str = 'random_forest',
                              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train regression model for predicting target variable
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
        target_column : str
            Column to predict
        feature_columns : List[str], optional
            Feature columns to use. If None, uses all numeric columns except target
        model_type : str
            Type of model: 'random_forest', 'linear', 'svr', 'gradient_boosting'
        test_size : float
            Fraction of data to use for testing
            
        Returns:
        --------
        Dict[str, Any]
            Training results and model performance
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            # Prepare features
            if feature_columns is None:
                feature_columns = [col for col in data.columns 
                                 if pd.api.types.is_numeric_dtype(data[col]) and col != target_column]
            
            # Clean data
            analysis_data = data[feature_columns + [target_column]].dropna()
            
            if len(analysis_data) < 50:
                raise ValueError("Insufficient data for model training (need at least 50 samples)")
            
            X = analysis_data[feature_columns]
            y = analysis_data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self._get_regression_model(model_type)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            results = {
                'model_type': model_type,
                'target_column': target_column,
                'feature_columns': feature_columns,
                'data_summary': {
                    'total_samples': len(analysis_data),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features_count': len(feature_columns)
                },
                'performance': {
                    'training': {
                        'mse': float(mean_squared_error(y_train, y_train_pred)),
                        'rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                        'mae': float(mean_absolute_error(y_train, y_train_pred)),
                        'r2': float(r2_score(y_train, y_train_pred))
                    },
                    'test': {
                        'mse': float(mean_squared_error(y_test, y_test_pred)),
                        'rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
                        'mae': float(mean_absolute_error(y_test, y_test_pred)),
                        'r2': float(r2_score(y_test, y_test_pred))
                    }
                }
            }
            
            # Add feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                results['feature_importance'] = feature_importance.to_dict('records')
            
            # Store model and scaler
            model_key = f"{model_type}_{target_column}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            return results
            
        except ImportError:
            return {'error': 'scikit-learn not available for ML model training'}
        except Exception as e:
            return {'error': str(e)}
    
    def _get_regression_model(self, model_type: str):
        """Get regression model based on type"""
        try:
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'linear':
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            elif model_type == 'svr':
                from sklearn.svm import SVR
                return SVR(kernel='rbf')
            elif model_type == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except ImportError as e:
            raise ImportError(f"Required sklearn module not available: {e}")
    
    @monitor_performance
    def predict(self, model_key: str, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model
        
        Parameters:
        -----------
        model_key : str
            Key for the stored model
        data : pd.DataFrame
            Data to make predictions on
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        
        # Scale data
        data_scaled = scaler.transform(data)
        
        # Make predictions
        predictions = model.predict(data_scaled)
        
        return predictions


class PredictiveAnalyzer(BaseAnalyzer):
    """
    Perform predictive analysis on aethalometer data
    """
    
    def __init__(self):
        super().__init__("PredictiveAnalyzer")
        self.logger = get_logger(__name__)
        
    @monitor_performance
    @handle_errors
    def forecast_time_series(self, data: pd.Series, 
                           forecast_periods: int = 24,
                           method: str = 'simple_exponential') -> Dict[str, Any]:
        """
        Forecast time series data
        
        Parameters:
        -----------
        data : pd.Series
            Time series data with datetime index
        forecast_periods : int
            Number of periods to forecast
        method : str
            Forecasting method: 'simple_exponential', 'holt', 'linear_trend'
            
        Returns:
        --------
        Dict[str, Any]
            Forecasting results
        """
        # Clean data
        clean_data = data.dropna()
        
        if len(clean_data) < 24:
            raise ValueError("Insufficient data for forecasting (need at least 24 points)")
        
        if method == 'simple_exponential':
            return self._simple_exponential_smoothing(clean_data, forecast_periods)
        elif method == 'holt':
            return self._holt_smoothing(clean_data, forecast_periods)
        elif method == 'linear_trend':
            return self._linear_trend_forecast(clean_data, forecast_periods)
        else:
            raise ValueError(f"Unknown forecasting method: {method}")
    
    def _simple_exponential_smoothing(self, data: pd.Series, periods: int) -> Dict[str, Any]:
        """Simple exponential smoothing forecast"""
        alpha = 0.3  # Smoothing parameter
        
        # Calculate smoothed values
        smoothed = [data.iloc[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data.iloc[i] + (1 - alpha) * smoothed[-1])
        
        # Forecast
        last_smoothed = smoothed[-1]
        forecast = [last_smoothed] * periods
        
        # Generate forecast index
        last_time = data.index[-1]
        freq = pd.infer_freq(data.index)
        if freq is None:
            freq = 'H'  # Default to hourly
            
        forecast_index = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=periods,
            freq=freq
        )
        
        return {
            'method': 'Simple Exponential Smoothing',
            'parameters': {'alpha': alpha},
            'forecast_values': forecast,
            'forecast_index': forecast_index.tolist(),
            'fitted_values': smoothed,
            'original_data_length': len(data),
            'forecast_periods': periods
        }
    
    def _holt_smoothing(self, data: pd.Series, periods: int) -> Dict[str, Any]:
        """Holt's linear trend method"""
        alpha = 0.3  # Level smoothing parameter
        beta = 0.1   # Trend smoothing parameter
        
        # Initialize
        level = data.iloc[0]
        trend = data.iloc[1] - data.iloc[0] if len(data) > 1 else 0
        
        fitted = [level]
        levels = [level]
        trends = [trend]
        
        # Calculate smoothed values
        for i in range(1, len(data)):
            prev_level = level
            level = alpha * data.iloc[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            
            fitted.append(level)
            levels.append(level)
            trends.append(trend)
        
        # Forecast
        forecast = []
        for h in range(1, periods + 1):
            forecast.append(level + h * trend)
        
        # Generate forecast index
        last_time = data.index[-1]
        freq = pd.infer_freq(data.index)
        if freq is None:
            freq = 'H'
            
        forecast_index = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=periods,
            freq=freq
        )
        
        return {
            'method': 'Holt Linear Trend',
            'parameters': {'alpha': alpha, 'beta': beta},
            'forecast_values': forecast,
            'forecast_index': forecast_index.tolist(),
            'fitted_values': fitted,
            'final_level': level,
            'final_trend': trend,
            'original_data_length': len(data),
            'forecast_periods': periods
        }
    
    def _linear_trend_forecast(self, data: pd.Series, periods: int) -> Dict[str, Any]:
        """Linear trend forecast using least squares"""
        # Create time index (0, 1, 2, ...)
        t = np.arange(len(data))
        
        # Fit linear trend
        coeffs = np.polyfit(t, data.values, 1)
        slope, intercept = coeffs
        
        # Generate fitted values
        fitted = slope * t + intercept
        
        # Forecast
        future_t = np.arange(len(data), len(data) + periods)
        forecast = slope * future_t + intercept
        
        # Calculate R-squared
        ss_res = np.sum((data.values - fitted) ** 2)
        ss_tot = np.sum((data.values - np.mean(data.values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Generate forecast index
        last_time = data.index[-1]
        freq = pd.infer_freq(data.index)
        if freq is None:
            freq = 'H'
            
        forecast_index = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=periods,
            freq=freq
        )
        
        return {
            'method': 'Linear Trend',
            'parameters': {'slope': float(slope), 'intercept': float(intercept)},
            'forecast_values': forecast.tolist(),
            'forecast_index': forecast_index.tolist(),
            'fitted_values': fitted.tolist(),
            'r_squared': float(r_squared),
            'original_data_length': len(data),
            'forecast_periods': periods
        }


class ClusterAnalyzer(BaseAnalyzer):
    """
    Perform cluster analysis on aethalometer data
    """
    
    def __init__(self):
        super().__init__("ClusterAnalyzer")
        self.logger = get_logger(__name__)
        
    @monitor_performance
    @handle_errors
    def perform_clustering(self, data: pd.DataFrame,
                          columns: Optional[List[str]] = None,
                          n_clusters: int = 3,
                          method: str = 'kmeans') -> Dict[str, Any]:
        """
        Perform clustering analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to cluster
        columns : List[str], optional
            Columns to use for clustering
        n_clusters : int
            Number of clusters
        method : str
            Clustering method: 'kmeans', 'hierarchical'
            
        Returns:
        --------
        Dict[str, Any]
            Clustering results
        """
        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            
            # Prepare data
            if columns is None:
                columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            
            analysis_data = data[columns].dropna()
            
            if len(analysis_data) < n_clusters:
                raise ValueError(f"Insufficient data for clustering (need at least {n_clusters} samples)")
            
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(analysis_data)
            
            # Perform clustering
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == 'hierarchical':
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            cluster_labels = clusterer.fit_predict(scaled_data)
            
            # Calculate metrics
            silhouette = silhouette_score(scaled_data, cluster_labels)
            
            # Analyze clusters
            cluster_analysis = self._analyze_clusters(analysis_data, cluster_labels, columns)
            
            results = {
                'method': method,
                'n_clusters': n_clusters,
                'columns_used': columns,
                'data_summary': {
                    'total_samples': len(analysis_data),
                    'features_count': len(columns)
                },
                'metrics': {
                    'silhouette_score': float(silhouette)
                },
                'cluster_labels': cluster_labels.tolist(),
                'cluster_analysis': cluster_analysis
            }
            
            # Add cluster centers for kmeans
            if method == 'kmeans':
                # Transform centers back to original scale
                centers_scaled = clusterer.cluster_centers_
                centers_original = scaler.inverse_transform(centers_scaled)
                
                centers_df = pd.DataFrame(centers_original, columns=columns)
                results['cluster_centers'] = centers_df.to_dict('records')
            
            return results
            
        except ImportError:
            return {'error': 'scikit-learn not available for clustering analysis'}
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_clusters(self, data: pd.DataFrame, labels: np.ndarray, 
                         columns: List[str]) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        cluster_stats = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            
            # Calculate statistics for each column
            stats = {}
            for col in columns:
                stats[col] = {
                    'mean': float(cluster_data[col].mean()),
                    'std': float(cluster_data[col].std()),
                    'min': float(cluster_data[col].min()),
                    'max': float(cluster_data[col].max()),
                    'count': int(cluster_data[col].count())
                }
            
            cluster_stats[f'cluster_{cluster_id}'] = {
                'size': int(np.sum(cluster_mask)),
                'percentage': float((np.sum(cluster_mask) / len(labels)) * 100),
                'statistics': stats
            }
        
        return cluster_stats
    
    @monitor_performance
    @handle_errors
    def find_optimal_clusters(self, data: pd.DataFrame,
                             columns: Optional[List[str]] = None,
                             max_clusters: int = 10,
                             method: str = 'kmeans') -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to analyze
        columns : List[str], optional
            Columns to use for clustering
        max_clusters : int
            Maximum number of clusters to test
        method : str
            Clustering method
            
        Returns:
        --------
        Dict[str, Any]
            Analysis of optimal cluster count
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            
            # Prepare data
            if columns is None:
                columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            
            analysis_data = data[columns].dropna()
            
            if len(analysis_data) < max_clusters:
                raise ValueError(f"Insufficient data for cluster analysis")
            
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(analysis_data)
            
            results = {
                'method': method,
                'max_clusters_tested': max_clusters,
                'columns_used': columns,
                'cluster_analysis': {}
            }
            
            inertias = []
            silhouette_scores = []
            
            for k in range(2, min(max_clusters + 1, len(analysis_data))):
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=k, random_state=42)
                    labels = clusterer.fit_predict(scaled_data)
                    inertias.append(clusterer.inertia_)
                else:
                    # For other methods, use kmeans for inertia calculation
                    temp_kmeans = KMeans(n_clusters=k, random_state=42)
                    temp_kmeans.fit(scaled_data)
                    inertias.append(temp_kmeans.inertia_)
                    
                    from sklearn.cluster import AgglomerativeClustering
                    clusterer = AgglomerativeClustering(n_clusters=k)
                    labels = clusterer.fit_predict(scaled_data)
                
                # Calculate silhouette score
                silhouette = silhouette_score(scaled_data, labels)
                silhouette_scores.append(silhouette)
                
                results['cluster_analysis'][k] = {
                    'inertia': float(inertias[-1]),
                    'silhouette_score': float(silhouette)
                }
            
            # Find optimal number of clusters
            if silhouette_scores:
                optimal_k = np.argmax(silhouette_scores) + 2  # +2 because we start from k=2
                results['optimal_clusters'] = {
                    'recommended_k': int(optimal_k),
                    'best_silhouette_score': float(max(silhouette_scores)),
                    'method': 'silhouette_analysis'
                }
            
            results['inertias'] = inertias
            results['silhouette_scores'] = silhouette_scores
            
            return results
            
        except ImportError:
            return {'error': 'scikit-learn not available for clustering analysis'}
        except Exception as e:
            return {'error': str(e)}
