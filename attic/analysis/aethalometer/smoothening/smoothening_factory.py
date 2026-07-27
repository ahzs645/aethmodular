"""Factory pattern for smoothening algorithm selection"""

from typing import Dict, Any, Optional, Union
from .ona_smoothening import ONASmoothing
from .cma_smoothening import CMASmoothing
from .dema_smoothening import DEMASmoothing


class SmoothingFactory:
    """
    Factory class for creating smoothening algorithm instances
    
    Provides a centralized way to create and configure smoothening algorithms
    based on method name and parameters.
    """
    
    AVAILABLE_METHODS = {
        'ONA': ONASmoothing,
        'CMA': CMASmoothing,
        'DEMA': DEMASmoothing
    }
    
    @classmethod
    def create_smoother(cls, method: str, **kwargs) -> Union[ONASmoothing, CMASmoothing, DEMASmoothing]:
        """
        Create a smoothening algorithm instance
        
        Parameters:
        -----------
        method : str
            Smoothening method name ('ONA', 'CMA', 'DEMA')
        **kwargs : dict
            Method-specific parameters
            
        Returns:
        --------
        BaseSmoothing subclass instance
        
        Examples:
        ---------
        >>> ona = SmoothingFactory.create_smoother('ONA', delta_atn_threshold=0.05)
        >>> cma = SmoothingFactory.create_smoother('CMA', window_size=15)
        >>> dema = SmoothingFactory.create_smoother('DEMA', alpha=0.2)
        """
        method = method.upper()
        
        if method not in cls.AVAILABLE_METHODS:
            available = list(cls.AVAILABLE_METHODS.keys())
            raise ValueError(f"Unknown smoothening method: {method}. Available: {available}")
        
        smoother_class = cls.AVAILABLE_METHODS[method]
        return smoother_class(**kwargs)
    
    @classmethod
    def get_available_methods(cls) -> list:
        """Get list of available smoothening methods"""
        return list(cls.AVAILABLE_METHODS.keys())
    
    @classmethod
    def get_method_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available smoothening methods
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Method information including parameters and descriptions
        """
        return {
            'ONA': {
                'class': ONASmoothing,
                'description': 'Optimized Noise-reduction Algorithm with adaptive time-averaging',
                'parameters': {
                    'delta_atn_threshold': {
                        'type': 'float',
                        'default': 0.05,
                        'description': 'Threshold for ΔATN to determine window size'
                    }
                },
                'best_for': 'Variable conditions, automatic adaptation to data characteristics'
            },
            'CMA': {
                'class': CMASmoothing,
                'description': 'Centered Moving Average for symmetric noise reduction',
                'parameters': {
                    'window_size': {
                        'type': 'int',
                        'default': 15,
                        'description': 'Size of the moving average window'
                    }
                },
                'best_for': 'Preserving temporal patterns while reducing noise'
            },
            'DEMA': {
                'class': DEMASmoothing,
                'description': 'Double Exponentially Weighted Moving Average with minimal lag',
                'parameters': {
                    'alpha': {
                        'type': 'float',
                        'default': 0.2,
                        'description': 'Smoothing factor (0 < alpha < 1)'
                    }
                },
                'best_for': 'Real-time applications where lag minimization is critical'
            }
        }
    
    @classmethod
    def recommend_method(cls, data_characteristics: Dict[str, Any]) -> str:
        """
        Recommend a smoothening method based on data characteristics
        
        Parameters:
        -----------
        data_characteristics : Dict[str, Any]
            Data characteristics such as noise level, variability, etc.
            
        Returns:
        --------
        str
            Recommended method name
        """
        # Simple rule-based recommendation
        noise_level = data_characteristics.get('noise_level', 'medium')
        variability = data_characteristics.get('variability', 'medium')
        real_time = data_characteristics.get('real_time_required', False)
        
        if real_time:
            return 'DEMA'
        elif variability == 'high':
            return 'ONA'
        elif noise_level == 'high':
            return 'CMA'
        else:
            return 'ONA'  # Default to adaptive method
