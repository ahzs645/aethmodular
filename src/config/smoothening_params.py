"""Smoothening algorithm parameters configuration"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ONAParameters:
    """ONA (Optimized Noise-reduction Algorithm) parameters"""
    delta_atn_threshold: float = 0.05
    min_window_size: int = 1
    max_window_size: int = 15
    high_variability_window: int = 5
    low_variability_window: int = 15
    weight_decay_factor: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'delta_atn_threshold': self.delta_atn_threshold,
            'min_window_size': self.min_window_size,
            'max_window_size': self.max_window_size,
            'high_variability_window': self.high_variability_window,
            'low_variability_window': self.low_variability_window,
            'weight_decay_factor': self.weight_decay_factor
        }


@dataclass
class CMAParameters:
    """CMA (Centered Moving Average) parameters"""
    window_size: int = 15
    min_valid_points: int = 3
    edge_handling: str = 'symmetric'  # 'symmetric', 'forward', 'backward'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'window_size': self.window_size,
            'min_valid_points': self.min_valid_points,
            'edge_handling': self.edge_handling
        }


@dataclass
class DEMAParameters:
    """DEMA (Double Exponentially Weighted Moving Average) parameters"""
    alpha: float = 0.2
    min_periods: int = 3
    adjust_for_missing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alpha': self.alpha,
            'min_periods': self.min_periods,
            'adjust_for_missing': self.adjust_for_missing
        }


class SmoothingParametersConfig:
    """Configuration manager for smoothening parameters"""
    
    # Preset configurations for different scenarios
    PRESETS = {
        'default': {
            'ONA': ONAParameters(),
            'CMA': CMAParameters(),
            'DEMA': DEMAParameters()
        },
        'high_noise': {
            'ONA': ONAParameters(delta_atn_threshold=0.03, max_window_size=20),
            'CMA': CMAParameters(window_size=21),
            'DEMA': DEMAParameters(alpha=0.15)
        },
        'low_noise': {
            'ONA': ONAParameters(delta_atn_threshold=0.08, max_window_size=10),
            'CMA': CMAParameters(window_size=9),
            'DEMA': DEMAParameters(alpha=0.3)
        },
        'real_time': {
            'ONA': ONAParameters(max_window_size=8),
            'CMA': CMAParameters(window_size=7),
            'DEMA': DEMAParameters(alpha=0.4)
        },
        'high_precision': {
            'ONA': ONAParameters(delta_atn_threshold=0.02, max_window_size=25),
            'CMA': CMAParameters(window_size=31),
            'DEMA': DEMAParameters(alpha=0.1)
        }
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> Dict[str, Any]:
        """Get a preset configuration"""
        if preset_name not in cls.PRESETS:
            available = list(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
        
        preset = cls.PRESETS[preset_name]
        return {
            method: params.to_dict() 
            for method, params in preset.items()
        }
    
    @classmethod
    def get_available_presets(cls) -> list:
        """Get list of available presets"""
        return list(cls.PRESETS.keys())
    
    @classmethod
    def get_preset_descriptions(cls) -> Dict[str, str]:
        """Get descriptions of available presets"""
        return {
            'default': 'Balanced parameters suitable for most applications',
            'high_noise': 'More aggressive smoothing for noisy data',
            'low_noise': 'Lighter smoothing to preserve fine details',
            'real_time': 'Optimized for real-time processing with minimal lag',
            'high_precision': 'Maximum smoothing for highest precision applications'
        }
    
    @classmethod
    def create_custom_parameters(cls, 
                                ona_params: Optional[Dict[str, Any]] = None,
                                cma_params: Optional[Dict[str, Any]] = None,
                                dema_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create custom parameter configuration"""
        config = {}
        
        if ona_params:
            config['ONA'] = ONAParameters(**ona_params).to_dict()
        
        if cma_params:
            config['CMA'] = CMAParameters(**cma_params).to_dict()
        
        if dema_params:
            config['DEMA'] = DEMAParameters(**dema_params).to_dict()
        
        return config
    
    @classmethod
    def validate_parameters(cls, method: str, params: Dict[str, Any]) -> bool:
        """Validate parameters for a specific method"""
        if method == 'ONA':
            return cls._validate_ona_parameters(params)
        elif method == 'CMA':
            return cls._validate_cma_parameters(params)
        elif method == 'DEMA':
            return cls._validate_dema_parameters(params)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @classmethod
    def _validate_ona_parameters(cls, params: Dict[str, Any]) -> bool:
        """Validate ONA parameters"""
        required = ['delta_atn_threshold']
        
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required ONA parameter: {param}")
        
        # Validate ranges
        if not 0 < params['delta_atn_threshold'] < 1:
            raise ValueError("delta_atn_threshold must be between 0 and 1")
        
        if 'min_window_size' in params and params['min_window_size'] < 1:
            raise ValueError("min_window_size must be >= 1")
        
        if 'max_window_size' in params and params['max_window_size'] < 1:
            raise ValueError("max_window_size must be >= 1")
        
        return True
    
    @classmethod
    def _validate_cma_parameters(cls, params: Dict[str, Any]) -> bool:
        """Validate CMA parameters"""
        required = ['window_size']
        
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required CMA parameter: {param}")
        
        # Validate ranges
        if params['window_size'] < 3:
            raise ValueError("window_size must be >= 3")
        
        if params['window_size'] % 2 == 0:
            raise ValueError("window_size should be odd for symmetric centering")
        
        return True
    
    @classmethod
    def _validate_dema_parameters(cls, params: Dict[str, Any]) -> bool:
        """Validate DEMA parameters"""
        required = ['alpha']
        
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required DEMA parameter: {param}")
        
        # Validate ranges
        if not 0 < params['alpha'] < 1:
            raise ValueError("alpha must be between 0 and 1")
        
        return True


# Wavelength-specific parameter adjustments
WAVELENGTH_ADJUSTMENTS = {
    'UV': {
        'ONA': {'delta_atn_threshold': 0.04},  # UV typically noisier
        'CMA': {'window_size': 17},             # Slightly larger window
        'DEMA': {'alpha': 0.18}                # More smoothing
    },
    'Blue': {
        'ONA': {'delta_atn_threshold': 0.045},
        'CMA': {'window_size': 15},
        'DEMA': {'alpha': 0.19}
    },
    'Green': {
        'ONA': {'delta_atn_threshold': 0.05},  # Standard
        'CMA': {'window_size': 15},
        'DEMA': {'alpha': 0.2}
    },
    'Red': {
        'ONA': {'delta_atn_threshold': 0.055},
        'CMA': {'window_size': 13},
        'DEMA': {'alpha': 0.21}
    },
    'IR': {
        'ONA': {'delta_atn_threshold': 0.05},  # Most stable wavelength
        'CMA': {'window_size': 15},
        'DEMA': {'alpha': 0.2}
    }
}


def get_wavelength_adjusted_parameters(method: str, wavelength: str, 
                                     base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Get parameters adjusted for specific wavelength"""
    if wavelength not in WAVELENGTH_ADJUSTMENTS:
        return base_params
    
    adjustments = WAVELENGTH_ADJUSTMENTS[wavelength].get(method, {})
    
    # Apply adjustments to base parameters
    adjusted_params = base_params.copy()
    adjusted_params.update(adjustments)
    
    return adjusted_params
