"""Quality assessment thresholds and classification parameters"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class QualityLevel(Enum):
    """Data quality levels"""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    MODERATE = "Moderate"
    POOR = "Poor"


@dataclass
class CompletenessThresholds:
    """Thresholds for data completeness classification"""
    excellent_max_missing: int = 10      # ≤ 10 missing minutes per period
    good_max_missing: int = 60           # ≤ 60 missing minutes per period
    moderate_max_missing: int = 240      # ≤ 240 missing minutes per period
    # Poor: > 240 missing minutes per period
    
    def classify_period(self, missing_minutes: int) -> QualityLevel:
        """Classify a period based on missing minutes"""
        if missing_minutes <= self.excellent_max_missing:
            return QualityLevel.EXCELLENT
        elif missing_minutes <= self.good_max_missing:
            return QualityLevel.GOOD
        elif missing_minutes <= self.moderate_max_missing:
            return QualityLevel.MODERATE
        else:
            return QualityLevel.POOR


@dataclass
class QualityFactorThresholds:
    """Thresholds for advanced quality factors"""
    max_consecutive_missing_good: int = 30        # Max consecutive missing for Good quality
    gap_penalty_threshold: int = 5                # Number of gaps before penalty
    large_gap_threshold_minutes: int = 120        # Duration considered a large gap
    weekend_tolerance_factor: float = 1.5         # Weekend tolerance multiplier
    
    # Column-specific thresholds
    max_negative_values_percent: float = 5.0      # Max % negative values
    max_zero_values_percent: float = 10.0         # Max % zero values
    min_correlation_between_wavelengths: float = 0.7  # Min correlation between wavelengths


@dataclass  
class ValidationThresholds:
    """Thresholds for data validation"""
    # BC (Black Carbon) value thresholds
    bc_min_realistic: float = -0.5        # Minimum realistic BC value (ng/m³)
    bc_max_realistic: float = 100.0       # Maximum realistic BC value (ng/m³)
    bc_max_spike: float = 50.0             # Maximum allowable spike
    
    # ATN (Attenuation) value thresholds  
    atn_min_realistic: float = 0.0        # Minimum realistic ATN
    atn_max_realistic: float = 200.0      # Maximum realistic ATN
    atn_max_jump: float = 20.0             # Maximum allowable ATN jump
    
    # Rate of change thresholds
    bc_max_rate_change: float = 10.0      # ng/m³ per minute
    atn_max_rate_change: float = 2.0      # ATN units per minute


class QualityThresholdsConfig:
    """Configuration manager for quality thresholds"""
    
    # Preset threshold configurations
    PRESETS = {
        'standard': {
            'completeness': CompletenessThresholds(),
            'quality_factors': QualityFactorThresholds(),
            'validation': ValidationThresholds()
        },
        'strict': {
            'completeness': CompletenessThresholds(
                excellent_max_missing=5,
                good_max_missing=30,
                moderate_max_missing=120
            ),
            'quality_factors': QualityFactorThresholds(
                max_consecutive_missing_good=15,
                gap_penalty_threshold=3,
                max_negative_values_percent=2.0,
                max_zero_values_percent=5.0
            ),
            'validation': ValidationThresholds(
                bc_max_realistic=50.0,
                bc_max_spike=25.0,
                atn_max_jump=10.0
            )
        },
        'lenient': {
            'completeness': CompletenessThresholds(
                excellent_max_missing=20,
                good_max_missing=120,
                moderate_max_missing=480
            ),
            'quality_factors': QualityFactorThresholds(
                max_consecutive_missing_good=60,
                gap_penalty_threshold=10,
                max_negative_values_percent=10.0,
                max_zero_values_percent=20.0
            ),
            'validation': ValidationThresholds(
                bc_max_realistic=200.0,
                bc_max_spike=100.0,
                atn_max_jump=40.0
            )
        },
        'research': {
            # Very strict for research applications
            'completeness': CompletenessThresholds(
                excellent_max_missing=2,
                good_max_missing=10,
                moderate_max_missing=30
            ),
            'quality_factors': QualityFactorThresholds(
                max_consecutive_missing_good=5,
                gap_penalty_threshold=2,
                max_negative_values_percent=1.0,
                max_zero_values_percent=2.0,
                min_correlation_between_wavelengths=0.85
            ),
            'validation': ValidationThresholds(
                bc_max_realistic=30.0,
                bc_max_spike=15.0,
                atn_max_jump=5.0,
                bc_max_rate_change=5.0,
                atn_max_rate_change=1.0
            )
        }
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> Dict[str, Any]:
        """Get a preset threshold configuration"""
        if preset_name not in cls.PRESETS:
            available = list(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
        
        return cls.PRESETS[preset_name]
    
    @classmethod
    def get_available_presets(cls) -> list:
        """Get list of available preset names"""
        return list(cls.PRESETS.keys())
    
    @classmethod
    def get_preset_descriptions(cls) -> Dict[str, str]:
        """Get descriptions of available presets"""
        return {
            'standard': 'Balanced thresholds suitable for most operational monitoring',
            'strict': 'Stricter thresholds for high-quality data requirements',
            'lenient': 'More forgiving thresholds for challenging environments',
            'research': 'Very strict thresholds for research-grade data quality'
        }
    
    @classmethod
    def create_custom_thresholds(cls,
                               completeness: Optional[Dict[str, Any]] = None,
                               quality_factors: Optional[Dict[str, Any]] = None,
                               validation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create custom threshold configuration"""
        config = {}
        
        if completeness:
            config['completeness'] = CompletenessThresholds(**completeness)
        
        if quality_factors:
            config['quality_factors'] = QualityFactorThresholds(**quality_factors)
        
        if validation:
            config['validation'] = ValidationThresholds(**validation)
        
        return config
    
    @classmethod
    def get_site_specific_adjustments(cls, site_type: str) -> Dict[str, Any]:
        """Get site-specific threshold adjustments"""
        adjustments = {
            'urban': {
                'validation': {
                    'bc_max_realistic': 150.0,  # Higher BC expected in urban areas
                    'bc_max_spike': 75.0
                }
            },
            'rural': {
                'validation': {
                    'bc_max_realistic': 50.0,   # Lower BC expected in rural areas
                    'bc_max_spike': 25.0
                },
                'quality_factors': {
                    'max_negative_values_percent': 2.0  # More strict for cleaner air
                }
            },
            'industrial': {
                'validation': {
                    'bc_max_realistic': 300.0,  # Very high BC possible near industry
                    'bc_max_spike': 150.0,
                    'bc_max_rate_change': 20.0
                },
                'completeness': {
                    'excellent_max_missing': 15,  # More tolerance due to harsh conditions
                    'good_max_missing': 90
                }
            },
            'background': {
                'validation': {
                    'bc_max_realistic': 20.0,   # Very low BC expected
                    'bc_max_spike': 10.0
                },
                'quality_factors': {
                    'max_negative_values_percent': 1.0,
                    'min_correlation_between_wavelengths': 0.8
                }
            }
        }
        
        return adjustments.get(site_type, {})


# Seasonal threshold adjustments for Ethiopian climate
SEASONAL_ADJUSTMENTS = {
    'Dry_Season': {  # Bega (October-February)
        'validation': {
            'bc_max_realistic': 120.0,      # Higher due to biomass burning
            'bc_max_spike': 60.0
        },
        'quality_factors': {
            'max_negative_values_percent': 3.0  # Dust can affect measurements
        }
    },
    'Belg_Rainy': {  # March-May
        'validation': {
            'bc_max_realistic': 80.0,       # Moderate levels
            'bc_max_spike': 40.0
        }
    },
    'Kiremt_Rainy': {  # June-September  
        'validation': {
            'bc_max_realistic': 60.0,       # Lowest due to washout
            'bc_max_spike': 30.0
        },
        'completeness': {
            'excellent_max_missing': 15,    # More tolerance due to weather
            'good_max_missing': 90
        }
    }
}


def get_season_adjusted_thresholds(base_thresholds: Dict[str, Any], 
                                 season: str) -> Dict[str, Any]:
    """Apply seasonal adjustments to base thresholds"""
    if season not in SEASONAL_ADJUSTMENTS:
        return base_thresholds
    
    adjustments = SEASONAL_ADJUSTMENTS[season]
    adjusted = base_thresholds.copy()
    
    # Apply adjustments
    for category, params in adjustments.items():
        if category in adjusted:
            for param, value in params.items():
                setattr(adjusted[category], param, value)
    
    return adjusted


def validate_threshold_consistency(thresholds: Dict[str, Any]) -> bool:
    """Validate that thresholds are internally consistent"""
    completeness = thresholds.get('completeness')
    if completeness:
        # Check that thresholds are in ascending order
        if not (completeness.excellent_max_missing <= 
                completeness.good_max_missing <= 
                completeness.moderate_max_missing):
            raise ValueError("Completeness thresholds must be in ascending order")
    
    validation = thresholds.get('validation')
    if validation:
        # Check that min < max for realistic values
        if validation.bc_min_realistic >= validation.bc_max_realistic:
            raise ValueError("BC min must be less than max")
        
        if validation.atn_min_realistic >= validation.atn_max_realistic:
            raise ValueError("ATN min must be less than max")
    
    return True
