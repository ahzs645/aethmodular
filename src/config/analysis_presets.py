"""Analysis preset configurations for different study types"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class AnalysisPreset:
    """Analysis preset configuration"""
    name: str
    description: str
    smoothing_method: str
    smoothing_params: Dict[str, Any]
    quality_thresholds: str  # Preset name from quality_thresholds.py
    seasonal_adjustments: bool
    validation_level: str
    output_format: str
    recommended_for: List[str]


class AnalysisPresetsConfig:
    """Configuration manager for analysis presets"""
    
    PRESETS = {
        'ETAD_STANDARD': AnalysisPreset(
            name='ETAD Standard Analysis',
            description='Standard analysis protocol for ETAD project data',
            smoothing_method='ONA',
            smoothing_params={
                'delta_atn_threshold': 0.05,
                'max_window_size': 15
            },
            quality_thresholds='standard',
            seasonal_adjustments=True,
            validation_level='standard',
            output_format='research',
            recommended_for=[
                'Routine ETAD data processing',
                'Cross-method comparisons',
                'Seasonal trend analysis'
            ]
        ),
        
        'HIGH_PRECISION': AnalysisPreset(
            name='High Precision Research',
            description='Maximum precision for research applications',
            smoothing_method='CMA',
            smoothing_params={
                'window_size': 31
            },
            quality_thresholds='research',
            seasonal_adjustments=True,
            validation_level='strict',
            output_format='publication',
            recommended_for=[
                'Publication-quality analysis',
                'Detailed source apportionment',
                'Method development studies'
            ]
        ),
        
        'REAL_TIME': AnalysisPreset(
            name='Real-time Monitoring',
            description='Optimized for real-time data processing',
            smoothing_method='DEMA',
            smoothing_params={
                'alpha': 0.3
            },
            quality_thresholds='lenient',
            seasonal_adjustments=False,
            validation_level='basic',
            output_format='monitoring',
            recommended_for=[
                'Continuous monitoring',
                'Alert systems',
                'Dashboard displays'
            ]
        ),
        
        'FIELD_CAMPAIGN': AnalysisPreset(
            name='Field Campaign Analysis',
            description='Intensive analysis for short-term field campaigns',
            smoothing_method='ADAPTIVE',  # Use adaptive selection
            smoothing_params={},  # Let adaptive method choose
            quality_thresholds='strict',
            seasonal_adjustments=False,  # Campaign-specific rather than seasonal
            validation_level='comprehensive',
            output_format='campaign',
            recommended_for=[
                'Intensive field campaigns',
                'Method intercomparison studies',
                'Site characterization'
            ]
        ),
        
        'URBAN_MONITORING': AnalysisPreset(
            name='Urban Air Quality Monitoring',
            description='Configured for urban environment monitoring',
            smoothing_method='CMA',
            smoothing_params={
                'window_size': 17  # Slightly larger for urban noise
            },
            quality_thresholds='standard',
            seasonal_adjustments=True,
            validation_level='standard',
            output_format='regulatory',
            recommended_for=[
                'Urban air quality monitoring',
                'Traffic-related studies',
                'Policy support analysis'
            ]
        ),
        
        'BACKGROUND_MONITORING': AnalysisPreset(
            name='Background Site Monitoring',
            description='Optimized for clean background sites',
            smoothing_method='ONA',
            smoothing_params={
                'delta_atn_threshold': 0.08,  # Less sensitive for clean conditions
                'max_window_size': 12
            },
            quality_thresholds='strict',
            seasonal_adjustments=True,
            validation_level='comprehensive',
            output_format='research',
            recommended_for=[
                'Background monitoring sites',
                'Regional trend analysis',
                'Climate studies'
            ]
        ),
        
        'BIOMASS_BURNING': AnalysisPreset(
            name='Biomass Burning Studies',
            description='Specialized for biomass burning research',
            smoothing_method='CMA',
            smoothing_params={
                'window_size': 19  # Larger window for spike handling
            },
            quality_thresholds='standard',
            seasonal_adjustments=True,
            validation_level='specialized',
            output_format='research',
            recommended_for=[
                'Biomass burning studies',
                'Seasonal emission analysis',
                'Source attribution research'
            ]
        ),
        
        'INSTRUMENT_TESTING': AnalysisPreset(
            name='Instrument Testing & QA',
            description='For instrument testing and quality assurance',
            smoothing_method='COMPARISON',  # Compare all methods
            smoothing_params={},
            quality_thresholds='research',
            seasonal_adjustments=False,
            validation_level='comprehensive',
            output_format='technical',
            recommended_for=[
                'Instrument characterization',
                'Quality assurance protocols',
                'Method validation studies'
            ]
        )
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> AnalysisPreset:
        """Get a specific analysis preset"""
        if preset_name not in cls.PRESETS:
            available = list(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
        
        return cls.PRESETS[preset_name]
    
    @classmethod
    def get_available_presets(cls) -> List[str]:
        """Get list of available preset names"""
        return list(cls.PRESETS.keys())
    
    @classmethod
    def get_preset_descriptions(cls) -> Dict[str, str]:
        """Get brief descriptions of all presets"""
        return {
            name: preset.description 
            for name, preset in cls.PRESETS.items()
        }
    
    @classmethod
    def get_presets_by_application(cls, application: str) -> List[str]:
        """Get presets suitable for a specific application"""
        matching_presets = []
        
        for name, preset in cls.PRESETS.items():
            if any(application.lower() in rec.lower() for rec in preset.recommended_for):
                matching_presets.append(name)
        
        return matching_presets
    
    @classmethod
    def create_custom_preset(cls, 
                           name: str,
                           description: str,
                           smoothing_method: str,
                           smoothing_params: Dict[str, Any],
                           quality_thresholds: str = 'standard',
                           seasonal_adjustments: bool = True,
                           validation_level: str = 'standard',
                           output_format: str = 'research',
                           recommended_for: Optional[List[str]] = None) -> AnalysisPreset:
        """Create a custom analysis preset"""
        return AnalysisPreset(
            name=name,
            description=description,
            smoothing_method=smoothing_method,
            smoothing_params=smoothing_params,
            quality_thresholds=quality_thresholds,
            seasonal_adjustments=seasonal_adjustments,
            validation_level=validation_level,
            output_format=output_format,
            recommended_for=recommended_for or []
        )
    
    @classmethod
    def get_recommended_preset(cls, study_characteristics: Dict[str, Any]) -> str:
        """Recommend a preset based on study characteristics"""
        study_type = study_characteristics.get('study_type', '').lower()
        environment = study_characteristics.get('environment', '').lower()
        priority = study_characteristics.get('priority', '').lower()
        
        # Decision tree for preset recommendation
        if 'real-time' in priority or 'monitoring' in priority:
            return 'REAL_TIME'
        
        elif 'precision' in priority or 'publication' in priority:
            return 'HIGH_PRECISION'
        
        elif 'campaign' in study_type or 'intensive' in study_type:
            return 'FIELD_CAMPAIGN'
        
        elif 'urban' in environment:
            return 'URBAN_MONITORING'
        
        elif 'background' in environment or 'rural' in environment:
            return 'BACKGROUND_MONITORING'
        
        elif 'biomass' in study_type or 'burning' in study_type:
            return 'BIOMASS_BURNING'
        
        elif 'instrument' in study_type or 'qa' in study_type:
            return 'INSTRUMENT_TESTING'
        
        else:
            return 'ETAD_STANDARD'  # Default fallback
    
    @classmethod
    def get_preset_configuration(cls, preset_name: str) -> Dict[str, Any]:
        """Get complete configuration for a preset"""
        preset = cls.get_preset(preset_name)
        
        config = {
            'preset_info': {
                'name': preset.name,
                'description': preset.description,
                'recommended_for': preset.recommended_for
            },
            'smoothing': {
                'method': preset.smoothing_method,
                'parameters': preset.smoothing_params
            },
            'quality': {
                'thresholds_preset': preset.quality_thresholds,
                'validation_level': preset.validation_level
            },
            'analysis': {
                'seasonal_adjustments': preset.seasonal_adjustments,
                'output_format': preset.output_format
            }
        }
        
        return config
    
    @classmethod
    def validate_preset_configuration(cls, preset: AnalysisPreset) -> bool:
        """Validate that a preset configuration is valid"""
        # Check smoothing method
        valid_methods = ['ONA', 'CMA', 'DEMA', 'ADAPTIVE', 'COMPARISON']
        if preset.smoothing_method not in valid_methods:
            raise ValueError(f"Invalid smoothing method: {preset.smoothing_method}")
        
        # Check quality thresholds preset
        valid_quality_presets = ['standard', 'strict', 'lenient', 'research']
        if preset.quality_thresholds not in valid_quality_presets:
            raise ValueError(f"Invalid quality thresholds preset: {preset.quality_thresholds}")
        
        # Check validation level
        valid_validation_levels = ['basic', 'standard', 'strict', 'comprehensive', 'specialized']
        if preset.validation_level not in valid_validation_levels:
            raise ValueError(f"Invalid validation level: {preset.validation_level}")
        
        # Check output format
        valid_output_formats = ['research', 'publication', 'monitoring', 'regulatory', 'campaign', 'technical']
        if preset.output_format not in valid_output_formats:
            raise ValueError(f"Invalid output format: {preset.output_format}")
        
        return True


# Site-specific preset modifications
SITE_MODIFICATIONS = {
    'addis_ababa': {
        'description': 'Modifications for Addis Ababa urban site',
        'modifications': {
            'validation_adjustments': {
                'bc_max_realistic': 150.0,  # Higher urban levels
                'bc_max_spike': 75.0
            },
            'smoothing_adjustments': {
                'CMA': {'window_size': 19},  # Larger window for urban noise
                'DEMA': {'alpha': 0.18}      # More smoothing
            }
        }
    },
    'rural_ethiopia': {
        'description': 'Modifications for rural Ethiopian sites',
        'modifications': {
            'validation_adjustments': {
                'bc_max_realistic': 50.0,   # Lower rural levels
                'bc_max_spike': 25.0
            },
            'quality_adjustments': {
                'max_negative_values_percent': 2.0  # Stricter for clean air
            }
        }
    }
}


def apply_site_modifications(preset_config: Dict[str, Any], 
                           site_type: str) -> Dict[str, Any]:
    """Apply site-specific modifications to preset configuration"""
    if site_type not in SITE_MODIFICATIONS:
        return preset_config
    
    modifications = SITE_MODIFICATIONS[site_type]['modifications']
    modified_config = preset_config.copy()
    
    # Apply modifications
    for category, params in modifications.items():
        if category not in modified_config:
            modified_config[category] = {}
        modified_config[category].update(params)
    
    return modified_config


# Example usage configurations
EXAMPLE_CONFIGURATIONS = {
    'quick_start': {
        'description': 'Quick start configuration for new users',
        'preset': 'ETAD_STANDARD',
        'site_modifications': None,
        'seasonal_period': 'current',
        'output_level': 'summary'
    },
    'comprehensive_analysis': {
        'description': 'Comprehensive analysis with all features',
        'preset': 'HIGH_PRECISION',
        'site_modifications': 'addis_ababa',
        'seasonal_period': 'all_seasons',
        'output_level': 'detailed',
        'additional_analyses': [
            'method_comparison',
            'quality_assessment',
            'seasonal_patterns',
            'trend_analysis'
        ]
    },
    'operational_monitoring': {
        'description': 'Configuration for operational monitoring',
        'preset': 'REAL_TIME',
        'site_modifications': None,
        'seasonal_period': 'current',
        'output_level': 'operational',
        'alert_thresholds': True,
        'automated_reporting': True
    }
}
