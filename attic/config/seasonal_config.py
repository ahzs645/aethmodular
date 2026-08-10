"""Ethiopian seasonal configuration for climate-specific analysis"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import calendar


@dataclass
class SeasonDefinition:
    """Definition of an Ethiopian season"""
    name: str
    months: List[int]
    description: str
    characteristics: Dict[str, Any]


class EthiopianSeasonConfig:
    """Configuration for Ethiopian climate seasons and analysis parameters"""
    
    # Ethiopian season definitions
    SEASONS = {
        'Dry_Season': SeasonDefinition(
            name='Dry Season (Bega)',
            months=[10, 11, 12, 1, 2],
            description='Dry season with minimal rainfall and potential biomass burning',
            characteristics={
                'precipitation': 'very_low',
                'temperature': 'moderate',
                'wind_patterns': 'stable',
                'biomass_burning': 'high',
                'dust_events': 'moderate',
                'expected_bc_range': (1.0, 25.0),  # μg/m³
                'data_quality_factors': {
                    'instrument_stability': 'good',
                    'weather_interference': 'low',
                    'seasonal_maintenance': 'recommended'
                }
            }
        ),
        'Belg_Rainy': SeasonDefinition(
            name='Belg Rainy Season',
            months=[3, 4, 5],
            description='Short rainy season with intermittent precipitation',
            characteristics={
                'precipitation': 'moderate',
                'temperature': 'moderate_to_high',
                'wind_patterns': 'variable',
                'biomass_burning': 'low',
                'dust_events': 'low',
                'expected_bc_range': (0.5, 15.0),  # μg/m³
                'data_quality_factors': {
                    'instrument_stability': 'good',
                    'weather_interference': 'moderate',
                    'seasonal_maintenance': 'optional'
                }
            }
        ),
        'Kiremt_Rainy': SeasonDefinition(
            name='Kiremt Rainy Season',
            months=[6, 7, 8, 9],
            description='Main rainy season with heavy precipitation and washout',
            characteristics={
                'precipitation': 'high',
                'temperature': 'moderate',
                'wind_patterns': 'unstable',
                'biomass_burning': 'very_low',
                'dust_events': 'very_low',
                'expected_bc_range': (0.2, 8.0),  # μg/m³
                'data_quality_factors': {
                    'instrument_stability': 'challenging',
                    'weather_interference': 'high',
                    'seasonal_maintenance': 'critical'
                }
            }
        )
    }
    
    @classmethod
    def get_season_from_month(cls, month: int) -> str:
        """Get season name from month number (1-12)"""
        for season_key, season in cls.SEASONS.items():
            if month in season.months:
                return season_key
        raise ValueError(f"Invalid month: {month}")
    
    @classmethod
    def get_season_from_date(cls, date: datetime) -> str:
        """Get season name from datetime object"""
        return cls.get_season_from_month(date.month)
    
    @classmethod
    def get_season_info(cls, season_name: str) -> SeasonDefinition:
        """Get detailed information about a season"""
        if season_name not in cls.SEASONS:
            available = list(cls.SEASONS.keys())
            raise ValueError(f"Unknown season: {season_name}. Available: {available}")
        return cls.SEASONS[season_name]
    
    @classmethod
    def get_all_seasons(cls) -> Dict[str, SeasonDefinition]:
        """Get all season definitions"""
        return cls.SEASONS.copy()
    
    @classmethod
    def get_season_sequence(cls, start_month: int, duration_months: int) -> List[str]:
        """Get sequence of seasons for a given period"""
        seasons = []
        for i in range(duration_months):
            month = ((start_month - 1 + i) % 12) + 1
            season = cls.get_season_from_month(month)
            if not seasons or seasons[-1] != season:
                seasons.append(season)
        return seasons
    
    @classmethod
    def get_seasonal_analysis_parameters(cls, season_name: str) -> Dict[str, Any]:
        """Get analysis parameters specific to a season"""
        season_info = cls.get_season_info(season_name)
        
        # Base parameters adjusted for seasonal characteristics
        params = {
            'expected_bc_range': season_info.characteristics['expected_bc_range'],
            'smoothing_recommendations': cls._get_seasonal_smoothing_params(season_name),
            'quality_adjustments': cls._get_seasonal_quality_adjustments(season_name),
            'validation_thresholds': cls._get_seasonal_validation_thresholds(season_name)
        }
        
        return params
    
    @classmethod
    def _get_seasonal_smoothing_params(cls, season_name: str) -> Dict[str, Any]:
        """Get season-specific smoothing parameter recommendations"""
        season_info = cls.get_season_info(season_name)
        
        if season_name == 'Dry_Season':
            # Higher noise due to biomass burning and dust
            return {
                'recommended_method': 'CMA',  # Better for handling spikes
                'ONA': {'delta_atn_threshold': 0.04},  # More sensitive
                'CMA': {'window_size': 17},             # Larger window
                'DEMA': {'alpha': 0.15}                # More smoothing
            }
        elif season_name == 'Belg_Rainy':
            # Moderate conditions
            return {
                'recommended_method': 'ONA',  # Adaptive to changing conditions
                'ONA': {'delta_atn_threshold': 0.05},  # Standard
                'CMA': {'window_size': 15},
                'DEMA': {'alpha': 0.2}
            }
        elif season_name == 'Kiremt_Rainy':
            # Challenging conditions with weather interference
            return {
                'recommended_method': 'DEMA',  # Less affected by gaps
                'ONA': {'delta_atn_threshold': 0.06},  # Less sensitive
                'CMA': {'window_size': 19},             # Larger window for stability
                'DEMA': {'alpha': 0.18}                # More robust smoothing
            }
        
        return {}
    
    @classmethod
    def _get_seasonal_quality_adjustments(cls, season_name: str) -> Dict[str, Any]:
        """Get season-specific quality threshold adjustments"""
        if season_name == 'Dry_Season':
            return {
                'completeness_tolerance': 1.0,      # Standard tolerance
                'negative_values_tolerance': 1.2,   # Slightly more due to dust
                'spike_tolerance': 1.5              # More spikes expected
            }
        elif season_name == 'Belg_Rainy':
            return {
                'completeness_tolerance': 1.1,      # Slightly more tolerant
                'negative_values_tolerance': 1.0,   # Standard
                'spike_tolerance': 1.0              # Standard
            }
        elif season_name == 'Kiremt_Rainy':
            return {
                'completeness_tolerance': 1.5,      # Much more tolerant
                'negative_values_tolerance': 1.3,   # More tolerant due to washout
                'spike_tolerance': 0.8              # Fewer spikes expected
            }
        
        return {}
    
    @classmethod
    def _get_seasonal_validation_thresholds(cls, season_name: str) -> Dict[str, Any]:
        """Get season-specific validation thresholds"""
        season_info = cls.get_season_info(season_name)
        bc_min, bc_max = season_info.characteristics['expected_bc_range']
        
        return {
            'bc_max_realistic': bc_max * 2.0,      # Allow 2x expected max
            'bc_max_spike': bc_max * 1.5,          # Spike threshold
            'bc_max_rate_change': bc_max * 0.2,    # Rate change threshold
        }
    
    @classmethod
    def get_transition_periods(cls) -> List[Dict[str, Any]]:
        """Get information about seasonal transition periods"""
        transitions = [
            {
                'name': 'Dry to Belg transition',
                'months': [2, 3],
                'characteristics': 'Variable conditions, onset of rains',
                'data_considerations': 'Potential instrument stress from changing humidity'
            },
            {
                'name': 'Belg to Kiremt transition', 
                'months': [5, 6],
                'characteristics': 'Intensifying rains, increasing cloud cover',
                'data_considerations': 'Increasing data gaps due to weather'
            },
            {
                'name': 'Kiremt to Dry transition',
                'months': [9, 10],
                'characteristics': 'Decreasing rains, biomass burning onset',
                'data_considerations': 'Changing aerosol composition and concentrations'
            }
        ]
        
        return transitions
    
    @classmethod
    def get_monthly_characteristics(cls) -> Dict[int, Dict[str, Any]]:
        """Get detailed characteristics for each month"""
        monthly_chars = {}
        
        for month in range(1, 13):
            season = cls.get_season_from_month(month)
            season_info = cls.get_season_info(season)
            month_name = calendar.month_name[month]
            
            # Month-specific adjustments within seasons
            month_adjustments = cls._get_month_specific_adjustments(month, season)
            
            monthly_chars[month] = {
                'month_name': month_name,
                'season': season,
                'season_description': season_info.description,
                'base_characteristics': season_info.characteristics,
                'month_adjustments': month_adjustments
            }
        
        return monthly_chars
    
    @classmethod
    def _get_month_specific_adjustments(cls, month: int, season: str) -> Dict[str, Any]:
        """Get month-specific adjustments within seasons"""
        adjustments = {}
        
        if season == 'Dry_Season':
            if month in [12, 1]:  # Peak dry season
                adjustments = {
                    'biomass_burning_intensity': 'peak',
                    'bc_concentration_factor': 1.3,
                    'dust_events_probability': 'high'
                }
            elif month in [10, 2]:  # Transition months
                adjustments = {
                    'biomass_burning_intensity': 'moderate',
                    'bc_concentration_factor': 1.1,
                    'dust_events_probability': 'moderate'
                }
            elif month == 11:  # Peak burning season
                adjustments = {
                    'biomass_burning_intensity': 'maximum',
                    'bc_concentration_factor': 1.5,
                    'dust_events_probability': 'high'
                }
        
        elif season == 'Belg_Rainy':
            if month == 4:  # Peak Belg
                adjustments = {
                    'precipitation_intensity': 'peak',
                    'bc_concentration_factor': 0.7,
                    'washout_probability': 'high'
                }
            else:  # Transition months
                adjustments = {
                    'precipitation_intensity': 'moderate',
                    'bc_concentration_factor': 0.9,
                    'washout_probability': 'moderate'
                }
        
        elif season == 'Kiremt_Rainy':
            if month in [7, 8]:  # Peak Kiremt
                adjustments = {
                    'precipitation_intensity': 'maximum',
                    'bc_concentration_factor': 0.4,
                    'washout_probability': 'maximum',
                    'instrument_challenges': 'highest'
                }
            else:  # Early/late Kiremt
                adjustments = {
                    'precipitation_intensity': 'high',
                    'bc_concentration_factor': 0.6,
                    'washout_probability': 'high',
                    'instrument_challenges': 'high'
                }
        
        return adjustments


# Default seasonal analysis configuration
DEFAULT_SEASONAL_CONFIG = {
    'use_seasonal_adjustments': True,
    'transition_period_handling': 'interpolate',  # 'interpolate', 'separate', 'ignore'
    'seasonal_comparison_metrics': [
        'mean_bc_concentration',
        'median_bc_concentration', 
        'bc_variability',
        'data_completeness',
        'quality_distribution'
    ],
    'seasonal_plotting_preferences': {
        'Dry_Season': {'color': '#D2691E', 'marker': 'o'},      # Orange-brown
        'Belg_Rainy': {'color': '#9ACD32', 'marker': 's'},      # Yellow-green  
        'Kiremt_Rainy': {'color': '#4682B4', 'marker': '^'}     # Steel blue
    }
}


def get_analysis_recommendations_by_season(season: str) -> Dict[str, Any]:
    """Get comprehensive analysis recommendations for a season"""
    season_info = EthiopianSeasonConfig.get_season_info(season)
    params = EthiopianSeasonConfig.get_seasonal_analysis_parameters(season)
    
    recommendations = {
        'season': season,
        'description': season_info.description,
        'data_collection': {
            'expected_challenges': season_info.characteristics['data_quality_factors'],
            'maintenance_priority': season_info.characteristics['data_quality_factors']['seasonal_maintenance']
        },
        'analysis_parameters': params,
        'interpretation_notes': _get_interpretation_notes(season),
        'quality_control': _get_seasonal_qc_recommendations(season)
    }
    
    return recommendations


def _get_interpretation_notes(season: str) -> List[str]:
    """Get interpretation notes for seasonal analysis"""
    if season == 'Dry_Season':
        return [
            "Higher BC concentrations expected due to biomass burning",
            "Watch for dust interference in measurements",
            "Diurnal patterns may be more pronounced",
            "Consider regional burning patterns in interpretation"
        ]
    elif season == 'Belg_Rainy':
        return [
            "Moderate BC levels with intermittent washout",
            "Variable meteorological conditions",
            "Transition period dynamics may affect trends",
            "Good period for instrument calibration checks"
        ]
    elif season == 'Kiremt_Rainy':
        return [
            "Lowest BC concentrations due to washout",
            "Expect more data gaps due to weather",
            "Lower signal-to-noise ratio challenges",
            "Consider precipitation data for interpretation"
        ]
    return []


def _get_seasonal_qc_recommendations(season: str) -> List[str]:
    """Get quality control recommendations for each season"""
    if season == 'Dry_Season':
        return [
            "Monitor for dust contamination on inlet",
            "Check for biomass burning spike artifacts",
            "Verify flow rate stability in dusty conditions",
            "Consider more frequent filter changes"
        ]
    elif season == 'Belg_Rainy':
        return [
            "Standard QC procedures sufficient",
            "Monitor humidity effects on instrument",
            "Check inlet heating system if available",
            "Verify data continuity during rain events"
        ]
    elif season == 'Kiremt_Rainy':
        return [
            "Critical: Monitor instrument environmental conditions",
            "Check for water ingress in systems",
            "Verify inlet heating and drying systems",
            "Implement more aggressive outlier detection",
            "Plan for extended maintenance windows"
        ]
    return []
