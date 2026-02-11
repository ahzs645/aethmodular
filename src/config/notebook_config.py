# src/config/notebook_config.py
"""
Enhanced configuration management for notebook usage
Centralizes all configuration parameters and file paths
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

from src.config.project_paths import get_data_root


@dataclass
class NotebookConfig:
    """Configuration container for notebook analysis"""
    
    # Site and analysis parameters
    site_code: str = 'ETAD'
    wavelength: str = 'Red'  # Options: 'Red', 'Blue', 'Green', 'UV', 'IR'
    quality_threshold: int = 10  # Maximum missing minutes for "excellent" quality
    output_format: str = "jpl"  # 'jpl' or 'standard' format
    
    # File paths - will be set during initialization
    aethalometer_files: Dict[str, str] = field(default_factory=dict)
    ftir_db_path: str = ""
    output_dir: str = "outputs"
    
    # Analysis parameters
    min_samples_for_analysis: int = 30
    confidence_level: float = 0.95
    outlier_threshold: float = 3.0
    
    # Plotting parameters
    figure_size: tuple = (12, 8)
    font_size: int = 10
    dpi: int = 300
    
    def __post_init__(self):
        """Initialize after dataclass creation"""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

class ConfigurationManager:
    """Manages configuration for different environments"""
    
    @staticmethod
    def create_etad_config(base_data_path: Optional[str] = None) -> NotebookConfig:
        """
        Create ETAD-specific configuration
        
        Parameters:
        -----------
        base_data_path : str, optional
            Base path to data directory. If None, uses default ETAD paths.
        """
        
        if base_data_path is None:
            # Use environment-driven data root; defaults to repo-relative research data.
            base_data_path = str(get_data_root())
        
        config = NotebookConfig()
        
        # Set ETAD-specific file paths
        root = Path(base_data_path)
        config.aethalometer_files = {
            'pkl_data': str(root / "processed_sites" / "df_Addis_Ababa_9am_resampled.pkl"),
            'csv_data': str(root / "Filter Data" / "unified_filter_dataset.csv"),
        }

        config.ftir_db_path = str(root / "spartan_ftir_hips.db")
        
        return config
    
    @staticmethod
    def create_custom_config(
        site_code: str,
        aethalometer_files: Dict[str, str],
        ftir_db_path: str,
        **kwargs
    ) -> NotebookConfig:
        """
        Create custom configuration
        
        Parameters:
        -----------
        site_code : str
            Site identifier
        aethalometer_files : dict
            Dictionary of aethalometer file paths
        ftir_db_path : str
            Path to FTIR database
        **kwargs
            Additional configuration parameters
        """
        
        config = NotebookConfig(
            site_code=site_code,
            aethalometer_files=aethalometer_files,
            ftir_db_path=ftir_db_path,
            **kwargs
        )
        
        return config
    
    @staticmethod
    def validate_config(config: NotebookConfig) -> Dict[str, bool]:
        """
        Validate configuration paths and parameters
        
        Returns:
        --------
        dict
            Validation results
        """
        
        results = {
            'config_valid': True,
            'aethalometer_files_exist': {},
            'ftir_db_exists': False,
            'output_dir_writable': False
        }
        
        # Check aethalometer files
        for name, path in config.aethalometer_files.items():
            exists = os.path.exists(path)
            results['aethalometer_files_exist'][name] = exists
            if not exists:
                results['config_valid'] = False
        
        # Check FTIR database
        results['ftir_db_exists'] = os.path.exists(config.ftir_db_path)
        if not results['ftir_db_exists']:
            results['config_valid'] = False
        
        # Check output directory
        try:
            test_file = os.path.join(config.output_dir, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            results['output_dir_writable'] = True
        except (OSError, PermissionError):
            results['output_dir_writable'] = False
            results['config_valid'] = False
        
        return results
    
    @staticmethod
    def print_config_summary(config: NotebookConfig):
        """Print configuration summary"""
        
        print("📊 Configuration Summary:")
        print(f"   Site: {config.site_code}")
        print(f"   Wavelength: {config.wavelength}")
        print(f"   Output format: {config.output_format}")
        print(f"   Quality threshold: {config.quality_threshold} minutes")
        print(f"   Output directory: {config.output_dir}")
        
        print(f"\n📁 File paths:")
        for name, path in config.aethalometer_files.items():
            status = "✅" if os.path.exists(path) else "❌"
            print(f"   {name}: {status} {Path(path).name}")
        
        ftir_status = "✅" if os.path.exists(config.ftir_db_path) else "❌"
        print(f"   FTIR DB: {ftir_status} {Path(config.ftir_db_path).name}")

# Example usage functions
def get_default_etad_config() -> NotebookConfig:
    """Get default ETAD configuration"""
    return ConfigurationManager.create_etad_config()

def get_validated_config(config: NotebookConfig) -> NotebookConfig:
    """Get validated configuration with error reporting"""
    
    validation = ConfigurationManager.validate_config(config)
    
    if not validation['config_valid']:
        print("⚠️ Configuration validation issues found:")
        
        for name, exists in validation['aethalometer_files_exist'].items():
            if not exists:
                print(f"   ❌ Aethalometer file not found: {name}")
        
        if not validation['ftir_db_exists']:
            print(f"   ❌ FTIR database not found")
        
        if not validation['output_dir_writable']:
            print(f"   ❌ Output directory not writable")
    
    ConfigurationManager.print_config_summary(config)
    
    return config
