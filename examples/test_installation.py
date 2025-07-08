"""
Installation Test Script

This script tests that the modular structure is properly set up
and can be imported without errors.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all main modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test config imports
        from config.settings import QUALITY_THRESHOLDS, MIN_SAMPLES_FOR_ANALYSIS
        print("✓ Config modules imported successfully")
        
        # Test core imports
        from core.base import BaseAnalyzer, BaseLoader
        from core.exceptions import AnalysisError, DataValidationError
        print("✓ Core modules imported successfully")
        
        # Test data imports
        from data.processors.validation import validate_columns_exist
        print("✓ Data processor modules imported successfully")
        
        # Test analysis imports
        from analysis.statistics.descriptive import calculate_basic_statistics
        from analysis.correlations.pearson import calculate_pearson_correlation
        print("✓ Analysis modules imported successfully")
        
        # Test utility imports
        from utils.file_io import save_results_to_json
        print("✓ Utility modules imported successfully")
        
        print("\\n🎉 All imports successful! The modular structure is ready to use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with sample data"""
    print("\\nTesting basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        data = pd.DataFrame({
            'fabs': [10, 20, 30],
            'ec_ftir': [2, 4, 6],
            'oc_ftir': [8, 16, 24]
        })
        
        # Test validation
        from data.processors.validation import validate_columns_exist, get_valid_data_mask
        validate_columns_exist(data, ['fabs', 'ec_ftir'])
        mask = get_valid_data_mask(data, ['fabs', 'ec_ftir'])
        
        # Test statistics
        from analysis.statistics.descriptive import calculate_basic_statistics
        mac_values = data['fabs'] / data['ec_ftir']
        stats = calculate_basic_statistics(mac_values, 'mac')
        
        # Test correlation
        from analysis.correlations.pearson import calculate_pearson_correlation
        correlation = calculate_pearson_correlation(data['fabs'], data['ec_ftir'])
        
        print(f"✓ Sample data created: {len(data)} samples")
        print(f"✓ Validation passed: {mask.sum()} valid samples")
        print(f"✓ Statistics calculated: MAC mean = {stats['mac_mean']:.2f}")
        print(f"✓ Correlation calculated: r = {correlation['pearson_r']:.3f}")
        
        print("\\n🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analyzers():
    """Test that analyzers can be instantiated and used"""
    print("\\nTesting analyzers...")
    
    try:
        import pandas as pd
        
        # Create sample data
        data = pd.DataFrame({
            'fabs': [10, 20, 30, 40],
            'ec_ftir': [2, 4, 6, 8],
            'oc_ftir': [8, 16, 24, 32]
        })
        
        # Test FabsECAnalyzer
        from analysis.ftir.fabs_ec_analyzer import FabsECAnalyzer
        fabs_analyzer = FabsECAnalyzer()
        fabs_results = fabs_analyzer.analyze(data)
        
        print(f"✓ FabsECAnalyzer: {fabs_results['sample_info']['valid_samples']} samples processed")
        print(f"  MAC mean: {fabs_results['mac_statistics']['mac_mean']:.2f}")
        
        # Test OCECAnalyzer
        from analysis.ftir.oc_ec_analyzer import OCECAnalyzer
        oc_analyzer = OCECAnalyzer()
        oc_results = oc_analyzer.analyze(data)
        
        print(f"✓ OCECAnalyzer: {oc_results['sample_info']['valid_samples']} samples processed")
        print(f"  OC/EC ratio: {oc_results['statistics']['oc_ec_ratio']['oc_ec_ratio_mean']:.2f}")
        
        print("\\n🎉 Analyzer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("MODULAR STRUCTURE INSTALLATION TEST")
    print("=" * 60)
    
    success = True
    
    # Test imports
    success &= test_imports()
    
    # Test basic functionality
    success &= test_basic_functionality()
    
    # Test analyzers
    success &= test_analyzers()
    
    # Final result
    print("\\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! The modular structure is working correctly.")
        print("\\nNext steps:")
        print("1. Run 'python examples/basic_usage.py' to see a full example")
        print("2. Run 'python examples/before_after_comparison.py' to see the benefits")
        print("3. Check the README.md for detailed usage instructions")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that Python path includes the src directory")
        print("3. Verify all module files exist and are properly formatted")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
