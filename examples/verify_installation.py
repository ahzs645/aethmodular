"""
Installation Verification Script
===============================

This script verifies that all ETAD modules and dependencies are properly installed
and functioning correctly.
"""

import sys
import os
from pathlib import Path

print("🔍 ETAD Installation Verification")
print("=" * 50)

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.insert(0, str(src_path))

# Test basic imports
print("\n1️⃣  Testing Core Module Imports...")
try:
    from src.core.base import BaseAnalyzer
    from src.core.exceptions import ETADError
    print("   ✅ Core modules imported successfully")
    core_success = True
except ImportError as e:
    print(f"   ❌ Core module import failed: {e}")
    core_success = False

# Test infrastructure imports
print("\n2️⃣  Testing Infrastructure Modules...")
try:
    from src.core.monitoring import monitor_performance, handle_errors
    from src.core.parallel_processing import ParallelProcessor
    from src.utils.memory_optimization import MemoryOptimizer
    from src.utils.logging.logger import get_logger
    print("   ✅ Infrastructure modules imported successfully")
    infrastructure_success = True
except ImportError as e:
    print(f"   ❌ Infrastructure module import failed: {e}")
    infrastructure_success = False

# Test analysis modules
print("\n3️⃣  Testing Analysis Modules...")
try:
    from src.analysis.aethalometer.smoothening import ONASmoothing, CMASmoothing
    from src.analysis.correlations.pearson import PearsonAnalyzer
    from src.analysis.ftir.enhanced_mac_analyzer import EnhancedMACAnalyzer
    print("   ✅ Basic analysis modules imported successfully")
    analysis_success = True
except ImportError as e:
    print(f"   ❌ Analysis module import failed: {e}")
    analysis_success = False

# Test quality modules
print("\n4️⃣  Testing Quality Analysis Modules...")
try:
    from src.analysis.quality.completeness_analyzer import CompletenessAnalyzer
    from src.analysis.quality.missing_data_analyzer import MissingDataAnalyzer
    from src.analysis.quality.period_classifier import PeriodClassifier
    print("   ✅ Quality analysis modules imported successfully")
    quality_success = True
except ImportError as e:
    print(f"   ❌ Quality analysis module import failed: {e}")
    quality_success = False

# Test advanced analytics modules
print("\n5️⃣  Testing Advanced Analytics Modules...")
try:
    from src.analysis.advanced.statistical_analysis import StatisticalComparator
    from src.analysis.advanced.ml_analysis import MLModelTrainer
    from src.analysis.advanced.time_series_analysis import TimeSeriesAnalyzer
    print("   ✅ Advanced analytics modules imported successfully")
    advanced_success = True
except ImportError as e:
    print(f"   ❌ Advanced analytics module import failed: {e}")
    advanced_success = False

# Test dependencies
print("\n6️⃣  Testing Dependencies...")
dependencies = {
    'pandas': 'pandas',
    'numpy': 'numpy', 
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scikit-learn': 'sklearn',
    'psutil': 'psutil'
}

dependency_results = {}
for name, import_name in dependencies.items():
    try:
        __import__(import_name)
        print(f"   ✅ {name}")
        dependency_results[name] = True
    except ImportError:
        print(f"   ❌ {name} - not installed")
        dependency_results[name] = False

# Test basic functionality
print("\n7️⃣  Testing Basic Functionality...")
functionality_success = True

try:
    # Test analyzer initialization
    analyzer = BaseAnalyzer("test")
    print("   ✅ BaseAnalyzer initialization")
except Exception as e:
    print(f"   ❌ BaseAnalyzer initialization failed: {e}")
    functionality_success = False

try:
    # Test monitoring decorator
    @monitor_performance
    def test_function():
        return "test"
    
    result = test_function()
    print("   ✅ Performance monitoring")
except Exception as e:
    print(f"   ❌ Performance monitoring failed: {e}")
    functionality_success = False

try:
    # Test memory optimization
    optimizer = MemoryOptimizer()
    print("   ✅ Memory optimization")
except Exception as e:
    print(f"   ❌ Memory optimization failed: {e}")
    functionality_success = False

if quality_success:
    try:
        # Test quality analyzer
        completeness = CompletenessAnalyzer()
        print("   ✅ Quality analysis initialization")
    except Exception as e:
        print(f"   ❌ Quality analysis initialization failed: {e}")
        functionality_success = False

# Summary
print("\n" + "=" * 50)
print("📊 INSTALLATION VERIFICATION SUMMARY")
print("=" * 50)

results = {
    'Core Modules': core_success,
    'Infrastructure': infrastructure_success,
    'Analysis Modules': analysis_success,
    'Quality Analysis': quality_success,
    'Advanced Analytics': advanced_success,
    'Basic Functionality': functionality_success
}

# Count successes
total_tests = len(results)
successful_tests = sum(results.values())
dependency_count = sum(dependency_results.values())
total_dependencies = len(dependency_results)

print(f"\n🎯 Module Tests: {successful_tests}/{total_tests} passed")
print(f"📦 Dependencies: {dependency_count}/{total_dependencies} installed")

for test_name, success in results.items():
    status = "✅" if success else "❌"
    print(f"   {status} {test_name}")

print(f"\n📦 Dependency Status:")
for dep_name, installed in dependency_results.items():
    status = "✅" if installed else "❌"
    print(f"   {status} {dep_name}")

# Overall status
if successful_tests == total_tests and dependency_count == total_dependencies:
    print(f"\n🎉 INSTALLATION COMPLETE AND VERIFIED!")
    print(f"   All modules and dependencies are working correctly.")
    print(f"   You can now run the comprehensive demo.")
    exit_code = 0
elif successful_tests >= total_tests * 0.8 and dependency_count >= total_dependencies * 0.8:
    print(f"\n✅ INSTALLATION MOSTLY COMPLETE")
    print(f"   Most modules are working correctly.")
    if dependency_count < total_dependencies:
        missing_deps = [name for name, installed in dependency_results.items() if not installed]
        print(f"   Missing dependencies: {', '.join(missing_deps)}")
        print(f"   Install with: pip install {' '.join(missing_deps)}")
    exit_code = 0
else:
    print(f"\n❌ INSTALLATION ISSUES DETECTED")
    print(f"   Some critical modules or dependencies are missing.")
    print(f"   Please check the error messages above and:")
    print(f"   1. Install missing dependencies: pip install -r requirements.txt")
    print(f"   2. Ensure all source files are present")
    print(f"   3. Check Python path configuration")
    exit_code = 1

# Installation instructions
if not all(dependency_results.values()):
    print(f"\n📋 Installation Instructions:")
    print(f"   1. Install dependencies:")
    print(f"      pip install -r requirements.txt")
    print(f"   2. Or install individually:")
    for name, installed in dependency_results.items():
        if not installed:
            print(f"      pip install {name}")

print(f"\n🚀 Next Steps:")
if exit_code == 0:
    print(f"   • Run examples/comprehensive_advanced_analytics_demo.py")
    print(f"   • Run tests with: python -m pytest tests/")
    print(f"   • Check examples/ directory for usage examples")
else:
    print(f"   • Fix installation issues above")
    print(f"   • Re-run this verification script")

sys.exit(exit_code)
