#!/usr/bin/env python3
"""
Comprehensive test and setup verification script for the ETAD Aethalometer Analysis System
Consolidates setup testing, notebook compatibility, and system verification
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class ETADSystemTester:
    """Comprehensive tester for the ETAD system"""
    
    def __init__(self):
        self.passed_tests = 0
        self.total_tests = 0
        self.data_path = "/Users/ahzs645/Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University/Research/Grad/UC Davis Ann/NASA MAIA/Data/Aethelometry Data/Kyan Data/Mergedcleaned and uncleaned MA350 data20250707030704/df_uncleaned_Jacros_API_and_OG.pkl"
    
    def print_header(self, title):
        """Print a formatted header"""
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
    
    def print_test(self, test_name):
        """Print test name"""
        print(f"\n📋 {test_name}")
        print("-" * 50)
    
    def test_result(self, success, message):
        """Record and print test result"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            print(f"   ✅ {message}")
        else:
            print(f"   ❌ {message}")
        return success
    
    def test_directory_structure(self):
        """Test that all required directories exist"""
        self.print_test("Testing Directory Structure")
        
        required_dirs = [
            'src',
            'src/data',
            'src/data/loaders',
            'src/data/processors',
            'src/analysis',
            'src/analysis/bc',
            'src/utils',
            'src/core',
            'src/config',
            'notebooks',
            'data',
            'outputs'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            return self.test_result(False, f"Missing directories: {missing_dirs}")
        else:
            return self.test_result(True, f"All {len(required_dirs)} required directories exist")
    
    def test_required_files(self):
        """Test that all required files exist"""
        self.print_test("Testing Required Files")
        
        required_files = [
            'src/__init__.py',
            'src/core/base.py',
            'src/core/exceptions.py',
            'src/data/loaders/aethalometer.py',
            'src/data/processors/calibration.py',
            'src/utils/plotting.py',
            'src/utils/statistics.py',
            'src/utils/file_io.py',
            'src/config/plotting.py',
            'notebooks/aethalometer_data_analysis.ipynb',
            'requirements.txt',
            'setup.py'
        ]
        
        missing_files = []
        existing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
            else:
                existing_files.append(file_path)
        
        self.test_result(True, f"Found {len(existing_files)} required files")
        if missing_files:
            return self.test_result(False, f"Missing files: {missing_files}")
        else:
            return self.test_result(True, "All required files exist")
    
    def test_core_imports(self):
        """Test core module imports"""
        self.print_test("Testing Core Module Imports")
        
        try:
            from core.base import BaseLoader, BaseAnalyzer
            self.test_result(True, "Core base classes imported")
            
            from core.exceptions import DataValidationError, AnalysisError
            self.test_result(True, "Core exceptions imported")
            
            return True
        except ImportError as e:
            return self.test_result(False, f"Core import failed: {e}")
        except Exception as e:
            return self.test_result(False, f"Unexpected core import error: {e}")
    
    def test_data_loaders(self):
        """Test data loader imports"""
        self.print_test("Testing Data Loader Imports")
        
        try:
            from data.loaders.aethalometer import AethalometerPKLLoader, load_aethalometer_data
            self.test_result(True, "Aethalometer loader imported")
            
            return True
        except ImportError as e:
            return self.test_result(False, f"Data loader import failed: {e}")
        except Exception as e:
            return self.test_result(False, f"Unexpected data loader error: {e}")
    
    def test_utilities(self):
        """Test utility imports"""
        self.print_test("Testing Utility Imports")
        
        try:
            from utils.plotting import AethalometerPlotter
            self.test_result(True, "Plotting utilities imported")
            
            from utils.statistics import StatisticalAnalyzer
            self.test_result(True, "Statistical utilities imported")
            
            from utils.file_io import save_results_to_json, ensure_output_directory
            self.test_result(True, "File I/O utilities imported")
            
            from config.plotting import setup_plotting_style
            self.test_result(True, "Plotting configuration imported")
            
            return True
        except ImportError as e:
            return self.test_result(False, f"Utility import failed: {e}")
        except Exception as e:
            return self.test_result(False, f"Unexpected utility error: {e}")
    
    def test_analysis_modules(self):
        """Test analysis module imports"""
        self.print_test("Testing Analysis Module Imports")
        
        try:
            from analysis.bc.black_carbon_analyzer import BlackCarbonAnalyzer
            self.test_result(True, "Black carbon analyzer imported")
            
            # Try to import other analysis modules if they exist
            try:
                from analysis.bc.source_apportionment import SourceApportionmentAnalyzer
                self.test_result(True, "Source apportionment analyzer imported")
            except ImportError:
                self.test_result(False, "Source apportionment analyzer not available")
            
            return True
        except ImportError as e:
            return self.test_result(False, f"Analysis module import failed: {e}")
        except Exception as e:
            return self.test_result(False, f"Unexpected analysis module error: {e}")
    
    def test_class_instantiation(self):
        """Test class instantiation"""
        self.print_test("Testing Class Instantiation")
        
        try:
            from utils.plotting import AethalometerPlotter
            from utils.statistics import StatisticalAnalyzer
            from analysis.bc.black_carbon_analyzer import BlackCarbonAnalyzer
            
            # Test instantiation
            plotter = AethalometerPlotter()
            self.test_result(True, "AethalometerPlotter instantiated")
            
            stats_analyzer = StatisticalAnalyzer()
            self.test_result(True, "StatisticalAnalyzer instantiated")
            
            bc_analyzer = BlackCarbonAnalyzer()
            self.test_result(True, "BlackCarbonAnalyzer instantiated")
            
            return True
        except Exception as e:
            return self.test_result(False, f"Class instantiation failed: {e}")
    
    def test_data_file_access(self):
        """Test access to the aethalometer data file"""
        self.print_test("Testing Data File Access")
        
        data_file = Path(self.data_path)
        if data_file.exists():
            size_mb = data_file.stat().st_size / 1024**2
            return self.test_result(True, f"Data file found: {size_mb:.1f} MB")
        else:
            return self.test_result(False, f"Data file not found: {self.data_path}")
    
    def test_data_loading(self):
        """Test data loading with both methods"""
        self.print_test("Testing Data Loading")
        
        if not Path(self.data_path).exists():
            return self.test_result(False, "Data file not available for testing")
        
        try:
            # Test direct pandas loading
            df_direct = pd.read_pickle(self.data_path)
            self.test_result(True, f"Direct pandas load: {len(df_direct)} rows × {len(df_direct.columns)} columns")
            
            # Sample columns
            sample_cols = list(df_direct.columns)[:5]
            self.test_result(True, f"Sample columns: {sample_cols}")
            
            # Test modular loader
            from data.loaders.aethalometer import AethalometerPKLLoader
            loader = AethalometerPKLLoader(self.data_path, format_type="auto")
            summary = loader.get_data_summary()
            self.test_result(True, f"Modular loader: {summary['total_samples']} rows, format: {summary['format_type']}")
            
            return True
        except Exception as e:
            return self.test_result(False, f"Data loading failed: {e}")
    
    def test_plotting_setup(self):
        """Test plotting configuration"""
        self.print_test("Testing Plotting Setup")
        
        try:
            from config.plotting import setup_plotting_style
            from utils.plotting import AethalometerPlotter
            
            setup_plotting_style()
            self.test_result(True, "Plotting style configured")
            
            plotter = AethalometerPlotter()
            self.test_result(True, "AethalometerPlotter ready")
            
            return True
        except Exception as e:
            return self.test_result(False, f"Plotting setup failed: {e}")
    
    def test_jupyter_dependencies(self):
        """Test Jupyter notebook dependencies"""
        self.print_test("Testing Jupyter Dependencies")
        
        try:
            import jupyter
            self.test_result(True, "Jupyter installed")
            
            import notebook
            self.test_result(True, "Notebook installed")
            
            import ipywidgets
            self.test_result(True, "IPywidgets installed")
            
            return True
        except ImportError as e:
            return self.test_result(False, f"Jupyter dependency missing: {e}")
    
    def test_notebook_files(self):
        """Test notebook files exist and are valid"""
        self.print_test("Testing Notebook Files")
        
        notebook_path = Path("notebooks/aethalometer_data_analysis.ipynb")
        if notebook_path.exists():
            self.test_result(True, "Main analysis notebook exists")
            
            # Check if it's valid JSON
            try:
                import json
                with open(notebook_path, 'r') as f:
                    notebook_data = json.load(f)
                self.test_result(True, "Notebook is valid JSON format")
                
                if 'cells' in notebook_data:
                    cell_count = len(notebook_data['cells'])
                    self.test_result(True, f"Notebook has {cell_count} cells")
                
                return True
            except json.JSONDecodeError:
                return self.test_result(False, "Notebook has invalid JSON format")
        else:
            return self.test_result(False, "Main analysis notebook missing")
    
    def run_all_tests(self):
        """Run all tests"""
        self.print_header("ETAD AETHALOMETER ANALYSIS SYSTEM - COMPREHENSIVE TEST")
        
        # Core system tests
        tests = [
            self.test_directory_structure,
            self.test_required_files,
            self.test_core_imports,
            self.test_data_loaders,
            self.test_utilities,
            self.test_analysis_modules,
            self.test_class_instantiation,
        ]
        
        # Data and notebook tests
        data_tests = [
            self.test_data_file_access,
            self.test_data_loading,
            self.test_plotting_setup,
            self.test_jupyter_dependencies,
            self.test_notebook_files,
        ]
        
        # Run core tests
        self.print_header("CORE SYSTEM TESTS")
        core_passed = sum(test() for test in tests)
        
        # Run data/notebook tests
        self.print_header("DATA & NOTEBOOK TESTS")
        data_passed = sum(test() for test in data_tests)
        
        # Summary
        self.print_header("TEST SUMMARY")
        print(f"📊 RESULTS: {self.passed_tests}/{self.total_tests} tests passed")
        print(f"🔧 Core System: {core_passed}/{len(tests)} tests passed")
        print(f"📓 Data & Notebooks: {data_passed}/{len(data_tests)} tests passed")
        
        if self.passed_tests == self.total_tests:
            print(f"\n🎉 ALL TESTS PASSED!")
            print("✅ The ETAD system is fully operational!")
            print("\n🚀 NEXT STEPS:")
            print("1. Launch notebook: python launch_notebook.py")
            print("2. Or manual launch: jupyter notebook notebooks/aethalometer_data_analysis.ipynb")
            print("3. Run all cells in the notebook to analyze your data")
            print("4. Customize the analysis using notebooks/CUSTOMIZATION_GUIDE.md")
        else:
            print(f"\n⚠️  {self.total_tests - self.passed_tests} TESTS FAILED")
            print("🔧 TROUBLESHOOTING:")
            if core_passed < len(tests):
                print("   - Check that all required modules are properly installed")
                print("   - Verify the directory structure matches the setup")
                print("   - Run: pip install -r requirements.txt")
            if data_passed < len(data_tests):
                print("   - Verify the data file path is correct")
                print("   - Install Jupyter: pip install jupyter notebook ipywidgets")
                print("   - Check that the notebook files are not corrupted")
        
        print(f"\n{'='*60}")
        return self.passed_tests == self.total_tests

def main():
    """Main function"""
    tester = ETADSystemTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
