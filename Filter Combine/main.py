import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent

class FilterDataIntegrator:
    """
    Comprehensive class to integrate FTIR, HIPS, and Chemical Speciation data
    into a unified long-format dataset for analysis
    """
    
    def __init__(self):
        self.unified_data = None
        self.data_sources = {}
        
    def load_ftir_data(self, file_path):
        """Load FTIR data (already in long format)"""
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = BASE_DIR / file_path
        df = pd.read_csv(file_path)
        # Remove field blank records, which are denoted by FilterType == 'FB'
        if 'FilterType' in df.columns:
            fb_mask = df['FilterType'].astype(str).str.upper() == 'FB'
            if fb_mask.any():
                removed = fb_mask.sum()
                print(f"⚠️ Removing {removed} FTIR field blank records (FilterType='FB')")
                df = df.loc[~fb_mask].copy()
        df['DataSource'] = 'FTIR'
        
        # Add Units column and clean column names
        df['Concentration_Units'] = 'ug/m3'  # FTIR concentrations are in ug/m3
        df['MDL_Units'] = 'ug/m3'  # FTIR MDL values are in ug/m3
        df['MassLoading_Units'] = 'ug'  # Mass loading is in micrograms
        
        # Rename columns to remove unit suffixes
        df = df.rename(columns={
            'Concentration_ug_m3': 'Concentration',
            'MDL_ug_m3': 'MDL', 
            'Uncertainty_ug_m3': 'Uncertainty'
        })
        
        self.data_sources['FTIR'] = df
        print(f"✅ FTIR loaded: {len(df)} measurements, {df['FilterId'].nunique()} filters")
        return df
    
    def load_hips_data(self, file_path):
        """Load and convert HIPS data to long format"""
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = BASE_DIR / file_path
        hips_df = pd.read_csv(file_path)
        if 'FilterType' in hips_df.columns:
            fb_mask = hips_df['FilterType'].astype(str).str.upper() == 'FB'
            if fb_mask.any():
                removed = fb_mask.sum()
                print(f"⚠️ Removing {removed} HIPS field blank records (FilterType='FB')")
                hips_df = hips_df.loc[~fb_mask].copy()
        
        # Parameters to extract from HIPS with their units
        hips_parameters = {
            'T1': 'K',  # Temperature in Kelvin
            'R1': 'K',  # Temperature in Kelvin  
            'Intercept': 'dimensionless',
            'Slope': 'dimensionless',
            't': 'dimensionless',
            'r': 'dimensionless', 
            'tau': 'dimensionless',
            'Fabs': 'Mm-1',  # Absorption coefficient
            'MDL': 'Mm-1',   # Detection limit for absorption
            'Uncertainty': 'Mm-1'  # Uncertainty for absorption
        }
        
        hips_long = []
        for _, row in hips_df.iterrows():
            base_info = {
                'Site': row['Site'],
                'Latitude': row['Latitude'], 
                'Longitude': row['Longitude'],
                'Barcode': row['Barcode'],
                'FilterId': row['FilterId'],
                'SampleDate': row['SampleDate'],
                'FilterType': row['FilterType'],
                'LotId': row['LotId'],
                'AnalysisDate': row['AnalysisDate'],
                'AnalysisTime': row['AnalysisTime'],
                'CalibrationSetId': 'HIPS',
                'DepositArea_cm2': row['DepositArea'],
                'Volume_m3': row['Volume'],
                'FilterComments': row['FilterComments'],
                'DataSource': 'HIPS'
            }
            
            for param, unit in hips_parameters.items():
                if pd.notna(row[param]):
                    param_row = base_info.copy()
                    param_row.update({
                        'Parameter': f'HIPS_{param}',
                        'MassLoading_ug': np.nan,
                        'Concentration': row[param],
                        'Concentration_Units': unit,
                        'MDL': row['MDL'] if param == 'MDL' else np.nan,
                        'MDL_Units': 'Mm-1' if param == 'MDL' else np.nan,
                        'Uncertainty': row['Uncertainty'] if param == 'Uncertainty' else np.nan,
                        'MassLoading_Units': np.nan
                    })
                    hips_long.append(param_row)
        
        df = pd.DataFrame(hips_long)
        self.data_sources['HIPS'] = df
        print(f"✅ HIPS loaded & converted: {len(df)} measurements, {df['FilterId'].nunique()} filters")
        return df
    
    def parse_chem_spec_file(self, file_path):
        """Parse chemical speciation file, deleting first 3 comment rows"""
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = BASE_DIR / file_path
        df = pd.read_csv(file_path, skiprows=3)  # Skip first 3 comment lines
        print(f"   Loaded {len(df)} measurements from {Path(file_path).name}")
        return df

    def load_chem_spec_data(self, file_paths):
        """Load and convert chemical speciation data to unified format"""
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        chem_long = []
        total_files = len(file_paths)
        total_measurements = 0
        sites_processed = []
        
        print(f"Processing {total_files} chemical speciation files...")
        
        for file_path in file_paths:
            path_obj = Path(file_path)
            if not path_obj.is_absolute():
                path_obj = BASE_DIR / path_obj
            print(f"📄 Processing {path_obj.name}...")
            chem_df = self.parse_chem_spec_file(path_obj)
            
            # Track site info
            if len(chem_df) > 0:
                site = chem_df['Site_Code'].iloc[0]
                sites_processed.append(site)
                filters_in_file = chem_df['Filter_ID'].nunique()
                print(f"   Site: {site}, Filters: {filters_in_file}, Measurements: {len(chem_df)}")
            
            total_measurements += len(chem_df)
            
            for _, row in chem_df.iterrows():
                # Convert date format
                sample_date = f"{row['Start_Year_local']}-{row['Start_Month_local']:02d}-{row['Start_Day_local']:02d}"
                
                # Clean and standardize units
                units = row['Units'] if pd.notna(row['Units']) else 'unknown'
                # Standardize common unit representations
                units_clean = units.replace('Micrograms per cubic meter (ug/m3)', 'ug/m3')
                units_clean = units_clean.replace('Nanograms per cubic meter (ng/m3)', 'ng/m3')
                units_clean = units_clean.replace('Micrograms per cubic meter', 'ug/m3')
                units_clean = units_clean.replace('Nanograms per cubic meter', 'ng/m3')
                
                unified_row = {
                    'Site': row['Site_Code'],
                    'Latitude': row['Latitude'],
                    'Longitude': row['Longitude'],
                    'Barcode': np.nan,
                    'FilterId': row['Filter_ID'],
                    'SampleDate': sample_date,
                    'FilterType': 'PM2.5',
                    'LotId': np.nan,
                    'AnalysisDate': np.nan,
                    'AnalysisTime': np.nan,
                    'CalibrationSetId': f'ChemSpec_{row["Method_Code"]}',
                    'DepositArea_cm2': np.nan,
                    'Volume_m3': np.nan,
                    'Parameter': f'ChemSpec_{row["Parameter_Name"].replace(" ", "_")}',
                    'MassLoading_ug': np.nan,
                    'MassLoading_Units': np.nan,
                    'Concentration': row['Value'],
                    'Concentration_Units': units_clean,
                    'MDL': row['MDL'],
                    'MDL_Units': units_clean,
                    'Uncertainty': row['UNC'],
                    'FilterComments': row['Flag'],
                    'DataSource': 'ChemSpec'
                }
                chem_long.append(unified_row)
        
        df = pd.DataFrame(chem_long)
        self.data_sources['ChemSpec'] = df
        
        print(f"✅ ChemSpec summary:")
        print(f"   Sites processed: {', '.join(sites_processed)}")
        print(f"   Total measurements: {len(df)}")
        print(f"   Unique filters: {df['FilterId'].nunique()}")
        print(f"   Parameters: {df['Parameter'].nunique()}")
        
        return df
    
    def create_unified_dataset(self):
        """Combine all loaded datasets into unified format"""
        if not self.data_sources:
            raise ValueError("No data sources loaded. Load data first.")
        
        # Combine all datasets
        all_data = []
        for source_name, df in self.data_sources.items():
            all_data.append(df)
        
        self.unified_data = pd.concat(all_data, ignore_index=True)
        
        # Sort for better organization
        self.unified_data = self.unified_data.sort_values(['FilterId', 'Parameter']).reset_index(drop=True)
        
        print(f"\n🎉 UNIFIED DATASET CREATED!")
        print(f"Total measurements: {len(self.unified_data):,}")
        print(f"Unique filters: {self.unified_data['FilterId'].nunique()}")
        print(f"Parameter types: {self.unified_data['Parameter'].nunique()}")
        print(f"Sites: {', '.join(sorted(self.unified_data['Site'].unique()))}")
        
        return self.unified_data
    
    def analyze_dataset(self):
        """Comprehensive analysis of the unified dataset"""
        if self.unified_data is None:
            raise ValueError("Create unified dataset first")
        
        df = self.unified_data
        
        print("\n" + "="*60)
        print("UNIFIED DATASET ANALYSIS")
        print("="*60)
        
        # 1. Data source breakdown
        print("\n📊 Data Source Breakdown:")
        source_counts = df['DataSource'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} measurements")
        
        # 2. Site coverage
        print("\n🌍 Site Coverage:")
        site_analysis = []
        for site in sorted(df['Site'].unique()):
            site_data = df[df['Site'] == site]
            sources = site_data['DataSource'].unique()
            filters = site_data['FilterId'].nunique()
            measurements = len(site_data)
            site_analysis.append({
                'Site': site,
                'Filters': filters,
                'Measurements': measurements,
                'Sources': ', '.join(sorted(sources))
            })
        
        site_df = pd.DataFrame(site_analysis)
        print(site_df.to_string(index=False))
        
        # 3. Parameter richness
        print(f"\n🔬 Parameter Analysis:")
        param_counts = df['Parameter'].value_counts()
        print(f"Total parameter types: {len(param_counts)}")
        
        # Group by data source
        ftir_params = len(df[df['DataSource'] == 'FTIR']['Parameter'].unique())
        hips_params = len(df[df['DataSource'] == 'HIPS']['Parameter'].unique()) 
        chem_params = len(df[df['DataSource'] == 'ChemSpec']['Parameter'].unique())
        
        print(f"  FTIR parameters: {ftir_params}")
        print(f"  HIPS parameters: {hips_params}")
        print(f"  ChemSpec parameters: {chem_params}")
        
        # 4. Temporal coverage
        print(f"\n📅 Temporal Coverage:")
        date_ranges = {}
        for source in df['DataSource'].unique():
            source_data = df[df['DataSource'] == source]
            dates = pd.to_datetime(source_data['SampleDate']).dropna()
            if len(dates) > 0:
                date_ranges[source] = f"{dates.min().date()} to {dates.max().date()}"
        
        for source, range_str in date_ranges.items():
            print(f"  {source}: {range_str}")
        
        # 5. Filter overlap analysis
        print(f"\n🔗 Filter Overlap Analysis:")
        ftir_filters = set(df[df['DataSource'] == 'FTIR']['FilterId'].unique())
        hips_filters = set(df[df['DataSource'] == 'HIPS']['FilterId'].unique())
        chem_filters = set(df[df['DataSource'] == 'ChemSpec']['FilterId'].unique())
        
        ftir_hips_overlap = len(ftir_filters & hips_filters)
        all_three_overlap = len(ftir_filters & hips_filters & chem_filters)
        
        # 6. Units analysis
        print(f"\n📏 Units Analysis:")
        concentration_units = df['Concentration_Units'].value_counts()
        print("Concentration units:")
        for unit, count in concentration_units.items():
            print(f"  {unit}: {count:,} measurements")
        
        mdl_units = df['MDL_Units'].value_counts()
        if len(mdl_units) > 0:
            print("MDL units:")
            for unit, count in mdl_units.items():
                if pd.notna(unit):
                    print(f"  {unit}: {count:,} measurements")
        
        return {
            'site_analysis': site_df,
            'parameter_counts': param_counts,
            'date_ranges': date_ranges,
            'filter_overlaps': {
                'ftir_hips': ftir_hips_overlap,
                'all_three': all_three_overlap
            }
        }
    
    def save_dataset(self, csv_path=None, pkl_path=None):
        """Save the unified dataset to CSV and pickle formats"""
        if self.unified_data is None:
            raise ValueError("Create unified dataset first")

        if csv_path is None:
            csv_path = BASE_DIR / 'unified_filter_dataset.csv'
        csv_path = Path(csv_path)
        if not csv_path.is_absolute():
            csv_path = BASE_DIR / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.unified_data.to_csv(csv_path, index=False)

        if pkl_path is None:
            pkl_path = csv_path.with_suffix('.pkl')
        pkl_path = Path(pkl_path)
        if not pkl_path.is_absolute():
            pkl_path = BASE_DIR / pkl_path
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        self.unified_data.to_pickle(pkl_path)

        print(f"\n💾 Dataset saved: {csv_path}")
        print(f"💾 Pickle saved: {pkl_path}")
        return csv_path, pkl_path
    
    def get_filter_analysis(self, filter_id):
        """Get complete analysis for a specific filter"""
        if self.unified_data is None:
            raise ValueError("Create unified dataset first")
        
        filter_data = self.unified_data[self.unified_data['FilterId'] == filter_id]
        
        if len(filter_data) == 0:
            print(f"Filter {filter_id} not found")
            return None
        
        analysis = {
            'FilterId': filter_id,
            'Site': filter_data['Site'].iloc[0],
            'SampleDate': filter_data['SampleDate'].iloc[0],
            'total_measurements': len(filter_data),
            'data_sources': list(filter_data['DataSource'].unique()),
            'parameters_by_source': {}
        }
        
        for source in filter_data['DataSource'].unique():
            source_data = filter_data[filter_data['DataSource'] == source]
            analysis['parameters_by_source'][source] = {
                'count': len(source_data),
                'parameters': list(source_data['Parameter'].unique())
            }
        
        return analysis
    
    def compare_parameters_across_sites(self, parameter_name):
        """Compare a specific parameter across all sites"""
        if self.unified_data is None:
            raise ValueError("Create unified dataset first")
        
        param_data = self.unified_data[self.unified_data['Parameter'] == parameter_name]
        
        if len(param_data) == 0:
            print(f"Parameter {parameter_name} not found")
            return None
        
        comparison = param_data.groupby('Site')['Concentration'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        
        return comparison

# Example usage and demonstration
def main():
    """Demonstrate the complete integration workflow"""
    
    # Initialize the integrator
    integrator = FilterDataIntegrator()
    
    # Load all datasets
    print("🔄 Loading datasets...")
    ftir_path = BASE_DIR / 'Four_Sites_FTIR_data.v2.csv'
    integrator.load_ftir_data(ftir_path)

    hips_path = BASE_DIR / 'Four_Sites_HIPS_data.csv'
    if hips_path.exists():
        integrator.load_hips_data(hips_path)
    else:
        print(f"⚠️ HIPS data not found at {hips_path}; skipping HIPS integration")
    
    # Load all four chemical speciation site files
    chem_spec_filenames = [
        'FilterBased_ChemSpecPM25_CHTS.csv',
        'FilterBased_ChemSpecPM25_INDH.csv', 
        'FilterBased_ChemSpecPM25_ETAD.csv',
        'FilterBased_ChemSpecPM25_USPA.csv'
    ]
    chem_spec_files = []
    for filename in chem_spec_filenames:
        candidate = BASE_DIR / filename
        if candidate.exists():
            chem_spec_files.append(candidate)
        else:
            print(f"⚠️ ChemSpec file missing: {candidate}")

    if chem_spec_files:
        integrator.load_chem_spec_data(chem_spec_files)
    else:
        print("⚠️ No ChemSpec files loaded; unified dataset will only contain FTIR/HIPS data")
    
    # Create unified dataset
    unified_df = integrator.create_unified_dataset()
    
    # Analyze the dataset
    analysis_results = integrator.analyze_dataset()
    
    # Save the unified dataset
    csv_path, pkl_path = integrator.save_dataset()
    
    # Example analyses
    print("\n📋 Example Analyses:")
    
    # 1. Get analysis for a specific filter from each site
    sites = ['CHTS', 'INDH', 'ETAD', 'USPA']
    for site in sites:
        site_filters = unified_df[unified_df['Site'] == site]['FilterId'].unique()
        if len(site_filters) > 0:
            filter_id = site_filters[0]
            filter_analysis = integrator.get_filter_analysis(filter_id)
            if filter_analysis:
                print(f"\n Example from {site} - Filter {filter_id}:")
                print(f"   Measurements: {filter_analysis['total_measurements']}")
                print(f"   Sources: {', '.join(filter_analysis['data_sources'])}")
    
    # 2. Compare a parameter across sites
    if 'EC_ftir' in unified_df['Parameter'].values:
        print(f"\n📊 EC_ftir comparison across sites:")
        ec_comparison = integrator.compare_parameters_across_sites('EC_ftir')
        print(ec_comparison)
    
    # 3. Show chemical speciation coverage by site
    chem_data = unified_df[unified_df['DataSource'] == 'ChemSpec']
    if len(chem_data) > 0:
        print(f"\n🔬 Chemical Speciation coverage by site:")
        chem_coverage = chem_data.groupby('Site').agg({
            'FilterId': 'nunique',
            'Parameter': 'nunique'
        }).rename(columns={'FilterId': 'Filters', 'Parameter': 'Parameters'})
        print(chem_coverage)
    
    print(f"\n🎉 Integration complete! Your unified dataset is ready for analysis.")
    print(f"Total measurements: {len(unified_df):,}")
    print(f"Sites with complete coverage: {len(unified_df['Site'].unique())}")
    print(f"CSV saved to: {csv_path}")
    print(f"PKL saved to: {pkl_path}")

    return integrator, unified_df

if __name__ == "__main__":
    integrator, unified_data = main()
