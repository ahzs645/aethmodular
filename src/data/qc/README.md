# Quality Control (QC) Module

The Quality Control (QC) module provides comprehensive tools for assessing data quality in aethalometer time series data and mapping filter sampling periods to high-quality data periods. This modular system allows you to use individual components as needed or run complete quality assessments.

## Module Overview

The QC module is organized into focused sub-modules:

- **`missing_data`**: Analysis of missing data patterns and gaps
- **`quality_classifier`**: Classification of data quality periods based on missing data
- **`seasonal_patterns`**: Seasonal and temporal pattern analysis
- **`filter_mapping`**: Mapping filter samples to aethalometer quality periods
- **`visualization`**: Plotting and visualization tools
- **`reports`**: Comprehensive quality reports and exports

## Quick Start

### Basic Quality Check
```python
from src.data.qc import quick_quality_check

# Quick overview of data quality
quick_quality_check(your_dataframe)
```

### Individual Module Usage
```python
from src.data.qc import MissingDataAnalyzer, QualityClassifier, QualityVisualizer

# 1. Analyze missing data patterns
analyzer = MissingDataAnalyzer()
missing_analysis = analyzer.analyze_missing_patterns(df, freq='min')

# 2. Classify quality periods
classifier = QualityClassifier()
quality_periods = classifier.classify_9am_to_9am_periods(missing_analysis)

# 3. Visualize results
visualizer = QualityVisualizer()
visualizer.plot_quality_distribution(quality_periods)
```

### Comprehensive Report
```python
from src.data.qc.reports import create_comprehensive_report

# Generate complete quality assessment
results = create_comprehensive_report(
    df=your_dataframe,
    db_path='path/to/filter_database.db',  # Optional
    output_dir='outputs/quality_assessment'
)
```

## Module Components

### 1. Missing Data Analyzer (`missing_data.py`)

Analyzes missing data patterns in time series data.

**Key Features:**
- Identifies missing timestamps in expected timeline
- Categorizes missing periods (full days vs partial)
- Calculates temporal and seasonal missing patterns
- Provides continuous missing period analysis

**Example:**
```python
from src.data.qc.missing_data import MissingDataAnalyzer

analyzer = MissingDataAnalyzer()
results = analyzer.analyze_missing_patterns(df, freq='min')

# Get missing data statistics
print(f"Data completeness: {100 - results['timeline']['missing_percentage']:.1f}%")
print(f"Full missing days: {results['daily_patterns']['n_full_missing_days']}")

# Get significant data gaps
gaps = analyzer.get_missing_periods(min_duration='1H')
```

### 2. Quality Classifier (`quality_classifier.py`)

Classifies data quality periods based on missing data thresholds.

**Default Quality Thresholds:**
- **Excellent**: ≤10 minutes missing
- **Good**: 11-60 minutes missing  
- **Moderate**: 61-240 minutes missing
- **Poor**: >240 minutes missing

**Example:**
```python
from src.data.qc.quality_classifier import QualityClassifier

classifier = QualityClassifier()

# Classify by daily periods (midnight-to-midnight)
daily_quality = classifier.classify_daily_periods(missing_analysis)

# Classify by 9am-to-9am periods (for filter sample alignment)
filter_quality = classifier.classify_9am_to_9am_periods(missing_analysis)

# Get high-quality periods only
high_quality = classifier.get_high_quality_periods(filter_quality)
```

### 3. Seasonal Pattern Analyzer (`seasonal_patterns.py`)

Analyzes seasonal and diurnal patterns in data quality and missingness.

**Features:**
- Seasonal missing data patterns (supports custom season definitions)
- Diurnal (hourly) pattern analysis
- Weekly pattern analysis
- Quality comparison across seasons

**Example:**
```python
from src.data.qc.seasonal_patterns import SeasonalPatternAnalyzer

analyzer = SeasonalPatternAnalyzer()

# Analyze seasonal patterns (excludes full missing days by default)
seasonal_results = analyzer.analyze_seasonal_missing_patterns(missing_analysis)

# Compare quality across seasons
quality_comparison = analyzer.compare_seasonal_quality(quality_series)

# Get diurnal patterns
diurnal_results = analyzer.analyze_diurnal_patterns(missing_analysis, quality_series)
```

### 4. Filter Sample Mapper (`filter_mapping.py`)

Maps filter sampling periods to aethalometer data quality periods for comparison studies.

**Features:**
- Loads filter data from SQLite databases or CSV files
- Maps 24-hour filter sampling periods to 9am-to-9am aethalometer periods
- Identifies overlaps with high-quality aethalometer data
- Exports quality-filtered datasets

**Example:**
```python
from src.data.qc.filter_mapping import FilterSampleMapper

mapper = FilterSampleMapper()

# Load filter data from database
etad_data, ftir_data = mapper.load_filter_data_from_db('database.db', 'ETAD')

# Map to quality periods
overlap_results = mapper.map_to_quality_periods(quality_series, etad_data, ftir_data)

# Export usable periods for comparison studies
usable_periods = mapper.export_quality_filtered_periods(
    include_all_quality=True,
    output_path='outputs/filter_quality_periods.csv'
)

# Get periods suitable for EC-BC comparison
comparison_periods = mapper.get_usable_periods(min_quality='Good')
```

### 5. Quality Visualizer (`visualization.py`)

Comprehensive visualization tools for quality assessment results.

**Visualization Types:**
- Missing data pattern plots
- Quality distribution charts
- Seasonal pattern heatmaps
- Filter sample overlap summaries
- Quality timelines

**Example:**
```python
from src.data.qc.visualization import QualityVisualizer

viz = QualityVisualizer()

# Plot missing data patterns
viz.plot_missing_patterns(missing_analysis)

# Plot quality distribution
viz.plot_quality_distribution(quality_series, "Data Quality (9am-to-9am)")

# Plot seasonal patterns
viz.plot_seasonal_patterns(seasonal_analysis)

# Plot filter overlap summary
viz.plot_filter_overlap_summary(overlap_results)

# Plot quality trends over time
viz.plot_quality_timeline(quality_series, window='M')
```

### 6. Report Generator (`reports.py`)

Generates comprehensive quality assessment reports combining all analysis components.

**Features:**
- Complete automated quality assessment workflow
- JSON and CSV exports
- Summary statistics compilation
- Integrated visualization generation

**Example:**
```python
from src.data.qc.reports import QualityReportGenerator

generator = QualityReportGenerator('outputs/quality_assessment')

# Generate complete report
results = generator.generate_complete_report(
    df=aethalometer_data,
    db_path='filter_database.db',
    site_code='ETAD',
    period_type='9am_to_9am'
)

# Generate quick summary
quick_summary = generator.generate_quick_summary(df)

# Export only usable periods
usable_df = generator.export_usable_periods_only(
    min_quality='Good',
    output_path='outputs/usable_periods.csv'
)
```

## Quality Thresholds

The default quality classification uses these thresholds for missing minutes per period:

| Quality Level | Missing Minutes | Description |
|---------------|----------------|-------------|
| Excellent     | ≤ 10          | Minimal data gaps, suitable for all analyses |
| Good          | 11-60         | Minor gaps, suitable for most analyses |
| Moderate      | 61-240        | Noticeable gaps, use with caution |
| Poor          | > 240         | Significant gaps, generally not suitable |

You can customize these thresholds:

```python
custom_thresholds = {
    'excellent': 5,    # More stringent
    'good': 30,        # More stringent  
    'moderate': 120    # More stringent
}

classifier = QualityClassifier(custom_thresholds)
```

## Output Files

The QC module generates several types of output files:

### Summary Reports
- `quality_report_summary_YYYYMMDD_HHMMSS.json`: Complete analysis summary
- `quality_periods_YYYYMMDD_HHMMSS.csv`: Quality classifications by period

### Filter Analysis
- `filter_samples_quality_9am_to_9am_YYYYMMDD_HHMMSS.csv`: Filter samples with quality info
- `usable_periods.csv`: Periods suitable for comparison studies

### Visualizations
Generated automatically and displayed during analysis.

## Database Schema

For filter sample mapping, the module expects a SQLite database with this structure:

```sql
-- Filters table
CREATE TABLE filters (
    filter_id TEXT PRIMARY KEY,
    sample_date DATE,
    site_code TEXT
);

-- Measurements table  
CREATE TABLE ftir_sample_measurements (
    filter_id TEXT,
    ec_ftir REAL,
    oc_ftir REAL,
    fabs REAL,
    FOREIGN KEY(filter_id) REFERENCES filters(filter_id)
);
```

## Season Definitions

The module supports custom season mapping functions. The default uses Ethiopian seasons:

- **Dry Season**: October-February
- **Belg Rainy Season**: March-May  
- **Kiremt Rainy Season**: June-September

To use meteorological seasons:

```python
from src.data.qc.seasonal_patterns import SeasonalPatternAnalyzer

# Use built-in meteorological seasons
analyzer = SeasonalPatternAnalyzer()
analyzer.season_mapping_func = analyzer._meteorological_season_mapping

# Or define custom seasons
def custom_seasons(month):
    if month in [11, 12, 1, 2]:
        return 'Dry Season'
    elif month in [3, 4, 5, 6]:
        return 'Transition'
    else:
        return 'Rainy Season'

analyzer = SeasonalPatternAnalyzer(custom_seasons)
```

## Example Workflow

Here's a complete workflow for quality assessment:

```python
import pandas as pd
from src.data.qc import *

# 1. Load your data
df = pd.read_csv('aethalometer_data.csv', index_col='datetime', parse_dates=True)

# 2. Quick check
quick_quality_check(df)

# 3. Detailed analysis
analyzer = MissingDataAnalyzer()
missing_analysis = analyzer.analyze_missing_patterns(df)

classifier = QualityClassifier()
quality_series = classifier.classify_9am_to_9am_periods(missing_analysis)

# 4. Filter mapping (if you have filter samples)
mapper = FilterSampleMapper()
etad_data, ftir_data = mapper.load_filter_data_from_db('filters.db')
overlap_results = mapper.map_to_quality_periods(quality_series, etad_data, ftir_data)

# 5. Export results
usable_periods = mapper.export_quality_filtered_periods(
    output_path='outputs/quality_filtered_periods.csv'
)

# 6. Visualize
visualizer = QualityVisualizer()
visualizer.plot_quality_distribution(quality_series)
visualizer.plot_filter_overlap_summary(overlap_results)
```

## Best Practices

1. **Always check data frequency**: Ensure the `freq` parameter matches your data's actual frequency.

2. **Use appropriate period types**: 
   - Use `'daily'` for general quality assessment
   - Use `'9am_to_9am'` when mapping to filter samples

3. **Customize quality thresholds**: Adjust thresholds based on your specific analysis requirements.

4. **Exclude full missing days**: For pattern analysis, exclude fully missing days to focus on partial data gaps.

5. **Validate filter mapping**: Always check that filter sample dates align correctly with your quality periods.

6. **Export intermediate results**: Save quality classifications and usable periods for reproducible analysis.

## Dependencies

The QC module requires:
- pandas
- numpy 
- matplotlib
- seaborn
- sqlite3 (for database functionality)

Make sure these are installed in your environment.
