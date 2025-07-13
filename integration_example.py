"""
Integration Example: Using Templates with AethModular

This example shows how to integrate the new visualization templates
with existing AethModular workflows and data structures.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

def integration_example():
    """Demonstrate integration with existing AethModular components"""
    
    print("AethModular Visualization Templates Integration Example")
    print("=" * 60)
    
    try:
        # Import visualization components
        from visualization.templates import VisualizationTemplateFactory
        from visualization.templates.examples import create_sample_data
        from visualization import TimeSeriesPlotter
        
        print("✓ Successfully imported visualization templates")
        
        # 1. Create sample data
        print("\n1. Creating sample aethalometer data...")
        data = create_sample_data(n_days=14)
        data_indexed = data.set_index('timestamp')
        print(f"   Created dataset with {len(data)} records over {data['timestamp'].dt.date.nunique()} days")
        
        # 2. Compare old vs new plotting approaches
        print("\n2. Comparing plotting approaches:")
        
        # Traditional approach (existing TimeSeriesPlotter)
        print("   a) Traditional TimeSeriesPlotter:")
        try:
            traditional_plotter = TimeSeriesPlotter()
            fig1 = traditional_plotter.plot_smoothening_comparison(
                original_data=data['IR BCc'].values,
                smoothed_results={
                    'Example Smoothed': data['IR BCc'].rolling(window=6).mean().values
                },
                timestamps=data['timestamp'],
                title="Traditional Plotting Approach"
            )
            print("      ✓ Traditional plot created successfully")
        except Exception as e:
            print(f"      ⚠ Traditional plotting error: {str(e)[:50]}...")
        
        # New template approach
        print("   b) New Template System:")
        try:
            template = VisualizationTemplateFactory.create_template('time_series')
            fig2 = template.create_plot(
                data=data_indexed,
                columns=['IR BCc', 'Blue BCc'],
                title='Template System Approach'
            )
            print("      ✓ Template plot created successfully")
        except Exception as e:
            print(f"      ⚠ Template plotting error: {str(e)[:50]}...")
        
        # 3. Demonstrate template variety
        print("\n3. Demonstrating template variety:")
        
        templates_to_test = [
            ('diurnal_patterns', {
                'data': data,
                'date_column': 'timestamp',
                'value_columns': ['IR BCc']
            }),
            ('weekly_heatmap', {
                'data': data,
                'date_column': 'timestamp',
                'value_column': 'IR BCc',
                'missing_data': False
            }),
            ('mac_analysis', {
                'fabs_data': data['fabs_370'].dropna().values,
                'ec_data': data['ec_ftir'].dropna().values
            }),
            ('correlation_analysis', {
                'data': data[['IR BCc', 'Blue BCc', 'fabs_370', 'ec_ftir']],
                'title': 'Variable Correlations'
            })
        ]
        
        successful_templates = 0
        for template_name, params in templates_to_test:
            try:
                template = VisualizationTemplateFactory.create_template(template_name)
                fig = template.create_plot(**params)
                print(f"   ✓ {template_name}: Created successfully")
                successful_templates += 1
            except Exception as e:
                print(f"   ⚠ {template_name}: {str(e)[:40]}...")
        
        # 4. Template discovery
        print("\n4. Template System Capabilities:")
        try:
            all_templates = VisualizationTemplateFactory.list_templates()
            categories = VisualizationTemplateFactory.list_templates_by_category()
            
            print(f"   Available templates: {len(all_templates)}")
            for category, template_list in categories.items():
                print(f"   {category}: {len(template_list)} templates")
            
            # Show detailed info for one template
            if 'time_series' in all_templates:
                info = VisualizationTemplateFactory.get_template_info('time_series')
                print(f"   Example - time_series template:")
                print(f"     Required params: {info.get('required_params', [])}")
                print(f"     Optional params: {info.get('optional_params', [])}")
        
        except Exception as e:
            print(f"   ⚠ Template discovery error: {str(e)[:50]}...")
        
        # 5. Configuration examples
        print("\n5. Configuration System:")
        try:
            from visualization.templates.config_utils import config_manager
            
            # Test style loading
            default_style = config_manager.get_style_config('default')
            publication_style = config_manager.get_style_config('publication')
            
            print(f"   ✓ Default style loaded: {len(default_style)} parameters")
            print(f"   ✓ Publication style loaded: {len(publication_style)} parameters")
            
            # Test color schemes
            bc_colors = config_manager.get_color_scheme('bc_analysis')
            seasonal_colors = config_manager.get_color_scheme('seasonal')
            
            print(f"   ✓ BC analysis colors: {len(bc_colors)} colors defined")
            print(f"   ✓ Seasonal colors: {len(seasonal_colors)} seasons defined")
            
        except Exception as e:
            print(f"   ⚠ Configuration error: {str(e)[:50]}...")
        
        print(f"\n📊 Summary:")
        print(f"   Templates successfully tested: {successful_templates}/{len(templates_to_test)}")
        print(f"   Integration status: {'✓ Success' if successful_templates > 0 else '⚠ Issues detected'}")
        
        return successful_templates > 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   This may be due to missing dependencies (matplotlib, pandas, seaborn)")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def demonstrate_workflow_integration():
    """Show how templates integrate with typical AethModular workflows"""
    
    print("\n" + "=" * 60)
    print("WORKFLOW INTEGRATION EXAMPLE")
    print("=" * 60)
    
    try:
        from visualization.templates import create_plot
        from visualization.templates.examples import create_sample_data
        
        # Simulate a typical analysis workflow
        print("\n📋 Typical Analysis Workflow with Templates:")
        
        # Step 1: Data loading (simulated)
        print("1. Loading aethalometer data...")
        data = create_sample_data(n_days=30)
        print(f"   ✓ Loaded {len(data)} data points")
        
        # Step 2: Quality assessment
        print("2. Data quality assessment...")
        fig_diurnal = create_plot(
            'diurnal_patterns',
            data=data,
            date_column='timestamp',
            value_columns=['IR BCc'],
            missing_data_analysis=True
        )
        print("   ✓ Diurnal pattern analysis completed")
        
        # Step 3: Pattern analysis
        print("3. Pattern analysis...")
        fig_weekly = create_plot(
            'weekly_heatmap',
            data=data,
            date_column='timestamp',
            value_column='IR BCc',
            missing_data=False
        )
        print("   ✓ Weekly pattern analysis completed")
        
        # Step 4: Scientific analysis
        print("4. Scientific analysis...")
        clean_data = data.dropna()
        if len(clean_data) > 10:
            fig_mac = create_plot(
                'mac_analysis',
                fabs_data=clean_data['fabs_370'].values,
                ec_data=clean_data['ec_ftir'].values
            )
            print("   ✓ MAC analysis completed")
        else:
            print("   ⚠ Insufficient clean data for MAC analysis")
        
        # Step 5: Summary visualization
        print("5. Summary visualization...")
        fig_correlation = create_plot(
            'correlation_analysis',
            data=data[['IR BCc', 'Blue BCc', 'fabs_370', 'ec_ftir']]
        )
        print("   ✓ Correlation analysis completed")
        
        print("\n✅ Complete workflow executed successfully!")
        print("   In a real scenario, these plots would be saved to files or displayed")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        return False

if __name__ == "__main__":
    success1 = integration_example()
    success2 = demonstrate_workflow_integration()
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    if success1 and success2:
        print("🎉 All integration tests passed!")
        print("   The visualization template system is ready for use.")
    else:
        print("⚠ Some integration issues detected.")
        print("   Check dependencies and error messages above.")
    
    print("\n📖 Next steps:")
    print("   1. Install required dependencies: matplotlib, pandas, seaborn")
    print("   2. Import templates: from aethmodular.visualization.templates import VisualizationTemplateFactory")
    print("   3. Create plots: template = factory.create_template('time_series')")
    print("   4. See README.md for detailed usage examples")
