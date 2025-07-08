"""File I/O utilities"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import json

def save_results_to_json(results: Dict[str, Any], 
                        output_path: str, 
                        indent: int = 2) -> None:
    """
    Save analysis results to JSON file
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Analysis results to save
    output_path : str
        Path to output JSON file
    indent : int, default 2
        JSON indentation level
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj
    
    # Recursively convert numpy types
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(item) for item in d]
        else:
            return convert_numpy_types(d)
    
    clean_results = clean_dict(results)
    
    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=indent, default=str)

def load_results_from_json(input_path: str) -> Dict[str, Any]:
    """
    Load analysis results from JSON file
    
    Parameters:
    -----------
    input_path : str
        Path to input JSON file
        
    Returns:
    --------
    Dict[str, Any]
        Loaded analysis results
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        return json.load(f)

def save_dataframe_to_csv(df: pd.DataFrame, 
                         output_path: str, 
                         index: bool = False) -> None:
    """
    Save DataFrame to CSV file
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    output_path : str
        Path to output CSV file
    index : bool, default False
        Whether to include index in output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=index)

def ensure_output_directory(output_dir: str) -> Path:
    """
    Ensure output directory exists
    
    Parameters:
    -----------
    output_dir : str
        Path to output directory
        
    Returns:
    --------
    Path
        Path object for the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
