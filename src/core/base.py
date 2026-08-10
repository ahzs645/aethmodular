"""Base classes and interfaces."""

from typing import Any, Dict
import pandas as pd

class BaseAnalyzer:
    """Base class for all analyzers"""
    
    def __init__(self, name: str):
        self.name = name
        self.results: Dict[str, Any] = {}
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform analysis on data"""
        raise NotImplementedError("Subclasses should implement analyze()")
    
    def validate_input(self, data: pd.DataFrame, required_columns: list) -> None:
        """Validate input data"""
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"{self.name}: Missing columns {missing}")

class BaseLoader:
    """Base class for all data loaders"""
    
    def load(self, source: str) -> pd.DataFrame:
        """Load data from source"""
        raise NotImplementedError("Subclasses should implement load()")
