"""Base classes and interfaces"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class BaseAnalyzer(ABC):
    """Base class for all analyzers"""
    
    def __init__(self, name: str):
        self.name = name
        self.results: Dict[str, Any] = {}
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform analysis on data"""
        pass
    
    def validate_input(self, data: pd.DataFrame, required_columns: list) -> None:
        """Validate input data"""
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"{self.name}: Missing columns {missing}")

class BaseLoader(ABC):
    """Base class for all data loaders"""
    
    @abstractmethod
    def load(self, source: str) -> pd.DataFrame:
        """Load data from source"""
        pass
