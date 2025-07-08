"""
FTIR and Aethalometer Analysis Package

A modular package for analyzing FTIR and Aethalometer data
for carbonaceous aerosol measurements.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config.settings import *
from .core.base import BaseAnalyzer, BaseLoader
