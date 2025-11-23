"""
Data module for loading, processing, and preparing financial data.
For now, this package only exposes a few helper functions used in the
early stages of the project.
"""

from .load_data import load_fake_earnings_data
from .build_labels import add_labels, get_features_and_labels

__all__ = [
    "load_fake_earnings_data",
    "add_labels",
    "get_features_and_labels",
]
