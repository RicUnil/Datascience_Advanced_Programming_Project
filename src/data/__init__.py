"""
Data module for loading, processing, and preparing financial data.

This module contains functions for:
- Loading firm fundamentals from Capital IQ
- Downloading stock prices from Yahoo Finance
- Building pre-earnings features
- Constructing post-earnings labels
"""

from .load_data import load_fundamentals, load_prices, load_spy_benchmark
from .build_features import build_features, FeatureEngineer
from .build_labels import build_labels, LabelConstructor

__all__ = [
    'load_fundamentals',
    'load_prices',
    'load_spy_benchmark',
    'build_features',
    'FeatureEngineer',
    'build_labels',
    'LabelConstructor',
]
