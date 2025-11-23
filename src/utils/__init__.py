"""
Utilities module for metrics and plotting.

This module contains helper functions for model evaluation and visualization.
"""

from .metrics import (
    evaluate_classifier,
    evaluate_regressor,
    calculate_sharpe_ratio,
    backtest_strategy
)
from .plotting import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_returns_distribution
)

__all__ = [
    'evaluate_classifier',
    'evaluate_regressor',
    'calculate_sharpe_ratio',
    'backtest_strategy',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_feature_importance',
    'plot_returns_distribution',
]
