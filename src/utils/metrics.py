"""
Evaluation metrics for classification and regression models.

This module provides functions for:
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Regression metrics (MSE, MAE, R²)
- Financial metrics (Sharpe ratio, information ratio)
- Backtesting functions
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)


def evaluate_classifier(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_pred_proba: Optional[np.ndarray] = None) -> Dict:
    """
    Evaluate classification model performance.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities (for ROC-AUC)
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    # TODO: Implement classification evaluation
    # 1. Calculate accuracy, precision, recall, F1
    # 2. Calculate ROC-AUC if probabilities provided
    # 3. Generate confusion matrix
    # 4. Return metrics dictionary
    
    raise NotImplementedError("Function to be implemented")


def evaluate_regressor(y_true: np.ndarray,
                       y_pred: np.ndarray) -> Dict:
    """
    Evaluate regression model performance.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    # TODO: Implement regression evaluation
    # 1. Calculate MSE, RMSE, MAE
    # 2. Calculate R²
    # 3. Calculate directional accuracy
    # 4. Return metrics dictionary
    
    raise NotImplementedError("Function to be implemented")


def calculate_sharpe_ratio(returns: pd.Series,
                           risk_free_rate: float = 0.0,
                           periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    risk_free_rate : float, default=0.0
        Risk-free rate (annualized)
    periods_per_year : int, default=252
        Number of periods per year (252 for daily)
        
    Returns
    -------
    float
        Sharpe ratio
    """
    # TODO: Implement Sharpe ratio calculation
    raise NotImplementedError("Function to be implemented")


def calculate_information_ratio(returns: pd.Series,
                                 benchmark_returns: pd.Series) -> float:
    """
    Calculate information ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns
    benchmark_returns : pd.Series
        Benchmark returns
        
    Returns
    -------
    float
        Information ratio
    """
    # TODO: Implement information ratio calculation
    raise NotImplementedError("Function to be implemented")


def backtest_strategy(predictions: pd.DataFrame,
                      actual_returns: pd.DataFrame,
                      transaction_cost: float = 0.001) -> Dict:
    """
    Backtest trading strategy based on model predictions.
    
    Parameters
    ----------
    predictions : pd.DataFrame
        Model predictions (signals)
    actual_returns : pd.DataFrame
        Actual realized returns
    transaction_cost : float, default=0.001
        Transaction cost as fraction (0.001 = 0.1%)
        
    Returns
    -------
    dict
        Backtest results including returns, Sharpe ratio, etc.
    """
    # TODO: Implement backtesting logic
    # 1. Generate trading signals from predictions
    # 2. Calculate strategy returns
    # 3. Apply transaction costs
    # 4. Calculate performance metrics
    # 5. Return results dictionary
    
    raise NotImplementedError("Function to be implemented")


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
        
    Returns
    -------
    float
        Maximum drawdown
    """
    # TODO: Implement max drawdown calculation
    raise NotImplementedError("Function to be implemented")


def calculate_hit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate hit ratio (directional accuracy).
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        Hit ratio (fraction of correct direction predictions)
    """
    # TODO: Implement hit ratio calculation
    raise NotImplementedError("Function to be implemented")
