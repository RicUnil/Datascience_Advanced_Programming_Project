"""
Data loading module for Capital IQ fundamentals and Yahoo Finance prices.

This module provides functions to:
- Load firm fundamental data from Capital IQ CSV files
- Download stock price data from Yahoo Finance
- Load SPY benchmark data for computing excess returns
- Validate and clean raw data
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from datetime import datetime
import yfinance as yf


def load_fundamentals(filepath: str, 
                      date_column: str = 'date',
                      ticker_column: str = 'ticker') -> pd.DataFrame:
    """
    Load firm fundamental data from Capital IQ CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing fundamental data
    date_column : str, default='date'
        Name of the date column
    ticker_column : str, default='ticker'
        Name of the ticker/symbol column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with fundamental data, indexed by date and ticker
        
    Notes
    -----
    Expected columns include:
    - date: Reporting date
    - ticker: Stock ticker symbol
    - total_assets: Total assets
    - total_revenue: Total revenue
    - net_income: Net income
    - total_debt: Total debt
    - total_equity: Total equity
    - operating_cash_flow: Operating cash flow
    - etc.
    
    Examples
    --------
    >>> fundamentals = load_fundamentals('data/raw/fundamentals.csv')
    >>> print(fundamentals.head())
    """
    # TODO: Implement data loading logic
    # 1. Read CSV file
    # 2. Parse dates
    # 3. Set appropriate index
    # 4. Handle missing values
    # 5. Validate data types
    # 6. Sort by date and ticker
    
    raise NotImplementedError("Function to be implemented")


def load_prices(tickers: Union[str, List[str]], 
                start_date: str,
                end_date: Optional[str] = None,
                interval: str = '1d') -> pd.DataFrame:
    """
    Download stock price data from Yahoo Finance.
    
    Parameters
    ----------
    tickers : str or list of str
        Stock ticker symbol(s) to download
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses current date
    interval : str, default='1d'
        Data interval: '1d', '1wk', '1mo', etc.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with price data (Open, High, Low, Close, Adj Close, Volume)
        MultiIndex: (Date, Ticker) if multiple tickers
        
    Examples
    --------
    >>> prices = load_prices(['AAPL', 'MSFT'], start_date='2020-01-01')
    >>> print(prices.head())
    
    >>> single_stock = load_prices('AAPL', start_date='2020-01-01', end_date='2021-12-31')
    """
    # TODO: Implement Yahoo Finance data download
    # 1. Convert tickers to list if string
    # 2. Download data using yfinance
    # 3. Handle multiple tickers (MultiIndex)
    # 4. Handle missing data
    # 5. Validate date range
    # 6. Return clean DataFrame
    
    raise NotImplementedError("Function to be implemented")


def load_spy_benchmark(start_date: str,
                       end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load SPY (S&P 500 ETF) benchmark data from Yahoo Finance.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses current date
        
    Returns
    -------
    pd.DataFrame
        DataFrame with SPY price data and returns
        Columns: ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return']
        
    Examples
    --------
    >>> spy = load_spy_benchmark(start_date='2020-01-01')
    >>> print(spy.head())
    """
    # TODO: Implement SPY benchmark loading
    # 1. Download SPY data
    # 2. Calculate returns
    # 3. Handle missing data
    # 4. Return clean DataFrame
    
    raise NotImplementedError("Function to be implemented")


def merge_fundamentals_prices(fundamentals: pd.DataFrame,
                               prices: pd.DataFrame,
                               how: str = 'inner') -> pd.DataFrame:
    """
    Merge fundamental data with price data.
    
    Parameters
    ----------
    fundamentals : pd.DataFrame
        Fundamental data with (date, ticker) index
    prices : pd.DataFrame
        Price data with (date, ticker) index
    how : str, default='inner'
        Type of merge: 'inner', 'outer', 'left', 'right'
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with both fundamentals and prices
        
    Notes
    -----
    - Handles different frequencies (quarterly fundamentals, daily prices)
    - Forward-fills fundamental data to match daily price dates
    - Ensures no look-ahead bias
    
    Examples
    --------
    >>> merged = merge_fundamentals_prices(fundamentals, prices)
    """
    # TODO: Implement merge logic
    # 1. Align dates (forward-fill fundamentals)
    # 2. Merge on date and ticker
    # 3. Handle missing values
    # 4. Validate no look-ahead bias
    
    raise NotImplementedError("Function to be implemented")


def validate_data(df: pd.DataFrame,
                  required_columns: List[str],
                  date_column: str = 'date') -> bool:
    """
    Validate that DataFrame has required structure and columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list of str
        List of required column names
    date_column : str, default='date'
        Name of the date column to validate
        
    Returns
    -------
    bool
        True if validation passes, raises ValueError otherwise
        
    Raises
    ------
    ValueError
        If required columns are missing or data types are incorrect
        
    Examples
    --------
    >>> validate_data(df, required_columns=['ticker', 'close', 'volume'])
    True
    """
    # TODO: Implement validation logic
    # 1. Check required columns exist
    # 2. Check date column is datetime
    # 3. Check for duplicates
    # 4. Check for missing values in key columns
    # 5. Raise informative errors
    
    raise NotImplementedError("Function to be implemented")


class DataLoader:
    """
    Class for loading and managing financial data.
    
    Attributes
    ----------
    fundamentals : pd.DataFrame
        Loaded fundamental data
    prices : pd.DataFrame
        Loaded price data
    spy : pd.DataFrame
        SPY benchmark data
        
    Methods
    -------
    load_all(fundamentals_path, tickers, start_date, end_date)
        Load all required data
    get_merged_data()
        Get merged fundamentals and prices
    """
    
    def __init__(self):
        """Initialize DataLoader."""
        self.fundamentals = None
        self.prices = None
        self.spy = None
        
    def load_all(self,
                 fundamentals_path: str,
                 tickers: List[str],
                 start_date: str,
                 end_date: Optional[str] = None) -> None:
        """
        Load all required data (fundamentals, prices, SPY).
        
        Parameters
        ----------
        fundamentals_path : str
            Path to fundamentals CSV file
        tickers : list of str
            List of stock tickers
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        """
        # TODO: Implement loading all data sources
        raise NotImplementedError("Method to be implemented")
        
    def get_merged_data(self) -> pd.DataFrame:
        """
        Get merged fundamentals and prices.
        
        Returns
        -------
        pd.DataFrame
            Merged data
        """
        # TODO: Implement merge
        raise NotImplementedError("Method to be implemented")
