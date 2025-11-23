from pathlib import Path
from typing import Union

import pandas as pd


# Base directory for data files
DATA_DIR = Path("data")


def load_fake_earnings_data(
    filename: Union[str, Path] = "processed/fake_earnings_sample.csv",
) -> pd.DataFrame:
    """
    Load a small fake earnings dataset for testing the pipeline.

    Parameters
    ----------
    filename : str or Path
        Relative path inside the `data/` directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns such as ticker, date_earnings,
        rev_growth, eps_growth, roe, leverage, momentum_3m,
        vol_30d, er_30d.
    """
    file_path = DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Fake earnings file not found at: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["date_earnings"])
    return df


def load_fundamentals(path: Union[str, Path]) -> pd.DataFrame:
    """
    Placeholder for the real Capital IQ fundamentals loader.

    For now, if `path` points to the fake CSV, this just calls
    `load_fake_earnings_data`. Later, this function will be extended
    to handle real Capital IQ exports.
    """
    path = Path(path)
    if path.name == "fake_earnings_sample.csv":
        # Allow both "data/processed/..." and just "fake_earnings_sample.csv"
        rel_path = path if "data" in str(path) else Path("processed") / path.name
        return load_fake_earnings_data(rel_path)
    else:
        # Later: implement real logic for Capital IQ exports
        raise NotImplementedError(
            "load_fundamentals for real Capital IQ data is not implemented yet."
        )


def load_prices(*args, **kwargs):
    """
    Placeholder for future price loader using Yahoo Finance (yfinance).

    This will later download daily prices for individual tickers over a
    specified date range.
    """
    raise NotImplementedError("load_prices will be implemented once price data is ready.")


def load_spy_benchmark(*args, **kwargs):
    """
    Placeholder for future SPY benchmark loader using Yahoo Finance.

    This will later download daily prices for SPY and compute benchmark
    returns over the same windows as the individual stocks.
    """
    raise NotImplementedError("load_spy_benchmark will be implemented once data is ready.")
