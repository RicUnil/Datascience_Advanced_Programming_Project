from typing import List, Tuple

import pandas as pd


def add_labels(
    df: pd.DataFrame,
    er_col: str = "er_30d",
    pos_threshold: float = 0.02,
    neg_threshold: float = -0.02,
) -> pd.DataFrame:
    """
    Add regression and classification labels to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that must contain a column with excess returns.
    er_col : str
        Name of the excess return column (e.g. er_30d).
    pos_threshold : float
        Threshold above which a stock is considered an outperformer.
    neg_threshold : float
        Threshold below which a stock is considered an underperformer.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with two new columns:
        - y_reg: continuous excess return
        - y_class: 1 for outperform, 0 for underperform.
          Rows between thresholds are dropped for classification.
    """
    df = df.copy()

    if er_col not in df.columns:
        raise KeyError(f"Column '{er_col}' not found in dataframe.")

    # Regression label = excess return itself
    df["y_reg"] = df[er_col]

    # Classification label
    def classify(er: float):
        if er > pos_threshold:
            return 1
        elif er < neg_threshold:
            return 0
        else:
            return None

    df["y_class"] = df[er_col].apply(classify)

    # Remove neutral cases (close to zero excess return)
    df = df.dropna(subset=["y_class"])
    df["y_class"] = df["y_class"].astype(int)

    return df


def get_features_and_labels(
    df: pd.DataFrame,
    feature_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataframe into feature matrix X and labels y_class, y_reg.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe returned by `add_labels`.
    feature_cols : list of str, optional
        List of feature column names. If None, defaults to a simple set
        of columns suitable for the fake dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y_class : pd.Series
        Binary labels (1/0).
    y_reg : pd.Series
        Continuous labels (excess returns).
    """
    if feature_cols is None:
        feature_cols = [
            "rev_growth",
            "eps_growth",
            "roe",
            "leverage",
            "momentum_3m",
            "vol_30d",
        ]

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in dataframe: {missing}")

    X = df[feature_cols].copy()
    y_class = df["y_class"].copy()
    y_reg = df["y_reg"].copy()

    return X, y_class, y_reg
