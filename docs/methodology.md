# Methodology

## Overview

This document describes the detailed methodology for predicting post-earnings market reactions using machine learning.

---

## 1. Data Collection

### 1.1 Firm Fundamentals (Capital IQ)
- **Source**: Capital IQ database
- **Frequency**: Quarterly
- **Variables**:
  - Balance sheet items (assets, liabilities, equity)
  - Income statement items (revenue, expenses, net income)
  - Cash flow items (operating, investing, financing cash flows)
  
### 1.2 Stock Prices (Yahoo Finance)
- **Source**: Yahoo Finance API via `yfinance`
- **Frequency**: Daily
- **Variables**:
  - Open, High, Low, Close, Adjusted Close
  - Volume
  
### 1.3 Benchmark (SPY)
- **Source**: Yahoo Finance
- **Purpose**: Calculate excess returns
- **Data**: Daily prices and returns

---

## 2. Feature Engineering

### 2.1 Fundamental Ratios
- **Valuation**: P/E, P/B, P/S, P/CF
- **Profitability**: ROE, ROA, profit margin, operating margin
- **Leverage**: Debt-to-equity, debt-to-assets
- **Efficiency**: Asset turnover, inventory turnover

### 2.2 Momentum Indicators
- **Short-term**: 1-month (21 trading days) return
- **Medium-term**: 3-month (63 trading days) return
- **Long-term**: 6-month (126 trading days) return

### 2.3 Volatility Measures
- **Historical volatility**: Rolling standard deviation of returns (annualized)
- **Beta**: Rolling covariance with market / market variance

### 2.4 Growth Metrics
- **Revenue growth**: Year-over-year and quarter-over-quarter
- **Earnings growth**: Year-over-year and quarter-over-quarter

### 2.5 Liquidity Metrics
- **Average volume**: Rolling average trading volume
- **Dollar volume**: Price × Volume

---

## 3. Label Construction

### 3.1 Excess Returns
- **Formula**: Excess Return = Stock Return - SPY Return
- **Horizons**: 3-day, 5-day, 10-day post-earnings
- **Calculation**: From close on earnings date to close on horizon date

### 3.2 Classification Labels
- **Binary**: Positive (1) vs. Negative/Zero (0) excess return
- **Multi-class**: Strong Negative (0), Neutral (1), Strong Positive (2)
  - Thresholds: [-2%, +2%]

### 3.3 Regression Labels
- **Continuous excess returns**
- **Winsorization**: Cap extreme values at 1st and 99th percentiles

---

## 4. Baseline Models

### 4.1 Historical Mean
- **Method**: Average historical post-earnings excess return per stock
- **Purpose**: Naïve baseline for comparison

### 4.2 CAPM Expected Return
- **Formula**: E[R] = Rf + β × (E[Rm] - Rf)
- **Purpose**: Risk-adjusted baseline

---

## 5. Machine Learning Models

### 5.1 Logistic Regression
- **Type**: Linear classification
- **Regularization**: L2 (Ridge)
- **Purpose**: Interpretable baseline

### 5.2 Random Forest
- **Type**: Ensemble of decision trees
- **Parameters**: To be tuned via GridSearchCV
- **Purpose**: Capture non-linear relationships

---

## 6. Model Training

### 6.1 Train-Test Split
- **Method**: Time-series split (no shuffling)
- **Train**: First 70% of data
- **Test**: Last 30% of data

### 6.2 Cross-Validation
- **Method**: Time-series cross-validation
- **Folds**: 5
- **Purpose**: Hyperparameter tuning and model selection

### 6.3 Feature Scaling
- **Method**: StandardScaler (z-score normalization)
- **Applied to**: All numerical features

---

## 7. Evaluation

### 7.1 Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

### 7.2 Regression Metrics
- **MSE**: Mean squared error
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination
- **Hit Ratio**: Directional accuracy

### 7.3 Financial Metrics
- **Sharpe Ratio**: Risk-adjusted return
- **Information Ratio**: Excess return per unit of tracking error
- **Maximum Drawdown**: Largest peak-to-trough decline

---

## 8. Backtesting

### 8.1 Strategy
- **Long**: Stocks predicted to have positive excess returns
- **Short**: Stocks predicted to have negative excess returns (optional)
- **Position sizing**: Equal-weighted

### 8.2 Transaction Costs
- **Assumption**: 0.1% per trade (10 basis points)
- **Applied**: On entry and exit

### 8.3 Performance Evaluation
- **Cumulative returns**
- **Sharpe ratio**
- **Maximum drawdown**
- **Win rate**

---

## 9. Validation and Robustness

### 9.1 Out-of-Sample Testing
- **Strict temporal separation** between train and test sets
- **No data leakage**

### 9.2 Sensitivity Analysis
- **Feature importance**: Identify key drivers
- **Threshold sensitivity**: Test different classification thresholds
- **Time period sensitivity**: Test across different market regimes

---

## 10. Limitations

- **Survivorship bias**: Data may exclude delisted firms
- **Look-ahead bias**: Must ensure features use only past information
- **Market regime changes**: Model performance may vary over time
- **Transaction costs**: Real-world costs may be higher
- **Liquidity constraints**: May not be able to trade all predicted positions

---

## References

- Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns.
- Ball, R., & Brown, P. (1968). An empirical evaluation of accounting income numbers.
- [Additional references to be added]
