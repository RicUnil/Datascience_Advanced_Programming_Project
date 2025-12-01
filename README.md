# Datascience_Advanced_Programming_Project# Linking Pre-Earnings Fundamentals to Post-Earnings Market Reaction Using Machine Learning

**HEC Lausanne - MSc Finance**  
**Data Science & Advanced Programming - Final Project**  
**Academic Year: 2025-2026**

---

## Project Overview

This project builds a complete machine learning pipeline to predict post-earnings market reactions based on pre-earnings fundamental data. The analysis combines:

- **Firm fundamentals** from Capital IQ
- **Stock prices and benchmark data** from Yahoo Finance
- **Feature engineering** (momentum, volatility, valuation ratios)
- **Label construction** (excess returns, classification targets)
- **Baseline models** (historical mean, CAPM)
- **ML models** (Logistic Regression, Random Forest)
- **Event-driven backtesting**

---

## Project Structure

```
project/
│
├── README.md                  # This file
├── PROPOSAL.md                # Project proposal
├── AI_USAGE.md                # Documentation of AI assistance
├── requirements.txt           # Python dependencies
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data/                  # Data loading and processing
│   │   ├── __init__.py
│   │   ├── load_data.py       # Load Capital IQ and Yahoo Finance data
│   │   ├── build_features.py  # Feature engineering
│   │   └── build_labels.py    # Label construction
│   │
│   ├── models/                # ML models
│   │   ├── __init__.py
│   │   ├── classifier.py      # Classification models
│   │   └── regressor.py       # Regression models
│   │
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── metrics.py         # Evaluation metrics
│       └── plotting.py        # Visualization functions
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data files
│
├── notebooks/                 # Jupyter notebooks
│   └── 01_exploration.ipynb   # Exploratory data analysis
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_data_pipeline.py  # Tests for data pipeline
│   └── test_models.py         # Tests for models
│
├── results/                   # Model outputs and results
│   ├── figures/               # Plots and visualizations
│   └── models/                # Saved model artifacts
│
└── docs/                      # Additional documentation
    └── methodology.md         # Detailed methodology

```

---

## Installation

1. **Clone the repository** (or navigate to project directory)

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Loading
```python
from src.data.load_data import load_fundamentals, load_prices

# Load firm fundamentals from Capital IQ
fundamentals = load_fundamentals('data/raw/fundamentals.csv')

# Load stock prices from Yahoo Finance
prices = load_prices(tickers=['AAPL', 'MSFT'], start_date='2020-01-01')
```

### 2. Feature Engineering
```python
from src.data.build_features import build_features

# Compute pre-earnings features
features = build_features(fundamentals, prices)
```

### 3. Label Construction
```python
from src.data.build_labels import build_labels

# Compute post-earnings excess returns
labels = build_labels(prices, earnings_dates)
```

### 4. Model Training
```python
from src.models.classifier import EarningsClassifier
from src.models.regressor import EarningsRegressor

# Train classification model
clf = EarningsClassifier(model_type='random_forest')
clf.fit(X_train, y_train)

# Train regression model
reg = EarningsRegressor(model_type='random_forest')
reg.fit(X_train, y_train)
```

### 5. Evaluation
```python
from src.utils.metrics import evaluate_classifier, evaluate_regressor

# Evaluate models
clf_metrics = evaluate_classifier(clf, X_test, y_test)
reg_metrics = evaluate_regressor(reg, X_test, y_test)
```

---

## Data Sources

- **Capital IQ**: Firm fundamentals (balance sheet, income statement, cash flow)
- **Yahoo Finance**: Daily stock prices and SPY benchmark returns

---

## Methodology

1. **Data Collection**: Load fundamentals and price data
2. **Feature Engineering**: Compute momentum, volatility, valuation ratios
3. **Label Construction**: Calculate excess returns (stock - SPY)
4. **Baseline Models**: Historical mean, CAPM expected return
5. **ML Models**: Logistic Regression, Random Forest
6. **Evaluation**: Classification metrics, regression metrics, backtesting
7. **Backtesting**: Event-driven strategy simulation

---

## Testing

Run unit tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_data_pipeline.py -v
```

---

## Results

Results, figures, and saved models are stored in the `results/` directory.

---

## License

This project is submitted as part of the HEC Lausanne MSc Finance curriculum.

---

## Contact

**Author**: Ricardo Guerreiro  
**Institution**: HEC Lausanne  
**Program**: MSc Finance  
**Course**: Data Science & Advanced Programming
