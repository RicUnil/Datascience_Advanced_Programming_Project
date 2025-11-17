# Project Proposal – Linking Pre-Earnings Fundamentals to Market Reaction  
## Advanced Programming / Data Science — Individual Project 2025  
**Student:** Contente Ricardo  
**Category:** Data Analysis & Visualization + Business & Finance Tools / Statistical Analysis Tools**

---

## 1. Project Title  
**Linking Pre-Earnings Fundamentals to Post-Earnings Market Reaction Using Machine Learning**

---

## 2. Motivation & Problem Statement  

Quarterly earnings announcements are among the most impactful events in equity markets. Prices often react sharply when new financial results are released, yet empirical research shows that markets do not always adjust immediately or efficiently. The **Post-Earnings Announcement Drift (PEAD)** suggests that stocks with positive surprises often continue to outperform the market after earnings, while negative surprises tend to underperform in the following weeks.

This project examines whether **publicly available information before the earnings announcement** contains predictive power about how the market will react afterward. The goal is not to forecast the earnings numbers themselves but to study whether a combination of **firm fundamentals** (revenue/EPS trends, margins, leverage, returns on capital) and **market-based signals** (momentum, volatility, valuation ratios) can help anticipate a stock’s **30-day post-earnings excess return** relative to the SPY benchmark.

The key question is:  
**Can pre-earnings characteristics help predict whether a stock will overperform or underperform the market after reporting its results?**

---

## 3. Data Sources  

**Capital IQ (S&P Global)** will provide quarterly earnings dates and financial statement data: revenue, EPS, net income, margins, assets, debt, equity, free cash flow, leverage ratios.

**Yahoo Finance (`yfinance`)** will provide daily stock prices and SPY benchmark data. From these series, I will compute:  
- 1m/3m/6m pre-earnings momentum,  
- 30-day pre-earnings volatility,  
- 30-day post-earnings returns,  
- excess returns relative to SPY.

All data used will be observable before the announcement to avoid look-ahead bias.

---

## 4. Planned Methodology  

For each firm and each earnings event, I will construct:  
- a **feature vector** combining pre-announcement fundamentals and market indicators;  
- a **regression label**: 30-day excess return;  
- a **classification label**: 1 = outperform, 0 = underperform.

Following Francesco’s feedback, the **null hypothesis (H₀)** is that excess returns are unpredictable. Under H₀, naïve forecasts should perform as well as any model. I will therefore include two baselines:  
- **Historical mean excess return** (computed over the training period),  
- **CAPM-based expected excess return**, using betas estimated on the training sample.

I will train supervised models:  
- **Classification**: Logistic Regression, Random Forest Classifier  
- **Regression**: Ridge Regression, Random Forest Regressor  

A **time-based train/test split** (train ≤ 2019, test ≥ 2020) will simulate realistic forecasting conditions and avoid leakage.

Evaluation will rely on accuracy, ROC-AUC, RMSE, MAE, feature importance, and economic interpretation. A simple backtest will assess whether buying predicted outperformers beats the SPY benchmark.

---

## 5. Expected Contribution & Challenges  

This project builds a complete, reproducible ML pipeline linking pre-earnings firm characteristics to post-earnings market reaction. It tests whether any structure exists beyond naïve baselines. Challenges include aligning Capital IQ and Yahoo Finance data, handling missing observations, and dealing with noise in financial returns.

Success will be defined by:  
- a clean end-to-end implementation,  
- meaningful comparisons to naïve baselines,  
- interpretable results,  
- and documentation aligned with the course rulebook.
