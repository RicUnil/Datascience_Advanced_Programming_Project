# PROPOSAL.md

## Project Title
**A Comparative Analysis of European and U.S. Defense ETFs: Risk, Return, and Geopolitical Sensitivity (2015–2025)**

## Category
Financial Data Science / Time-Series Analysis

## Problem Statement & Motivation
In a world increasingly shaped by geopolitical tensions, the Defense industry has become one of the most strategically significant sectors of the global economy. Recent events such as the 2022 invasion of Ukraine, rising NATO defense budgets, and renewed conflicts in the Middle East have driven unprecedented market attention toward defense-related assets.

The objective of this project is to analyze and compare the financial performance and risk dynamics of the **European** and **U.S. Defense sectors** through two Exchange-Traded Funds (ETFs):

- *Invesco Aerospace & Defense UCITS ETF (DFND.L)* — representing the European and international defense market  
- *iShares U.S. Aerospace & Defense ETF (ITA)* — representing the U.S. defense sector  

By studying these ETFs over the 2015–2025 period, I aim to understand how the defense industry’s performance correlates with major geopolitical shocks, and whether European and American defense markets react differently to global instability.

## Planned Approach & Technologies
The analysis will be implemented in **Python**, primarily using:
- `pandas`, `numpy` for data manipulation and returns computation  
- `matplotlib` or `plotly` for visualizations  
- `yfinance` for retrieving historical price data  
- `scipy` for statistical computations and correlation testing  

The workflow will include:
1. **Data Collection and Cleaning** — download adjusted close prices for both ETFs (2015–2025), handle missing values, align trading days  
2. **Descriptive and Risk Analysis** — compute daily and annualized returns, volatility, Sharpe ratio, maximum drawdown, and rolling risk metrics  
3. **Event Sensitivity Study** — define ±20-day event windows around key geopolitical dates (e.g., Ukraine war, Brexit, NATO announcements) and compute cumulative abnormal returns (CAR)  
4. **Visualization and Interpretation** — generate clean, readable plots showing cumulative returns, rolling volatility, Sharpe ratios, drawdowns, and event reactions  

## Expected Challenges & Mitigation
- **Data limitations**: some ETFs (e.g., DFND.L) may have incomplete historical data before 2015. I will document gaps and, if necessary, use alternative tickers or reduced time windows  
- **Event window sensitivity**: defining the exact event impact period may introduce noise. To address this, I will test multiple window sizes (±10, ±20, ±30 days) for robustness  
- **Market comparability**: differences in trading hours and liquidity between U.S. and European ETFs will be acknowledged in the interpretation  

## Success Criteria
The project will be considered successful if:
1. A reproducible Python workflow runs end-to-end without errors  
2. The analysis produces clear, interpretable visualizations of risk and performance  
3. Event windows show meaningful differences between European and U.S. reactions  
4. The final report connects quantitative results with an economic narrative on geopolitical risk  

## Stretch Goals (if time permits)
- Extend the study to include a **Bloomberg European Defense Index** for validation  
- Estimate **beta coefficients** of each ETF relative to a broad market index (e.g., STOXX 600 or S&P500)  
- Build a simple **Monte Carlo simulation** to project future performance scenarios under varying volatility regimes  