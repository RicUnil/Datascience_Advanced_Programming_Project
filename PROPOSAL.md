# Project Proposal  
**Title:** Detecting Speculative Phases in AI-Related Assets Using Machine Learning  
**Category:** Supervised Machine Learning / Financial Data Science  

---

## Problem Statement and Motivation  

Over the past years, financial markets have experienced strong speculation around artificial intelligence (AI).  
Several analysts, including the IMF and major investment banks, have warned that a potential “AI bubble” could be forming, driven by high valuations and massive capital inflows toward tech and AI-related firms such as NVIDIA, Microsoft, and Meta.  

The goal of this project is to apply machine learning methods to **detect speculative phases in a portfolio of AI-related assets**.  
By identifying market regimes that exhibit characteristics of speculative behavior—such as abnormally high returns, volatility spikes, and increased trading volumes—the project aims to better understand how machine learning can be used to monitor financial bubbles and risk patterns.  

This topic is both **financially relevant** (market behavior, portfolio risk) and **technically aligned** with the course objectives, as it combines data collection, feature engineering, and supervised classification using real-world data.

---

## Planned Approach and Technologies  

1. **Data Collection:**  
   - Retrieve daily market data (prices, volumes, volatility) for a selection of AI-related stocks and ETFs (e.g., NVDA, MSFT, META, AIQ, ARKQ, QQQ) from Yahoo Finance using the `yfinance` API.  
   - Build an equal-weighted “AI portfolio” for analysis.  

2. **Feature Engineering:**  
   - Compute technical indicators: returns, rolling volatility, RSI, moving averages, and momentum.  
   - Create a binary target variable indicating whether a given period is “speculative” or “normal” (based on return and volume thresholds).  

3. **Machine Learning Models:**  
   - Train supervised classifiers such as Decision Trees, Random Forest, k-Nearest Neighbors, and AdaBoost from `scikit-learn`.  
   - Evaluate with cross-validation and standard metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix).  

4. **Interpretation:**  
   - Compare detected speculative phases with actual market events (e.g., the 2023 AI rally).  
   - Visualize results with `matplotlib` to highlight speculative vs. normal periods.

---

## Expected Challenges and How I’ll Address Them  

- **Defining the “speculative” label:** Choosing consistent thresholds for abnormal price and volume movements may be subjective. I will test multiple definitions (percentile-based, z-score-based) and validate their stability.  
- **Overfitting risk:** I will use cross-validation, hold-out sets, and regularization where possible to ensure generalization.  
- **Correlated features:** I will apply feature scaling and correlation checks to avoid redundant inputs.  

---

## Success Criteria  

The project will be considered successful if:  
- The model achieves an F1-score above 0.70 in classifying speculative vs. normal phases.  
- The output patterns visually align with known market events and plausible speculative behavior.  
- The entire pipeline (data → model → evaluation) is reproducible and well-documented on GitHub.

---

## Stretch Goals  

If time permits:  
- Extend the analysis to a **multiclass classification** (e.g., normal / speculative / correction).  
- Compare the AI portfolio with a benchmark index (e.g., S&P 500) to quantify excess volatility.  
- Experiment with additional ML techniques such as Gradient Boosting or feature importance interpretation (SHAP values).

---

**Author:** Ricardo Contente Guerreiro  
**Course:** Data Science & Advanced Programming (HEC Lausanne, MSc Finance)  
**Date:** November 2025
