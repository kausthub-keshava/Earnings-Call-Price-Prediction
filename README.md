# Earnings-Call-Price-Prediction

## Overview
This project predicts short-term stock price movements (1-day and 5-day horizons) following earnings announcements by analyzing earnings call transcripts combined with financial metrics.

## Motivation
Earnings calls contain valuable qualitative information (management tone, forward guidance) that may not be captured in quantitative metrics alone. This project tests whether NLP can extract predictive signals for trading strategies.

## Data
- **Transcripts**: Motley Fool earnings call transcripts
- **Financial Data**: S&P 500 price data, Yahoo finance
- **Time Period**: 
- **Sample Size**: 

## Methodology

### Target Variable
Binary classification: Market-adjusted stock price increases vs decreases N days post-earnings 
- 1-day horizon: (Price_t+1 - Price_t) / Price_t > 0
- 5-day horizon: (Price_t+5 - Price_t) / Price_t > 0

### Feature Engineering
**Text Features:**
- TF-IDF: 
- FinBERT Embeddings: 

**Financial Features:**
- 

### Models Compared
1. Random Baseline
2. Logistic Regression (financial features only)
3. Logistic Regression + TF-IDF
4. Neural Network + FinBERT (mean pooling)
5. Neural Network + FinBERT (attention pooling)
6. Neural Network + FinBERT + financial features
7. SVM + FinBERT
8. SVM + FinBERT + financial features

### Validation
- Time-based train/validation/test splits (prevents lookahead bias)
- Evaluation metric: AUC-ROC with 95% confidence intervals (bootstrap)

## Results


### Key Findings
- 
## Limitations
- Bootstrap confidence intervals assume IID samples (violated in financial time series - consider block bootstrap)
- 
## Future Work
- Incorporate sentiment scores
- Test on different market periods (bull vs bear)
- Extend to longer time horizons
- Add technical indicators

## Setup & Usage
Installation instructions:
How to run the code:

## References
