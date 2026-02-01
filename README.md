# Hull Tactical Market Prediction - Direct Position Optimization with LightGBM

End-to-end, ensemble LightGBM model that **directly predicts optimal S&P 500 positions** under volatility and return penalties for the [Hull Tactical Kaggle Competition](https://www.kaggle.com/competitions/hull-tactical-market-prediction).

## Business Problem

"Wisdom from most personal finance experts would suggest that it's irresponsible to try and time the market. The Efficient Market Hypothesis (EMH) would agree: everything knowable is already priced in, so don’t bother trying.

But in the age of machine learning, is it irresponsible to not try and time the market? Is the EMH an extreme oversimplification at best and possibly just…false?"

The Hull Tactical competition evaluates models using a modified Sharpe ratio that penalizes both volatility and suboptimal returns. Traditional approaches predict returns first, then convert to positions - introducing a disconnect between prediction and the actual objective function.

## Approach

**Custom Target Engineering:**
- Formulated an a risk-adjusted "optimal position" target for supervised learning by considering forward looking market returns, as well as volatility and under-performance penalties
  
**LightGBM Training:**
- Hyper-parameter tuned LightGMB with Optuna on the formulated "optimal position" labels
- Trained 2 models on 2 different time-windows (hyper-parameters optimized individually)
- Use the ensemble of the 2 models to predict the final trading signal - reducing noise and improve generalization
- Retrain one of the ensemble models periodically to adapt to additional / changing information

**Feature Engineering:**
Working mainly with blackbox market features, I further enhanced the feature universe by engineering simple technical indicator features for lagged target variables:
- **RSI (Relative Strength Index)** - momentum oscillator for regime identification
- **Momentum features** - capturing price trends and velocity on lagged targets
- **Lagged target-derived indicators** - incorporating historical return patterns

## Technical Skills Demonstrated

- **Custom target engineering** for risk-adjusted objectives
- **Time-series modeling** with financial market data (98 features across 7 factor families)
- **Feature engineering** on technical indicators
- **LightGBM** Hyper-parameter tuning and training for a classification problem
- **Quantitative portfolio optimization** An retraniable ML approach to estimate optimal market postions on a live inference environment

## Model: LightGBM

Selected for:
- Superior performance on tabular financial data
- Efficient handling of high-dimensional feature spaces (98 features) 
- Fast training on ~9000 daily market observations with 100+ features across macro, technical, and fundamental categories. This allows retraining the model in the inference time, under 5 minutes.
- Flexible objective functions for custom targets

## Results
- Inference in progress!  Current Sharpe on a small data period (1-month): 3.82

## Dataset

Hull Tactical competition data: 98 features spanning Market dynamics, Macro-economic indicators, and proprietary signals. Decades of S&P 500 market information provided via Kaggle.

**Competition Link:** https://www.kaggle.com/competitions/hull-tactical-market-prediction

## Project Structure
├── notebooks/ # Jupyter notebook with full analysis

├── requirements.txt # Python dependencies

└── README.md
