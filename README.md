# Hull Tactical Market Prediction - Direct Position Optimization with LightGBM

End-to-end LightGBM model that **directly predicts optimal S&P 500 positions** under volatility and return penalties for the [Hull Tactical Kaggle Competition](https://www.kaggle.com/competitions/hull-tactical-market-prediction).

## Business Problem

The Hull Tactical competition evaluates models using a modified Sharpe ratio that penalizes both volatility and suboptimal returns. Traditional approaches predict returns first, then convert to positions - introducing a disconnect between prediction and the actual objective function.

## Key Innovation: Direct Position Prediction

### Why This Approach is Superior

Rather than the conventional two-step process:
1. ~~Predict excess returns~~
2. ~~Convert predictions to positions using a separate optimization~~

**My model directly predicts optimal trading positions**, training LightGBM to output position sizes that account for:
- **Return penalties** - opportunity cost of being under-invested or wrongly positioned
- **Volatility penalties** - risk from excessive position concentration
- **Competition's modified Sharpe ratio** - aligning the model objective with evaluation metric

This **end-to-end optimization** ensures the model learns the actual trading objective, not a proxy.

### Technical Implementation

**Custom Target Engineering:**
- Formulated optimal position as the training target by incorporating:
  - Forward-looking returns
  - Volatility constraints from historical data
  - Risk-adjusted position sizing rules
  
**LightGBM Training:**
- Trained directly on optimal position labels
- Model learns the complex relationship between market features and risk-adjusted positioning
- Outputs actionable trading signals without post-processing

**Feature Engineering:**
Working with blackbox market features, I enhanced the signal with:
- **RSI (Relative Strength Index)** - momentum oscillator for regime identification
- **Momentum features** - capturing price trends and velocity on lagged targets
- **Lagged target-derived indicators** - incorporating historical return patterns

## Why This Matters

This approach demonstrates:
- **Quantitative sophistication** - understanding that the loss function should align with the business objective
- **Production thinking** - models should output decisions, not intermediate predictions requiring manual intervention
- **Risk management expertise** - incorporating volatility constraints at the modeling stage, not as an afterthought

In real trading systems, you want models that produce **risk-adjusted positions directly**, not raw forecasts that require separate position sizing logic.

## Technical Skills Demonstrated

- **Custom target engineering** for risk-adjusted objectives
- **Time-series modeling** with financial market data (98 features across 7 factor families)
- **Feature engineering** on technical indicators
- **LightGBM** implementation for regression
- **Quantitative portfolio optimization** principles embedded in ML

## Model: LightGBM

Selected for:
- Superior performance on tabular financial data
- Efficient handling of high-dimensional feature spaces (98 features) 
- Fast training on 8,990 time-series samples 
- Flexible objective functions for custom targets

## Results

To be confirmed

## Dataset

Hull Tactical competition data: 98 features spanning Market dynamics, Macro-economic indicators, and proprietary signals. Decades of S&P 500 market information provided via Kaggle.

**Competition Link:** https://www.kaggle.com/competitions/hull-tactical-market-prediction

## Project Structure
├── notebooks/ # Jupyter notebook with full analysis
├── requirements.txt # Python dependencies
└── README.md
