# 🚀 Hull Tactical Market Prediction Engine

<img width="1120" height="280" alt="Market Prediction Visualization" src="https://github.com/user-attachments/assets/72ecd01f-e354-458a-a3fa-1ffa48a5cb1a" />

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)](https://lightgbm.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Inference%20Active-orange)]()

An end-to-end ensemble LightGBM system that **directly predicts optimal S&P 500 positions** under volatility and return penalties for the [Hull Tactical Kaggle Competition](https://www.kaggle.com/competitions/hull-tactical-market-prediction).

## 🎯 Business Problem

*"Wisdom from most personal finance experts would suggest that it's irresponsible to try and time the market... But in the age of machine learning, is it irresponsible to not try?"*

The Hull Tactical competition challenges the Efficient Market Hypothesis by evaluating models on a modified Sharpe ratio. The aim is to build a model that predicts excess returns and includes a betting strategy designed to outperform the S&P 500 while staying within a 120% volatility constraint.

## 🏗 Solution Architecture

### 1. Custom Target Engineering (The "Secret Sauce")
Instead of predicting raw returns (regression), I formulated a **Risk-Adjusted Optimal Position Target** (classification) for supervised learning. This target incorporates:
*   Forward-looking market returns.
*   Under-performance and volatility penalties.
*   Aligns the model's loss function with the competition's specific Sharpe Ratio objective.

### 2. Ensemble LightGBM Strategy
*   **Dual-Window Training:** Trained two independent models on different time horizons to capture both long-term regime stability and short-term market shifts.
*   **Hyperparameter Tuning:** Optimized using **Optuna** to maximize generalization on out-of-sample data.
*   **Ensemble Logic:** Final trading signal is a weighted ensemble of the two models, reducing variance and smoothing signal noise.
*   **Adaptive Retraining:** Implemented a periodic retraining loop to allow the model to adapt to new market regimes during the inference phase.

### 3. Feature Engineering
Enhanced the provided 98 "black-box" features with domain-specific technical indicators:
*   **Momentum Indicators:** Capturing price velocity and trend strength.
*   **RSI (Relative Strength Index):** Identifying overbought/oversold regimes.
*   **Lagged Return Patterns:** Encoding historical market memory.

## 🛠 Technical Skills Demonstrated

*   **Quantitative Risk Modeling:** Formulating custom loss functions that penalize volatility (Sharpe-aware learning).
*   **Ensemble ML Engineering:** Designing a multi-model system with time-window diversity to improve robustness.
*   **Automated Retraining Pipelines:** Building a system that updates itself with new data, simulating a live trading environment.
*   **High-Performance ML:** Utilizing LightGBM for efficient handling of high-dimensional feature spaces (9000+ samples trained in <5 mins).

## 📊 Results

*   **Status:** Inference in progress (Live Testing Period).
*   **Current Performance:** Achieved a **Sharpe Ratio of 3.82** on the initial 1-month validation period.
*   **Impact:** The model successfully navigates volatility by adjusting exposure, demonstrating the effectiveness of the "Direct Position" target variable. Results will be more meaningful for the full testing period (6-months)

## 📂 Dataset & Structure

**Data Source:** [Hull Tactical Competition](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
*   98 Features: Market dynamics, Macro-economic indicators, Proprietary signals.
*   Target: S&P 500 daily returns (transformed into optimal position targets).

```text
├── notebooks/                  # Feature Engineering & Training Logic
├── src/                        # Inference pipeline scripts
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
