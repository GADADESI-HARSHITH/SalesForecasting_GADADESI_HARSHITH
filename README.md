# Sales Forecasting & Demand Intelligence System

## Project Overview

This project develops an end-to-end sales forecasting and demand intelligence system using the Superstore Sales dataset.

The system includes:

- Time Series Analysis
- Sales Forecasting
- Product Demand Segmentation
- Sales Anomaly Detection
- Interactive Streamlit Dashboard

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels (SARIMA)
- Facebook Prophet
- XGBoost
- Scikit-learn
- Streamlit

---

## Models Implemented

### 1. SARIMA
Statistical forecasting model capturing trend and seasonality.

### 2. Facebook Prophet
Forecasting model developed by Meta with automatic trend and seasonal decomposition.

### 3. XGBoost
Machine Learning regression model using lag features and rolling statistics.

---

## Additional Analysis

- Product Demand Segmentation using K-Means
- Sales Anomaly Detection using Z-Score
- Category & Region Forecast Comparison

---

## Dashboard Features

- Sales Overview
- Forecast Explorer
- Anomaly Detection
- Product Demand Segmentation

---

## Best Performing Model

XGBoost achieved the lowest forecasting error.

MAE: 14,458

RMSE: 18,905

MAPE: 0.1401

---

## Project Structure

```
SalesForecasting/
│
├── analysis.ipynb
├── app.py
├── requirements.txt
├── README.md
├── charts/
├── data/
└── summary.pdf
```

---

## Author

Harshith Gadadesi