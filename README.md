# Industrial Predictive Maintenance Engine

## Overview
This project implements an end-to-end Machine Learning pipeline to predict equipment failures in a manufacturing environment. Built with `scikit-learn` and `XGBoost`, the model analyzes raw sensor data (temperature, rotational speed, torque, tool wear) to trigger early warnings before mechanical breakdowns occur. 

The core objective of this project is rooted in industrial business logic: **minimizing catastrophic downtime by maximizing the Recall metric**, favoring cheap preventive inspections over expensive unexpected failures.

## Dataset
The project utilizes the **AI4I 2020 Predictive Maintenance Dataset**. It consists of 10,000 data points with 14 features representing machine conditions, sensor readings, and specific failure modes. 
*   **Target Variable:** `Machine failure` (Binary: 0 for functional, 1 for failure).
*   **Class Imbalance:** Highly imbalanced, with failures representing less than 4% of the data.

## Project Architecture
The codebase is structured for production-level modularity, separating exploration, preprocessing, training, and evaluation.

```text
predictive-maintenance/
├── data/
│   ├── raw/                 # Original AI4I 2020 dataset
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory Data Analysis & Visualizations
├── src/
│   ├── preprocess.py        # Data cleaning, feature isolation, and train-test splits
│   ├── train.py             # Pipeline construction, GridSearchCV, and XGBoost training
│   └── evaluate.py          # Model evaluation, Confusion Matrix, and Classification Report
├── models/
│   └── xgboost_pipeline.joblib # Serialized production-ready model
├── requirements.txt         # Project dependencies
└── README.md

## Key Engineering Decisions
1. Eliminating Data Leakage
During EDA, a correlation matrix revealed that specific error type columns (TWF, HDF, PWF, OSF, RNF) were perfectly correlated with the target variable. These were post-failure annotations made by technicians. To ensure the model learns strictly from pre-failure sensor data, these columns were explicitly dropped in the preprocess.py module.

2. Pipeline Integration
Instead of applying manual transformations to the dataframes, a scikit-learn Pipeline combined with a ColumnTransformer was used. This ensures that:

Numerical features are scaled using StandardScaler.

Categorical quality tiers (L, M, H) are properly encoded using OrdinalEncoder.

The entire preprocessing step is serialized with the model, meaning new raw data can be fed directly into xgboost_pipeline.joblib without manual cleaning.

3. Hyperparameter Tuning & The Precision-Recall Trade-off
To handle the severe class imbalance, the XGBoost algorithm was wrapped in a GridSearchCV optimized specifically for the Recall scoring metric.

Business Logic Application:
In a factory setting, a False Negative (a machine crashes unexpectedly) halts the production line and costs thousands of euros in damages and downtime. A False Positive (a false alarm) only costs 10 minutes of a technician's time for a visual inspection.

By hyper-tuning for Recall, the model achieved a 93% failure detection rate (Recall = 0.93) on unseen test data. The trade-off was a lower precision score, generating a controlled number of false alarms. This asymmetric cost optimization aligns perfectly with real-world preventive maintenance strategies.

## How to Run

1.Clone the repository and install dependencies:

 git clone [https://github.com/BimbiricaDenis/predictive-maintenance.git](https://github.com/BimbiricaDenis/predictive-maintenance.git)
cd predictive-maintenance
pip install -r requirements.txt


2.Train the model:
This will load the raw data, run the grid search for optimal parameters, and save the model to the models/ directory.

python src/train.py

3.Evaluate the model:

python src/evaluate.py