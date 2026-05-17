# Customer Churn Prediction — COMP462 Term Project

Predicting customer churn using IBM's Telco Customer Churn dataset (7043 customers, 33 features).

## Project Structure

```
CustomerChurnPrediction_COMP462/
├── dataset/
│   └── Telco_customer_churn.xlsx   # Raw dataset
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb      # Data Cleaning & Feature Engineering
│   └── 03_Models_Evaluation.ipynb  # Model Training & Comparison
├── src/
│   ├── preprocessing.py            # Reusable preprocessing functions
│   └── evaluation.py               # Metrics, plots, comparison table
├── outputs/
│   ├── figures/                    # Saved plots
│   ├── models/                     # Saved trained models (.pkl)
│   ├── X_train.csv / X_test.csv
│   └── y_train.csv / y_test.csv
└── requirements.txt
```

## Models

| Model | Class Imbalance Strategy |
|-------|--------------------------|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` + GridSearchCV |
| K-Nearest Neighbors | K tuned via 5-fold CV |

## Evaluation Metrics

Accuracy · Precision · Recall · F1-Score · MCC · ROC-AUC · Confusion Matrix

## Setup

```bash
pip install -r requirements.txt
```

Run notebooks in order: `01_EDA` → `02_Preprocessing` → `03_Models_Evaluation`

## Dataset

IBM Telco Customer Churn Dataset — 7043 customers, 33 variables.  
Target variable: `Churn Value` (1 = churned, 0 = retained)  
Class distribution: ~26% churn, ~74% retained
