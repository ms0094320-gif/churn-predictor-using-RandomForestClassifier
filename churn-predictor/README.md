# Customer Churn Predictor

Predict which customers are likely to cancel their subscription using a Random Forest classifier.

**Result:** ~80% accuracy on the Telco Customer Churn dataset

---

## Files

```
churn-predictor/
├── churn_predictor.py
├── requirements.txt
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Setup & Run

```bash
pip install -r requirements.txt
python churn_predictor.py
```

Download the dataset from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV inside a `data/` folder.

## What It Does

1. Loads and cleans the Telco dataset (7,043 customers, 21 features)
2. Encodes categorical columns with LabelEncoder
3. Trains a Random Forest (100 trees)
4. Prints a full classification report
5. Saves a confusion matrix and feature importance chart

## Results

| Metric | No Churn | Churn |
|--------|----------|-------|
| Precision | 0.84 | 0.67 |
| Recall | 0.92 | 0.49 |
| F1 | 0.88 | 0.57 |

**Top predictors:** tenure, TotalCharges, MonthlyCharges, Contract type

## Stack
`scikit-learn` · `pandas` · `matplotlib` · `joblib`
