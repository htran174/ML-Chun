# Telco Customer Churn Prediction
## Team 22

### Team Members
- Hien Tran
- Partner

---

# Project Overview

This project builds a machine learning pipeline to predict customer churn using the IBM Telco Customer Churn dataset.

The goal is to identify customers who are likely to leave a telecom company so the business can take preventive action such as retention offers or customer outreach.

The project focuses on:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Binary classification modeling
- Threshold tuning using business cost tradeoffs
- Model evaluation and visualization

---

# Problem Type

Supervised Machine Learning

Binary Classification

Target Variable:
- Churn
    - Yes = 1
    - No = 0

---

# Technologies Used

- Python
- JupyterLab
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

# Project Structure

```text
ML-CHUN/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── visualization.ipynb
│
├── runs/
│
├── src/
│   ├── data.py
│   ├── train.py
│   └── eval.py
│
├── requirements.txt
└── README.md