
# 30-Day Hospital Readmission Prediction (Diabetes 130-US Hospitals)

A proof-of-concept classifier that predicts whether a diabetic patient is readmitted within 30 days of discharge. This is a learning project, not a clinical tool.

I built it by following the method in [Emi-Johnson & Nkrumah (2025), Cureus](https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/), then made some changes of my own (listed below).

## Dataset
UCI Diabetes 130-US Hospitals (1999-2008), about 101,766 encounters.
Download it from the UCI Machine Learning Repository and update the file path in the notebook.

## What I did
- Cleaned the data: converted placeholder values to missing, handled heavily incomplete columns, grouped diagnosis codes, excluded expired/hospice encounters
- Explored the data: distributions, correlation, Cramér's V for categorical variables
- Split train/test **by patient** so no patient appears in both sets (prevents data leakage)
- Selected features with LASSO and examined PCA
- Compared logistic regression, random forest and a neural network, with hyperparameter tuning
- Handled class imbalance with class weights

## Results
Best model (neural network): **ROC-AUC about 0.67 on the held-out test set**.
Readmission within 30 days is rare and hard to predict from this data, so precision is low (about 0.19). The model is a weak signal, not a deployable predictor.

## Limitations
Historical data (1999-2008), limited features, strong class imbalance, no external validation.

## Tools
Python, Pandas, NumPy, Scikit-learn, Keras, Matplotlib, Seaborn
