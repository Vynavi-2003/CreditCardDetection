This project implements a Credit Card Fraud Detection pipeline using machine learning techniques. The dataset is heavily imbalanced, and this challenge is tackled using SMOTE (Synthetic Minority Over-sampling Technique). The classification model used is Random Forest, evaluated with and without SMOTE.

Architecture:
![Architecture](credit card fraud detection)

Key Features:

1. Exploratory Data Analysis (EDA)
2. Baseline Random Forest Classifier
3. Data balancing with SMOTE
4. Performance metrics: Confusion Matrix, ROC Curve, AUC


📁 Dataset
Source: Kaggle - Credit Card Fraud Detection
Contains 284,807 transactions, with 492 frauds (~0.172%)
Features: Time, Amount, and anonymized V1 to V28
Target variable: Class (0 = Legitimate, 1 = Fraud)

Key libraries:
pandas
numpy
matplotlib, seaborn
scikit-learn
imbalanced-learn

📊 Results
Baseline Random Forest (Without SMOTE)
Class Imbalance: Severe (~0.17% fraud)
AUC: ~0.97+
High Precision, but lower Recall for fraud cases
Random Forest with SMOTE
Class balance achieved through oversampling
Recall improved significantly
ROC-AUC remained competitive, with better fraud detection
Feature Importance
Most correlated features with fraud:
V17, V14, V12, V10

Future Work
1. Hyperparameter tuning with GridSearchCV
2. Try more models like XGBoost, LightGBM
3. Implement Neural Networks (e.g., LSTM for sequential data)
4. Explore cost-sensitive learning or ensemble stacking
5. Apply anomaly detection techniques

