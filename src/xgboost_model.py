# src/xgboost_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from preprocess import (
    preprocess_pipeline,
    handle_missing_values,
    remove_outliers
)


# =====================================
# LOAD PREPROCESSED DATA
# =====================================

X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
    'data/train.csv'
)


# =====================================
# CREATE MODEL
# =====================================

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)


# =====================================
# TRAIN MODEL
# =====================================

model.fit(
    X_train,
    y_train
)


# =====================================
# PREDICTION
# =====================================

y_pred = model.predict(X_test)


# =====================================
# EVALUATION
# =====================================

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


# =====================================
# PRINT RESULTS
# =====================================

print('=' * 50)
print('XGBOOST RESULTS')
print('=' * 50)

print(f'RMSE     : {rmse:.4f}')
print(f'R2 Score : {r2:.4f}')


# =====================================
# ORIGINAL SCALE RMSE
# =====================================

y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

rmse_original = np.sqrt(
    mean_squared_error(
        y_test_original,
        y_pred_original
    )
)

print('=' * 50)
print('ORIGINAL SCALE')
print('=' * 50)

print(f'RMSE Original Scale: {rmse_original:.2f}')


# =====================================
# ACTUAL VS PREDICTED
# =====================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel('Actual SalePrice (Log Scale)')
plt.ylabel('Predicted SalePrice (Log Scale)')

plt.title('XGBoost - Actual vs Predicted')

plt.show()


# =====================================
# RESIDUAL PLOT
# =====================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, linestyle='--')

plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.title('XGBoost - Residual Plot')

plt.show()


# =====================================
# FEATURE IMPORTANCE
# =====================================

df = pd.read_csv('data/train.csv')

df = handle_missing_values(df)
df = remove_outliers(df)

X = df.drop('SalePrice', axis=1)

X = pd.get_dummies(X, drop_first=True)

feature_names = X.columns


importance = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)


# =====================================
# TOP 15 IMPORTANT FEATURES
# =====================================

print('=' * 50)
print('TOP 15 IMPORTANT FEATURES')
print('=' * 50)

print(importance_df.head(15))


# =====================================
# FEATURE IMPORTANCE PLOT
# =====================================

top_features = importance_df.head(15)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features['Feature'][::-1],
    top_features['Importance'][::-1]
)

plt.xlabel('Importance')
plt.ylabel('Feature')

plt.title('Top 15 Feature Importances')

plt.show()