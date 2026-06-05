# src/train.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from preprocess import preprocess_pipeline


# =========================
# LOAD PREPROCESSED DATA
# =========================

X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
    'data/train.csv'
)


# =========================
# CREATE MODEL
# =========================

model = LinearRegression()


# =========================
# TRAIN MODEL
# =========================

model.fit(X_train, y_train)


# =========================
# PREDICTION
# =========================

y_pred = model.predict(X_test)


# =========================
# EVALUATION
# =========================

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


# =========================
# PRINT RESULTS
# =========================

print('=' * 50)
print('LINEAR REGRESSION RESULTS')
print('=' * 50)

print(f'RMSE     : {rmse:.4f}')
print(f'R2 Score : {r2:.4f}')


# =========================
# ACTUAL VS PREDICTED
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel('Actual SalePrice (Log Scale)')
plt.ylabel('Predicted SalePrice (Log Scale)')

plt.title('Actual vs Predicted')

plt.show()


# =========================
# RESIDUAL ANALYSIS
# =========================

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.title('Residual Plot')

plt.show()


# =========================
# OPTIONAL:
# CONVERT BACK TO ORIGINAL SCALE
# =========================

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