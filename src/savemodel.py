# src/save_model.py

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from preprocess import (
    handle_missing_values,
    remove_outliers
)


# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv('data/train.csv')

df = handle_missing_values(df)

df = remove_outliers(df)


# =====================================
# FEATURES / TARGET
# =====================================

X = df.drop('SalePrice', axis=1)

y = np.log1p(df['SalePrice'])


# =====================================
# ONE HOT ENCODING
# =====================================

X = pd.get_dummies(
    X,
    drop_first=True
)


# SAVE FEATURE COLUMNS
feature_columns = X.columns


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

model.fit(X, y)


# =====================================
# SAVE MODEL
# =====================================

joblib.dump(
    model,
    'models/xgboost_model.pkl'
)


# =====================================
# SAVE FEATURE COLUMNS
# =====================================

joblib.dump(
    feature_columns,
    'models/feature_columns.pkl'
)


print('=' * 50)
print('MODEL SAVED SUCCESSFULLY')
print('=' * 50)