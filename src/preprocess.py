# src/preprocess.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# LOAD DATA
# =========================

def load_data(path):
    df = pd.read_csv(path)
    return df


# =========================
# HANDLE MISSING VALUES
# =========================

def handle_missing_values(df):

    # NaN = feature không tồn tại
    none_cols = [
        'PoolQC',
        'MiscFeature',
        'Alley',
        'Fence',
        'FireplaceQu',
        'GarageType',
        'GarageFinish',
        'GarageQual',
        'GarageCond',
        'BsmtQual',
        'BsmtCond',
        'BsmtExposure',
        'BsmtFinType1',
        'BsmtFinType2',
        'MasVnrType'
    ]

    for col in none_cols:
        df[col] = df[col].fillna('None')

    # Numerical features -> median
    median_cols = [
        'LotFrontage',
        'MasVnrArea',
        'GarageYrBlt'
    ]

    for col in median_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical features -> mode
    mode_cols = [
        'Electrical',
        'MSZoning',
        'KitchenQual',
        'Exterior1st',
        'Exterior2nd',
        'SaleType'
    ]

    for col in mode_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# =========================
# REMOVE OUTLIERS
# =========================

def remove_outliers(df):

    df = df.drop(
        df[
            (df['GrLivArea'] > 4000) &
            (df['SalePrice'] < 300000)
        ].index
    )

    return df


# =========================
# FEATURE ENGINEERING
# =========================

def preprocess_features(df):

    # Tách X và y
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # Log transform target
    y = np.log1p(y)

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y


# =========================
# FEATURE SCALING
# =========================

def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# =========================
# TRAIN TEST SPLIT
# =========================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# =========================
# FULL PIPELINE
# =========================

def preprocess_pipeline(path):
    df = load_data(path)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    X, y = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler