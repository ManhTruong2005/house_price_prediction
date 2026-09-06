from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"

NONE_COLS = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
]

MEDIAN_COLS = [
    "LotFrontage",
    "MasVnrArea",
    "GarageYrBlt",
]

MODE_COLS = [
    "Electrical",
    "MSZoning",
    "KitchenQual",
    "Exterior1st",
    "Exterior2nd",
    "SaleType",
]


def load_data(path):
    return pd.read_csv(path)


def handle_missing_values(df):
    df = df.copy()

    for col in NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in MEDIAN_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in MODE_COLS:
        if col in df.columns and df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "None"
            df[col] = df[col].fillna(fill_value)

    return df


def remove_outliers(df):
    if {"GrLivArea", "SalePrice"}.issubset(df.columns):
        return df.drop(
            df[
                (df["GrLivArea"] > 4000)
                & (df["SalePrice"] < 300000)
            ].index
        )
    return df.copy()


def preprocess_features(df, is_train=True, feature_columns=None):
    df = df.copy()

    if is_train:
        if "SalePrice" not in df.columns:
            raise ValueError("Training data must contain a SalePrice column.")
        y = np.log1p(df["SalePrice"])
        X = df.drop("SalePrice", axis=1)
    else:
        y = None
        X = df.drop("SalePrice", axis=1, errors="ignore")

    X = X.drop("Id", axis=1, errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    if feature_columns is not None:
        X = X.reindex(columns=list(feature_columns), fill_value=0)

    if is_train:
        return X, y
    return X


def prepare_training_data(path=DEFAULT_TRAIN_PATH):
    df = load_data(path)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    return preprocess_features(df, is_train=True)


def prepare_inference_features(df, feature_columns):
    df = handle_missing_values(df)
    return preprocess_features(df, is_train=False, feature_columns=feature_columns)


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def preprocess_pipeline(path=DEFAULT_TRAIN_PATH):
    X, y = prepare_training_data(path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
