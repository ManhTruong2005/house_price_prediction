import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import prepare_training_data, preprocess_pipeline


def build_model():
    return XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        verbosity=0,
    )


def train_and_evaluate(show_plots=False):
    X_train, X_test, y_train, y_test, _ = preprocess_pipeline()

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)
    rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

    X, _ = prepare_training_data()
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual SalePrice (Log Scale)")
        plt.ylabel("Predicted SalePrice (Log Scale)")
        plt.title("XGBoost - Actual vs Predicted")
        plt.show()

        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("XGBoost - Residual Plot")
        plt.show()

        top_features = importance_df.head(15)
        plt.figure(figsize=(10, 8))
        plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Top 15 Feature Importances")
        plt.show()

    return {
        "model": model,
        "rmse": rmse,
        "r2": r2,
        "rmse_original": rmse_original,
        "importance": importance_df,
    }


def main():
    results = train_and_evaluate()

    print("=" * 50)
    print("XGBOOST RESULTS")
    print("=" * 50)
    print(f"RMSE     : {results['rmse']:.4f}")
    print(f"R2 Score : {results['r2']:.4f}")
    print("=" * 50)
    print("ORIGINAL SCALE")
    print("=" * 50)
    print(f"RMSE Original Scale: {results['rmse_original']:.2f}")
    print("=" * 50)
    print("TOP 15 IMPORTANT FEATURES")
    print("=" * 50)
    print(results["importance"].head(15))


if __name__ == "__main__":
    main()
