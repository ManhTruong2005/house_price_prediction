import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import prepare_training_data, preprocess_pipeline


def build_model():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
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
        plt.title("Random Forest - Actual vs Predicted")
        plt.show()

        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Random Forest - Residual Plot")
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
    print("RANDOM FOREST RESULTS")
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
