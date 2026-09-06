import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import preprocess_pipeline


def train_and_evaluate(show_plots=False):
    X_train, X_test, y_train, y_test, _ = preprocess_pipeline()

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)
    rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual SalePrice (Log Scale)")
        plt.ylabel("Predicted SalePrice (Log Scale)")
        plt.title("Actual vs Predicted")
        plt.show()

        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color="red", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()

    return {
        "model": model,
        "rmse": rmse,
        "r2": r2,
        "rmse_original": rmse_original,
    }


def main():
    results = train_and_evaluate()

    print("=" * 50)
    print("LINEAR REGRESSION RESULTS")
    print("=" * 50)
    print(f"RMSE     : {results['rmse']:.4f}")
    print(f"R2 Score : {results['r2']:.4f}")
    print("=" * 50)
    print("ORIGINAL SCALE")
    print("=" * 50)
    print(f"RMSE Original Scale: {results['rmse_original']:.2f}")


if __name__ == "__main__":
    main()
