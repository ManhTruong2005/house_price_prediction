from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess import PROJECT_ROOT, load_data, prepare_inference_features


MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"


def load_artifacts(model_path=MODEL_PATH, feature_columns_path=FEATURE_COLUMNS_PATH):
    model_path = Path(model_path)
    feature_columns_path = Path(feature_columns_path)

    if not model_path.exists() or not feature_columns_path.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Run `python src/savemodel.py` first."
        )

    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_columns_path)
    return model, feature_columns


def create_submission(
    test_path=TEST_PATH,
    output_path=SUBMISSION_PATH,
    model_path=MODEL_PATH,
    feature_columns_path=FEATURE_COLUMNS_PATH,
):
    test_df = load_data(test_path)

    if "Id" not in test_df.columns:
        raise ValueError("Test data must contain an Id column.")

    ids = test_df["Id"].copy()
    model, feature_columns = load_artifacts(model_path, feature_columns_path)
    X_test = prepare_inference_features(test_df, feature_columns)

    if getattr(model, "n_features_in_", X_test.shape[1]) != X_test.shape[1]:
        raise ValueError(
            "Model feature count does not match preprocessed test features. "
            "Run `python src/savemodel.py` to regenerate model artifacts."
        )

    predictions_log = model.predict(X_test)
    predictions = np.expm1(predictions_log)
    predictions = np.maximum(predictions, 0)

    submission = pd.DataFrame({
        "Id": ids,
        "SalePrice": predictions,
    })

    output_path = Path(output_path)
    submission.to_csv(output_path, index=False)
    return submission


def main():
    submission = create_submission()
    print("=" * 50)
    print("SUBMISSION CREATED")
    print("=" * 50)
    print(f"Path   : {SUBMISSION_PATH}")
    print(f"Shape  : {submission.shape}")
    print(f"Columns: {list(submission.columns)}")
    print(f"NaN SalePrice: {submission['SalePrice'].isna().sum()}")


if __name__ == "__main__":
    main()
