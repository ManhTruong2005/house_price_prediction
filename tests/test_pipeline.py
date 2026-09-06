import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import create_submission, load_artifacts
from preprocess import load_data, prepare_inference_features, prepare_training_data


def test_preprocess_train_and_test_feature_columns_match():
    X_train, y_train = prepare_training_data()
    test_df = load_data(PROJECT_ROOT / "data" / "test.csv")

    X_test = prepare_inference_features(test_df, X_train.columns)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(test_df)
    assert "SalePrice" not in test_df.columns
    assert "Id" not in X_train.columns
    assert "Id" not in X_test.columns
    assert list(X_test.columns) == list(X_train.columns)


def test_model_loads_and_predicts_test_rows():
    model, feature_columns = load_artifacts()
    test_df = load_data(PROJECT_ROOT / "data" / "test.csv")
    X_test = prepare_inference_features(test_df, feature_columns)

    predictions = model.predict(X_test)

    assert X_test.shape[1] == model.n_features_in_
    assert len(predictions) == len(test_df)
    assert np.isfinite(predictions).all()


def test_create_submission_file():
    output_path = PROJECT_ROOT / "submission_test.csv"
    try:
        submission = create_submission(output_path=output_path)
        saved_submission = pd.read_csv(output_path)
        test_df = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")

        assert output_path.exists()
        assert submission.shape == (1459, 2)
        assert saved_submission.shape == (1459, 2)
        assert list(saved_submission.columns) == ["Id", "SalePrice"]
        assert saved_submission["SalePrice"].isna().sum() == 0
        assert saved_submission["Id"].equals(test_df["Id"])
    finally:
        if output_path.exists():
            output_path.unlink()
