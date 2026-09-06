from pathlib import Path

import joblib
from xgboost import XGBRegressor

from preprocess import PROJECT_ROOT, prepare_training_data


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"


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


def train_and_save_model(
    train_path=PROJECT_ROOT / "data" / "train.csv",
    model_path=MODEL_PATH,
    feature_columns_path=FEATURE_COLUMNS_PATH,
):
    X, y = prepare_training_data(train_path)
    model = build_model()
    model.fit(X, y)

    model_path = Path(model_path)
    feature_columns_path = Path(feature_columns_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(X.columns, feature_columns_path)

    return model, X.columns


def main():
    model, feature_columns = train_and_save_model()

    print("=" * 50)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 50)
    print(f"Model path          : {MODEL_PATH}")
    print(f"Feature columns path: {FEATURE_COLUMNS_PATH}")
    print(f"Feature count       : {len(feature_columns)}")
    print(f"Model feature count : {model.n_features_in_}")


if __name__ == "__main__":
    main()
