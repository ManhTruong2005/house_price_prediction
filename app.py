import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import create_submission


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="H",
    layout="wide",
)


def run_prediction(input_file):
    if input_file is None:
        return create_submission(
            test_path=PROJECT_ROOT / "data" / "test.csv",
            output_path=PROJECT_ROOT / "submission.csv",
        )

    uploaded_df = pd.read_csv(input_file)
    temp_input_path = PROJECT_ROOT / "uploaded_input.csv"
    temp_output_path = PROJECT_ROOT / "uploaded_submission.csv"

    try:
        uploaded_df.to_csv(temp_input_path, index=False)
        return create_submission(
            test_path=temp_input_path,
            output_path=temp_output_path,
        )
    finally:
        if temp_input_path.exists():
            temp_input_path.unlink()
        if temp_output_path.exists():
            temp_output_path.unlink()


st.title("House Price Prediction")
st.caption("Dự đoán SalePrice từ file test CSV và tạo submission Kaggle.")

with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
    use_default = st.checkbox("Dùng data/test.csv mặc định", value=True)

    st.header("Model")
    st.write("Model: `models/xgboost_model.pkl`")
    st.write("Features: `models/feature_columns.pkl`")

input_file = None if use_default else uploaded_file

if not use_default and uploaded_file is None:
    st.info("Hãy upload file CSV có cột Id, hoặc bật lựa chọn dùng data/test.csv mặc định.")
else:
    if st.button("Predict", type="primary"):
        try:
            submission = run_prediction(input_file)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        st.success("Prediction completed.")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Rows", f"{submission.shape[0]:,}")
        metric_cols[1].metric("Columns", submission.shape[1])
        metric_cols[2].metric("NaN SalePrice", int(submission["SalePrice"].isna().sum()))

        st.subheader("Preview")
        st.dataframe(submission.head(50), use_container_width=True)

        csv_bytes = submission.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download submission.csv",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
        )

        if use_default:
            st.caption(f"Saved to: {PROJECT_ROOT / 'submission.csv'}")
