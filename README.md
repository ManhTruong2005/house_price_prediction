# House Price Prediction

Dự án dự đoán giá nhà với dataset Ames Housing từ Kaggle:
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Project Structure

```text
house_price_prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── data_description.txt
├── evaluation_plots/
│   ├── actual_vs_predicted.png
│   ├── residual_analysis.png
│   ├── feature_importance.png
│   └── metrics_comparison.png
├── models/
│   ├── xgboost_model.pkl
│   └── feature_columns.pkl
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   ├── compare_models.py
│   ├── savemodel.py
│   └── predict.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Trên Linux hoặc macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Các file dữ liệu gốc nằm trong `data/`:

- `train.csv`: dữ liệu train, có cột target `SalePrice`.
- `test.csv`: dữ liệu test Kaggle, không có `SalePrice`.
- `sample_submission.csv`: định dạng submission mẫu.
- `data_description.txt`: mô tả ý nghĩa các feature.

Không cần chỉnh sửa các file dữ liệu gốc.

## Preprocessing

Pipeline preprocessing dùng chung cho train và inference nằm trong `src/preprocess.py`.

Các bước chính:

- Fill missing values theo nhóm feature.
- Loại outlier chỉ khi xử lý training data có `SalePrice`.
- Log transform target bằng `np.log1p(SalePrice)` khi train.
- Drop `Id` khỏi feature để model không học theo mã dòng dữ liệu.
- One-hot encoding categorical features.
- Khi inference, align cột test theo đúng `feature_columns.pkl` đã lưu từ training.

## Training

Chạy từng model riêng:

```bash
python src/train.py
python src/random_forest.py
python src/xgboost_model.py
```

Các script này có `main()` guard, nên import module sẽ không tự động train model.

## Model Saving

Model binary trong `models/*.pkl` đang được `.gitignore` bỏ qua, vì vậy một người clone project mới cần tự tạo lại artifacts:

```bash
python src/savemodel.py
```

Lệnh này sẽ train XGBoost trên `data/train.csv` và tạo:

- `models/xgboost_model.pkl`
- `models/feature_columns.pkl`

`feature_columns.pkl` là danh sách cột sau preprocessing, dùng để đảm bảo inference có cùng thứ tự và số lượng feature với training.

## Prediction

Sau khi có model artifacts, tạo submission từ `data/test.csv`:

```bash
python src/predict.py
```

Output:

```text
submission.csv
```

File submission có đúng 2 cột:

```text
Id
SalePrice
```

và 1459 dòng dữ liệu.

## Web Interface

Chạy giao diện dự đoán bằng Streamlit:

```bash
streamlit run app.py
```

Giao diện cho phép:

- Dùng trực tiếp `data/test.csv`.
- Upload một file CSV khác có cột `Id`.
- Xem preview kết quả dự đoán.
- Download `submission.csv`.

## Model Comparison

Chạy so sánh Linear Regression, Random Forest và XGBoost:

```bash
python src/compare_models.py
```

Script sẽ lưu plots vào `evaluation_plots/` và kết quả chi tiết vào `model_comparison_results.xlsx`.
Mặc định script không mở GUI. Nếu muốn xem plot tương tác:

```bash
python src/compare_models.py --show-plots
```

Nếu chỉ muốn tạo plots mà không xuất Excel:

```bash
python src/compare_models.py --no-excel
```

## Testing

Chạy kiểm tra syntax:

```bash
python -m compileall src
```

Chạy automated tests:

```bash
pytest
```

Các test kiểm tra:

- Train/test preprocessing chạy được.
- `test.csv` không cần `SalePrice`.
- Feature train và inference khớp nhau.
- Model load được và predict đủ số dòng test.
- Submission có đúng cột, đúng số dòng, không có NaN trong `SalePrice`.

## Expected Fresh Clone Flow

```bash
pip install -r requirements.txt
python src/savemodel.py
python src/predict.py
pytest
```

Sau flow này, project sẽ có model artifacts trong `models/` và file `submission.csv` ở thư mục gốc.
