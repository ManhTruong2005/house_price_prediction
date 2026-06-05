# 🏠 House Price Prediction

Dự án dự đoán giá nhà sử dụng Machine Learning trên dataset [Ames Housing](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) từ Kaggle.

## 📁 Cấu Trúc Dự Án

```
house_price_prediction/
├── data/                              # Dữ liệu
│   ├── train.csv                      # Dữ liệu huấn luyện (1460 mẫu)
│   ├── test.csv                       # Dữ liệu kiểm tra (1459 mẫu)
│   ├── sample_submission.csv          # Mẫu file nộp bài Kaggle
│   └── data_description.txt           # Mô tả chi tiết các features
├── models/                            # Model đã train
│   ├── xgboost_model.pkl              # Model XGBoost đã lưu
│   └── feature_columns.pkl            # Danh sách feature columns
├── notebooks/                         # Jupyter Notebooks
│   └── EDA.ipynb                      # Exploratory Data Analysis
├── src/                               # Source code
│   ├── preprocess.py                  # Pipeline tiền xử lý dữ liệu
│   ├── train.py                       # Linear Regression
│   ├── random_forest.py               # Random Forest
│   ├── xgboost_model.py               # XGBoost
│   ├── compare_models.py              # So sánh tất cả models + xuất Excel
│   └── savemodel.py                   # Lưu model đã train
├── evaluation_plots/                   # Biểu đồ đánh giá model
│   ├── actual_vs_predicted.png        # Biểu đồ Actual vs Predicted
│   ├── residual_analysis.png          # Phân tích residuals
│   ├── feature_importance.png         # Feature importance (RF & XGB)
│   └── metrics_comparison.png         # So sánh metrics giữa các models
├── model_comparison_results.xlsx       # Kết quả so sánh models (Excel)
├── requirements.txt                    # Dependencies
├── .gitignore
└── README.md
```

## 🚀 Cài Đặt

```bash
# Clone repository
git clone https://github.com/ManhTruong2005/house_price_prediction.git
cd house_price_prediction

# Tạo virtual environment (khuyến nghị)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Mô tả |
|---|---|
| `pandas` | Xử lý và phân tích dữ liệu |
| `numpy` | Tính toán số học |
| `scikit-learn` | Thuật toán ML & metrics đánh giá |
| `xgboost` | Gradient Boosting framework |
| `matplotlib` | Vẽ biểu đồ |
| `joblib` | Lưu/tải model |
| `notebook` | Jupyter Notebook |
| `openpyxl` | Xuất kết quả ra file Excel |

## 📊 Sử Dụng

### So sánh tất cả models (khuyến nghị)

```bash
python src/compare_models.py
```

Script này sẽ:
- Train và evaluate 3 models (Linear Regression, Random Forest, XGBoost)
- Cross-validation 5-fold
- In bảng so sánh metrics ra console
- Lưu 4 biểu đồ đánh giá vào `evaluation_plots/`
- Xuất kết quả chi tiết ra file `model_comparison_results.xlsx`

### Chạy từng model riêng lẻ

```bash
python src/train.py            # Linear Regression
python src/random_forest.py    # Random Forest
python src/xgboost_model.py    # XGBoost
```

### Lưu model

```bash
python src/savemodel.py
```

> **Lưu ý**: Tất cả các lệnh phải được chạy từ **thư mục gốc** của dự án.

## 🧠 Models

| Model | Mô tả | Hyperparameters chính |
|---|---|---|
| **Linear Regression** | Hồi quy tuyến tính cơ bản | — |
| **Random Forest** | Ensemble learning với decision trees | `n_estimators=200`, `n_jobs=-1` |
| **XGBoost** | Gradient boosting | `n_estimators=1000`, `learning_rate=0.01`, `max_depth=3` |

## 📈 Đánh Giá Models

### Metrics sử dụng

| Metric | Ý nghĩa |
|---|---|
| **R² Score** | Hệ số xác định — tỷ lệ phương sai được giải thích (càng cao càng tốt) |
| **RMSE (Log)** | Root Mean Squared Error trên log scale (càng thấp càng tốt) |
| **MAE (Log)** | Mean Absolute Error trên log scale |
| **RMSE ($)** | RMSE quy đổi sang USD |
| **MAE ($)** | MAE quy đổi sang USD |
| **CV RMSE** | Cross-Validation RMSE (5-fold) |

### Biểu đồ đánh giá

Các biểu đồ được tự động tạo bởi `compare_models.py` và lưu trong `evaluation_plots/`:

1. **Actual vs Predicted** — So sánh giá trị thực tế và dự đoán cho cả 3 models
2. **Residual Analysis** — Phân tích phần dư để kiểm tra bias
3. **Feature Importance** — Top 15 features quan trọng nhất (Random Forest & XGBoost)
4. **Metrics Comparison** — Bar chart so sánh R², RMSE, CV RMSE giữa các models

### Kết quả Excel

File `model_comparison_results.xlsx` chứa 5 sheets:

| Sheet | Nội dung |
|---|---|
| Model Comparison | Bảng so sánh metrics tổng hợp |
| Hyperparameters | Chi tiết hyperparameters từng model |
| Feature Importance | Top 20 features quan trọng nhất |
| Dataset Info | Thông tin tổng quan dataset |
| Prediction Details | Chi tiết dự đoán từng mẫu |

## 🔧 Pipeline Xử Lý Dữ Liệu

1. **Xử lý Missing Values** — Categorical → `'None'`, Numerical → median, Categorical khác → mode
2. **Loại bỏ Outliers** — Loại nhà có `GrLivArea > 4000 sqft` nhưng `SalePrice < $300,000`
3. **Log Transform** — Target (`SalePrice`) được transform bằng `log1p`
4. **One-Hot Encoding** — Mã hóa các biến categorical
5. **Train/Test Split** — 80/20 với `random_state=42`

## 📝 Dataset

- **Nguồn**: [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Kích thước**: 1460 mẫu training, 1459 mẫu test
- **Features**: 79 features mô tả các khía cạnh của nhà ở tại Ames, Iowa
