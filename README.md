# 🏠 House Price Prediction

Dự án dự đoán giá nhà sử dụng Machine Learning trên dataset [Ames Housing](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) từ Kaggle.

## 📁 Cấu Trúc Dự Án

```
house_price_prediction/
├── data/                      # Dữ liệu
│   ├── train.csv              # Dữ liệu huấn luyện (1460 mẫu)
│   ├── test.csv               # Dữ liệu kiểm tra
│   ├── sample_submission.csv
│   └── data_description.txt   # Mô tả các features
├── models/                    # Model đã train
│   ├── xgboost_model.pkl
│   └── feature_columns.pkl
├── notebooks/                 # Jupyter Notebooks
│   └── EDA.ipynb              # Exploratory Data Analysis
├── src/                       # Source code
│   ├── preprocess.py          # Pipeline tiền xử lý dữ liệu
│   ├── train.py               # Linear Regression
│   ├── random_forest.py       # Random Forest
│   ├── xgboost_model.py       # XGBoost
│   ├── compare_models.py      # So sánh tất cả models
│   └── savemodel.py           # Lưu model đã train
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd house_price_prediction

# Tạo virtual environment (khuyến nghị)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

## 📊 Sử Dụng

### So sánh tất cả models (khuyến nghị)

```bash
python src/compare_models.py
```

Script này sẽ:
- Train và evaluate 3 models (Linear Regression, Random Forest, XGBoost)
- Cross-validation 5-fold
- In bảng so sánh metrics
- Lưu biểu đồ kết quả

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

| Model | Mô tả |
|---|---|
| **Linear Regression** | Mô hình hồi quy tuyến tính cơ bản |
| **Random Forest** | Ensemble learning với 200 decision trees |
| **XGBoost** | Gradient boosting với 1000 estimators |

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
