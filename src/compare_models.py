# src/compare_models.py
# So sánh hiệu suất các phương pháp dự đoán giá nhà

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import (
    cross_val_score,
    KFold
)

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

from preprocess import (
    load_data,
    handle_missing_values,
    remove_outliers,
    preprocess_features,
    split_data
)


# =============================================
# LOAD & PREPROCESS DATA
# =============================================

print('=' * 60)
print('  HOUSE PRICE PREDICTION - MODEL COMPARISON')
print('=' * 60)

# Tạo thư mục lưu ảnh đánh giá
EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation_plots')
os.makedirs(EVAL_DIR, exist_ok=True)

# Đường dẫn file Excel kết quả
EXCEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_comparison_results.xlsx')

df = load_data('data/train.csv')
df = handle_missing_values(df)
df = remove_outliers(df)

X, y = preprocess_features(df)

# Bỏ cột Id (không phải feature dự đoán)
X = X.drop('Id', axis=1, errors='ignore')

feature_columns = X.columns

X_train, X_test, y_train, y_test = split_data(X, y)

print(f'\n  Dataset Info:')
print(f'    Total samples  : {len(df)}')
print(f'    Training set   : {X_train.shape[0]}')
print(f'    Test set       : {X_test.shape[0]}')
print(f'    Features       : {X_train.shape[1]}')


# =============================================
# DEFINE MODELS
# =============================================

models = {

    'Linear Regression': LinearRegression(),

    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),

    'XGBoost': XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        verbosity=0
    ),
}


# =============================================
# TRAIN & EVALUATE EACH MODEL
# =============================================

results = {}

for name, model in models.items():

    print(f'\n{"=" * 60}')
    print(f'  {name}')
    print(f'{"=" * 60}')

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics — Log Scale
    rmse_log = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )
    r2 = r2_score(y_test, y_pred)
    mae_log = mean_absolute_error(y_test, y_pred)

    # Metrics — Original Scale (USD)
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)

    rmse_usd = np.sqrt(
        mean_squared_error(y_test_orig, y_pred_orig)
    )
    mae_usd = mean_absolute_error(y_test_orig, y_pred_orig)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_orig': y_pred_orig,
        'RMSE (Log)': rmse_log,
        'R² Score': r2,
        'MAE (Log)': mae_log,
        'RMSE ($)': rmse_usd,
        'MAE ($)': mae_usd,
    }

    print(f'  RMSE (Log)  : {rmse_log:.4f}')
    print(f'  R2 Score    : {r2:.4f}')
    print(f'  MAE (Log)   : {mae_log:.4f}')
    print(f'  RMSE (USD)  : ${rmse_usd:,.2f}')
    print(f'  MAE (USD)   : ${mae_usd:,.2f}')


# =============================================
# CROSS VALIDATION (5-Fold)
# =============================================

print(f'\n{"=" * 60}')
print('  CROSS VALIDATION (5-Fold)')
print(f'{"=" * 60}')

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_models = {

    'Linear Regression': LinearRegression(),

    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    'XGBoost': XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        verbosity=0
    ),
}

for name, cv_model in cv_models.items():

    cv_scores = cross_val_score(
        cv_model, X, y,
        cv=kf,
        scoring='neg_mean_squared_error'
    )

    cv_rmse = np.sqrt(-cv_scores)

    results[name]['CV RMSE Mean'] = cv_rmse.mean()
    results[name]['CV RMSE Std'] = cv_rmse.std()

    print(f'\n  {name}:')
    print(f'    CV RMSE : {cv_rmse.mean():.4f} +/- {cv_rmse.std():.4f}')
    print(f'    Folds   : {[f"{s:.4f}" for s in cv_rmse]}')


# =============================================
# COMPARISON TABLE
# =============================================

print(f'\n{"=" * 60}')
print('  MODEL COMPARISON SUMMARY')
print(f'{"=" * 60}\n')

comparison_df = pd.DataFrame({
    name: {
        'RMSE (Log)': f'{info["RMSE (Log)"]:.4f}',
        'R² Score': f'{info["R² Score"]:.4f}',
        'MAE (Log)': f'{info["MAE (Log)"]:.4f}',
        'RMSE ($)': f'${info["RMSE ($)"]:,.0f}',
        'MAE ($)': f'${info["MAE ($)"]:,.0f}',
        'CV RMSE': f'{info["CV RMSE Mean"]:.4f} +/- {info["CV RMSE Std"]:.4f}',
    }
    for name, info in results.items()
}).T

print(comparison_df.to_string())

# Best model
best_name = max(
    results,
    key=lambda k: results[k]['R² Score']
)

print(f'\n  >>> Best Model: {best_name}')
print(f'      R2 Score : {results[best_name]["R\u00b2 Score"]:.4f}')
print(f'      RMSE ($) : ${results[best_name]["RMSE ($)"]:,.2f}')


# =============================================
# PLOT 1: Actual vs Predicted
# =============================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, info) in enumerate(results.items()):

    ax = axes[i]

    ax.scatter(
        y_test, info['y_pred'],
        alpha=0.5, s=15, c='#3498db'
    )

    min_val = min(y_test.min(), info['y_pred'].min())
    max_val = max(y_test.max(), info['y_pred'].max())

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--', lw=2, label='Perfect Fit'
    )

    ax.set_xlabel('Actual (Log Scale)')
    ax.set_ylabel('Predicted (Log Scale)')
    ax.set_title(f'{name}\nR² = {info["R² Score"]:.4f}')
    ax.legend()

plt.suptitle(
    'Actual vs Predicted — All Models',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join(EVAL_DIR, 'actual_vs_predicted.png'),
    dpi=150, bbox_inches='tight'
)
plt.show()
print(f'\n[OK] Saved: {os.path.join(EVAL_DIR, "actual_vs_predicted.png")}')


# =============================================
# PLOT 2: Residuals
# =============================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, info) in enumerate(results.items()):

    ax = axes[i]

    residuals = y_test - info['y_pred']

    ax.scatter(
        info['y_pred'], residuals,
        alpha=0.5, s=15, c='#e74c3c'
    )

    ax.axhline(
        y=0, color='black',
        linestyle='--', lw=1.5
    )

    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{name}')

plt.suptitle(
    'Residual Analysis — All Models',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join(EVAL_DIR, 'residual_analysis.png'),
    dpi=150, bbox_inches='tight'
)
plt.show()
print(f'[OK] Saved: {os.path.join(EVAL_DIR, "residual_analysis.png")}')


# =============================================
# PLOT 3: Feature Importance (Tree Models)
# =============================================

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

tree_models = ['Random Forest', 'XGBoost']

for i, name in enumerate(tree_models):

    model = results[name]['model']
    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    top15 = importance_df.head(15)

    ax = axes[i]

    colors = plt.cm.viridis(
        np.linspace(0.3, 0.9, 15)
    )

    ax.barh(
        top15['Feature'][::-1],
        top15['Importance'][::-1],
        color=colors
    )

    ax.set_xlabel('Importance')
    ax.set_title(f'{name} — Top 15 Features')

plt.suptitle(
    'Feature Importance Comparison',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join(EVAL_DIR, 'feature_importance.png'),
    dpi=150, bbox_inches='tight'
)
plt.show()
print(f'[OK] Saved: {os.path.join(EVAL_DIR, "feature_importance.png")}')


# =============================================
# PLOT 4: Metrics Comparison Bar Chart
# =============================================

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

model_names = list(results.keys())
colors = ['#2ecc71', '#3498db', '#e67e22']


# --- R² Score ---
ax = axes[0]
values = [results[n]['R² Score'] for n in model_names]
bars = ax.bar(model_names, values, color=colors, alpha=0.85)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f'{val:.4f}',
        ha='center', va='bottom', fontweight='bold'
    )

ax.set_title('R2 Score (higher = better)')
r2_min = min(values)
ax.set_ylim(r2_min - 0.05, max(values) + 0.03)


# --- RMSE (Log) ---
ax = axes[1]
values = [results[n]['RMSE (Log)'] for n in model_names]
bars = ax.bar(model_names, values, color=colors, alpha=0.85)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f'{val:.4f}',
        ha='center', va='bottom', fontweight='bold'
    )

ax.set_title('RMSE - Log Scale (lower = better)')
rmse_min = min(values)
ax.set_ylim(rmse_min - 0.03, max(values) + 0.02)


# --- CV RMSE ---
ax = axes[2]
means = [results[n]['CV RMSE Mean'] for n in model_names]
stds = [results[n]['CV RMSE Std'] for n in model_names]
bars = ax.bar(
    model_names, means,
    yerr=stds, color=colors,
    alpha=0.85, capsize=5
)

for bar, val in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.008,
        f'{val:.4f}',
        ha='center', va='bottom', fontweight='bold'
    )

cv_min = min(means)
ax.set_ylim(cv_min - 0.03, max(means) + 0.03)

ax.set_title('Cross-Validation RMSE (lower = better)')


plt.suptitle(
    'Model Performance Comparison',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join(EVAL_DIR, 'metrics_comparison.png'),
    dpi=150, bbox_inches='tight'
)
plt.show()
print(f'[OK] Saved: {os.path.join(EVAL_DIR, "metrics_comparison.png")}')


# =============================================
# EXPORT TO EXCEL
# =============================================

print(f'\n{"=" * 60}')
print('  EXPORTING RESULTS TO EXCEL')
print(f'{"=" * 60}')

wb = Workbook()

# --- Styles ---
header_font = Font(name='Calibri', bold=True, size=12, color='FFFFFF')
header_fill = PatternFill(start_color='2C3E50', end_color='2C3E50', fill_type='solid')
best_fill = PatternFill(start_color='D5F5E3', end_color='D5F5E3', fill_type='solid')
thin_border = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)
center_align = Alignment(horizontal='center', vertical='center')


def style_header(ws, num_cols):
    """Áp dụng style cho header row."""
    for col_idx in range(1, num_cols + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = thin_border


def auto_width(ws):
    """Tự động điều chỉnh độ rộng cột."""
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
            cell.border = thin_border
            cell.alignment = center_align
        ws.column_dimensions[col_letter].width = max_len + 4


# =============================================
# Sheet 1: Model Comparison Summary
# =============================================

ws1 = wb.active
ws1.title = 'Model Comparison'

# Tạo DataFrame kết quả dạng số (không format chuỗi)
summary_data = []
for name, info in results.items():
    summary_data.append({
        'Model': name,
        'R² Score': round(info['R² Score'], 4),
        'RMSE (Log)': round(info['RMSE (Log)'], 4),
        'MAE (Log)': round(info['MAE (Log)'], 4),
        'RMSE ($)': round(info['RMSE ($)'], 2),
        'MAE ($)': round(info['MAE ($)'], 2),
        'CV RMSE Mean': round(info['CV RMSE Mean'], 4),
        'CV RMSE Std': round(info['CV RMSE Std'], 4),
    })

summary_df = pd.DataFrame(summary_data)

# Ghi header
for col_idx, col_name in enumerate(summary_df.columns, 1):
    ws1.cell(row=1, column=col_idx, value=col_name)

# Ghi data
for row_idx, row in enumerate(summary_df.itertuples(index=False), 2):
    for col_idx, value in enumerate(row, 1):
        ws1.cell(row=row_idx, column=col_idx, value=value)

# Format số tiền USD
for row_idx in range(2, len(summary_data) + 2):
    ws1.cell(row=row_idx, column=5).number_format = '#,##0.00'  # RMSE ($)
    ws1.cell(row=row_idx, column=6).number_format = '#,##0.00'  # MAE ($)

# Highlight best model
best_row = None
best_r2 = -1
for row_idx in range(2, len(summary_data) + 2):
    r2_val = ws1.cell(row=row_idx, column=2).value
    if r2_val > best_r2:
        best_r2 = r2_val
        best_row = row_idx

if best_row:
    for col_idx in range(1, len(summary_df.columns) + 1):
        ws1.cell(row=best_row, column=col_idx).fill = best_fill

style_header(ws1, len(summary_df.columns))
auto_width(ws1)

print('  [OK] Sheet: Model Comparison')


# =============================================
# Sheet 2: Hyperparameters
# =============================================

ws2 = wb.create_sheet('Hyperparameters')

hyper_data = []
for name, model in models.items():
    params = model.get_params()
    for param, value in params.items():
        hyper_data.append({
            'Model': name,
            'Parameter': param,
            'Value': str(value)
        })

hyper_df = pd.DataFrame(hyper_data)

for col_idx, col_name in enumerate(hyper_df.columns, 1):
    ws2.cell(row=1, column=col_idx, value=col_name)

for row_idx, row in enumerate(hyper_df.itertuples(index=False), 2):
    for col_idx, value in enumerate(row, 1):
        ws2.cell(row=row_idx, column=col_idx, value=value)

style_header(ws2, len(hyper_df.columns))
auto_width(ws2)

print('  [OK] Sheet: Hyperparameters')


# =============================================
# Sheet 3: Feature Importance (Top 20)
# =============================================

ws3 = wb.create_sheet('Feature Importance')

tree_model_names = ['Random Forest', 'XGBoost']
fi_data = []

for name in tree_model_names:
    model_obj = results[name]['model']
    importance = model_obj.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(20)

    for _, row in fi_df.iterrows():
        fi_data.append({
            'Model': name,
            'Feature': row['Feature'],
            'Importance': round(row['Importance'], 6)
        })

fi_full_df = pd.DataFrame(fi_data)

for col_idx, col_name in enumerate(fi_full_df.columns, 1):
    ws3.cell(row=1, column=col_idx, value=col_name)

for row_idx, row in enumerate(fi_full_df.itertuples(index=False), 2):
    for col_idx, value in enumerate(row, 1):
        ws3.cell(row=row_idx, column=col_idx, value=value)

style_header(ws3, len(fi_full_df.columns))
auto_width(ws3)

print('  [OK] Sheet: Feature Importance')


# =============================================
# Sheet 4: Dataset Info
# =============================================

ws4 = wb.create_sheet('Dataset Info')

dataset_info = [
    ['Property', 'Value'],
    ['Total Samples', len(df)],
    ['Training Set Size', X_train.shape[0]],
    ['Test Set Size', X_test.shape[0]],
    ['Number of Features', X_train.shape[1]],
    ['Target Variable', 'SalePrice (log1p transformed)'],
    ['Test Size Ratio', '20%'],
    ['Random State', 42],
    ['Best Model', best_name],
    ['Best R² Score', round(results[best_name]['R² Score'], 4)],
    ['Best RMSE ($)', round(results[best_name]['RMSE ($)'], 2)],
]

for row_idx, row_data in enumerate(dataset_info, 1):
    for col_idx, value in enumerate(row_data, 1):
        ws4.cell(row=row_idx, column=col_idx, value=value)

style_header(ws4, 2)
auto_width(ws4)

print('  [OK] Sheet: Dataset Info')


# =============================================
# Sheet 5: Prediction Details (sample)
# =============================================

ws5 = wb.create_sheet('Prediction Details')

pred_data = {'Actual (Log)': y_test.values, 'Actual ($)': np.expm1(y_test).values}
for name, info in results.items():
    pred_data[f'{name} Pred (Log)'] = info['y_pred']
    pred_data[f'{name} Pred ($)'] = info['y_pred_orig']

pred_df = pd.DataFrame(pred_data)

# Ghi header
for col_idx, col_name in enumerate(pred_df.columns, 1):
    ws5.cell(row=1, column=col_idx, value=col_name)

# Ghi data
for row_idx, row in enumerate(pred_df.itertuples(index=False), 2):
    for col_idx, value in enumerate(row, 1):
        ws5.cell(row=row_idx, column=col_idx, value=round(float(value), 4))

# Format số tiền
usd_cols = [col_idx for col_idx, col_name in enumerate(pred_df.columns, 1) if '($)' in col_name]
for row_idx in range(2, len(pred_df) + 2):
    for col_idx in usd_cols:
        ws5.cell(row=row_idx, column=col_idx).number_format = '#,##0.00'

style_header(ws5, len(pred_df.columns))
auto_width(ws5)

print('  [OK] Sheet: Prediction Details')


# =============================================
# SAVE EXCEL
# =============================================

wb.save(EXCEL_PATH)
print(f'\n  >>> Excel saved: {EXCEL_PATH}')


# =============================================
# DONE
# =============================================

print(f'\n{"=" * 60}')
print('  DONE! All evaluations completed.')
print(f'{"=" * 60}')
print(f'\n  Output files:')
print(f'    Excel  : {EXCEL_PATH}')
print(f'    Plots  : {EVAL_DIR}/')
print(f'      - actual_vs_predicted.png')
print(f'      - residual_analysis.png')
print(f'      - feature_importance.png')
print(f'      - metrics_comparison.png')
