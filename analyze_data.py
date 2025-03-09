import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import json
import os
import joblib
from pathlib import Path

# 创建目录
os.makedirs('model', exist_ok=True)
os.makedirs('frontend/public/data', exist_ok=True)

# 读取数据
print("正在读取数据...")
df = pd.read_csv('resources/house_samples_features.csv')

# 显示基本信息
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()[:20]}...")

# 排除不应作为特征的列
exclude_cols = ['prop_id', 'std_address', 'y_label']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# 保存特征列名
joblib.dump(feature_cols, 'model/feature_cols.joblib')

# 基本数据描述
print("\n数据描述:")
desc = df.describe(include='all')
print(desc.head())

# 查看缺失值
print("\n缺失值统计:")
missing = df[feature_cols].isnull().sum()
print(missing[missing > 0])

# 保存数据信息到JSON文件
data_info = {
    "total_records": len(df),
    "features_count": len(feature_cols),
    "target_column": "y_label",
    "price_range": {
        "min": float(df['y_label'].min()),
        "max": float(df['y_label'].max()),
        "mean": float(df['y_label'].mean()),
        "median": float(df['y_label'].median())
    }
}

with open('frontend/public/data/data_info.json', 'w') as f:
    json.dump(data_info, f, indent=2)

# 数据准备
print("\n准备数据进行建模...")
X = df[feature_cols].copy()  # 确保使用副本
y = df['y_label'].copy()

# 处理缺失值
X = X.fillna(X.median())

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将缩放后的数据转换回DataFrame以保留特征名称
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# 保存特征标准化器
joblib.dump(scaler, 'model/scaler.joblib')

# 训练XGBoost模型
print("训练XGBoost模型...")
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# 保存XGBoost模型
joblib.dump(xgb_model, 'model/xgb_model.joblib')

# 评估XGBoost模型
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"XGBoost RMSE: {xgb_rmse}")
print(f"XGBoost MAE: {xgb_mae}")
print(f"XGBoost R²: {xgb_r2}")

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# 保存特征重要性
top_features = feature_importance.head(20).to_dict(orient='records')
with open('frontend/public/data/feature_importance.json', 'w') as f:
    json.dump(top_features, f, indent=2)

# 为样本数据生成SHAP值
print("计算SHAP值...")
explainer = shap.TreeExplainer(xgb_model)
# 选择少量样本以计算SHAP值
shap_sample_indices = np.random.choice(len(X_test), size=min(100, len(X_test)), replace=False)
# 确保使用带特征名称的DataFrame
shap_values = explainer.shap_values(X_test.iloc[shap_sample_indices])

# 保存全局SHAP摘要数据
shap_summary = []
for i, feature in enumerate(feature_cols):
    if i < len(shap_values[0]):  # 确保索引在范围内
        shap_summary.append({
            'feature': feature,
            'mean_abs_shap': float(np.abs(shap_values[:, i]).mean()),
            'min_shap': float(np.min(shap_values[:, i])),
            'max_shap': float(np.max(shap_values[:, i]))
        })

# 按平均绝对SHAP值排序
shap_summary = sorted(shap_summary, key=lambda x: x['mean_abs_shap'], reverse=True)
with open('frontend/public/data/shap_summary.json', 'w') as f:
    json.dump(shap_summary[:20], f, indent=2)  # 只保存前20个特征

# 生成样本房产的预测和解释
sample_properties = []
for idx in shap_sample_indices[:5]:  # 只取前5个样本
    sample_idx = X_test.index[idx]
    sample = X_test.iloc[idx]
    # 使用带特征名称的DataFrame进行预测
    pred_price = xgb_model.predict(sample.to_frame().T)[0]
    actual_price = y_test.iloc[idx]
    
    # 获取此样本的SHAP值
    sample_shap = shap_values[np.where(shap_sample_indices == idx)[0][0]]
    
    # 获取影响最大的5个特征
    top_shap_indices = np.argsort(np.abs(sample_shap))[-5:]
    top_features_impact = [{
        'feature': feature_cols[i],
        'value': float(sample[feature_cols[i]]),
        'shap_value': float(sample_shap[i]),
        'is_positive': bool(sample_shap[i] > 0)
    } for i in top_shap_indices]
    
    # 获取房产标识
    prop_id = df.loc[sample_idx, 'prop_id']
    address = df.loc[sample_idx, 'std_address']
    
    sample_properties.append({
        'prop_id': str(prop_id),
        'address': address,
        'predicted_price': float(pred_price),
        'actual_price': float(actual_price),
        'error_percent': float(100 * (pred_price - actual_price) / actual_price),
        'top_features': top_features_impact
    })

with open('frontend/public/data/sample_properties.json', 'w') as f:
    json.dump(sample_properties, f, indent=2)

# 保存模型评估指标
model_metrics = {
    'xgboost': {
        'rmse': float(xgb_rmse),
        'mae': float(xgb_mae),
        'r2': float(xgb_r2)
    }
}
with open('frontend/public/data/model_metrics.json', 'w') as f:
    json.dump(model_metrics, f, indent=2)

print("分析完成，结果已保存到 frontend/public/data/ 目录，模型已保存到 model/ 目录") 