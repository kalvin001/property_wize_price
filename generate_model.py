import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import json
from collections import OrderedDict

print("开始使用最佳参数生成模型...")

# 确保输出目录存在
os.makedirs('model', exist_ok=True)
os.makedirs('frontend/public/data', exist_ok=True)

# 最佳参数
best_params = OrderedDict([
    ('colsample_bytree', 0.7230311876559942), 
    ('gamma', 0.026276132221638653), 
    ('learning_rate', 0.16356457461011642), 
    ('max_depth', 5), 
    ('min_child_weight', 4), 
    ('n_estimators', 154), 
    ('reg_alpha', 0.0005684034097210145), 
    ('reg_lambda', 0.04983347481602247), 
    ('subsample', 0.6913389933109518)
])

print(f"使用的最佳参数: {best_params}")

# 读取数据
print("读取数据...")
df = pd.read_csv('resources/house_samples_features.csv')

# 排除不应作为特征的列
exclude_cols = ['prop_id', 'std_address', 'y_label']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# 保存特征列名
joblib.dump(feature_cols, 'model/feature_cols.joblib')

# 数据准备
print("准备数据进行建模...")
X = df[feature_cols].copy()
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

# 检查是否存在已保存的特征选择结果
selected_features = None
try:
    if os.path.exists('model/selected_features.joblib'):
        # selected_features = joblib.load('model/selected_features.joblib')
        # print(f"已加载选定的特征，共 {len(selected_features)} 个")
        print("存在特征选择结果，但本次将使用全部特征")
except Exception as e:
    print(f"无法加载特征选择结果: {e}")

# 如果没有特征选择结果，则使用所有特征
# if selected_features is None:
selected_features = feature_cols
print(f"使用所有特征，共 {len(selected_features)} 个")
joblib.dump(selected_features, 'model/selected_features.joblib')

# 使用选择的特征训练模型
print("使用选择的特征和最佳参数训练XGBoost模型...")
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 使用最佳参数训练XGBoost模型
xgb_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_selected, y_train)
xgb_pred = xgb_model.predict(X_test_selected)

# 保存XGBoost模型
joblib.dump(xgb_model, 'model/xgb_model.joblib')

# 保存最佳参数
with open('model/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

# 评估XGBoost模型
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"模型评估结果:")
print(f"- RMSE: {xgb_rmse}")
print(f"- MAE: {xgb_mae}")
print(f"- R²: {xgb_r2}")

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': xgb_model.feature_importances_
})
print("len(feature_importance):", len(feature_importance))
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\n前10个重要特征:")
for i, (feature, importance) in enumerate(zip(feature_importance['feature'].head(10), 
                                            feature_importance['importance'].head(10))):
    print(f"{i+1}. {feature}: {importance:.4f}")

# 保存模型指标
model_metrics = {
    'performance': {
        'rmse': float(xgb_rmse),
        'mae': float(xgb_mae),
        'r2': float(xgb_r2)
    },
    'parameters': best_params,
    'feature_importance': [{
        'feature': feature,
        'importance': float(importance)
    } for feature, importance in zip(
        feature_importance['feature'].head(20), 
        feature_importance['importance'].head(20)
    )]
}

if os.path.exists('frontend/public/data/model_metrics.json'):
    try:
        with open('frontend/public/data/model_metrics.json', 'r') as f:
            existing_metrics = json.load(f)
        # 更新性能指标
        existing_metrics['performance'] = model_metrics['performance']
        existing_metrics['parameters'] = model_metrics['parameters']
        existing_metrics['feature_importance'] = model_metrics['feature_importance']
        model_metrics = existing_metrics
    except Exception as e:
        print(f"读取现有指标文件时出错: {e}")

with open('frontend/public/data/model_metrics.json', 'w') as f:
    json.dump(model_metrics, f, indent=2)

print("\n模型生成完成!")
print("- 模型已保存到: model/xgb_model.joblib")
print("- 模型指标已保存到: frontend/public/data/model_metrics.json") 