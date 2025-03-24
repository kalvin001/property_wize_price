import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json

# 设置路径
data_file = 'resources/house_samples_engineered_original.csv'
output_dir = 'model_original'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
print("读取数据...")
df = pd.read_csv(data_file)
print(f"数据集形状: {df.shape}")

# 准备数据
print("准备数据...")
# 排除ID列和目标列
X = df.drop(['y_label'], axis=1)
if 'prop_id' in X.columns:
    X = X.drop(['prop_id'], axis=1)
y = df['y_label']

# 处理分类特征和对象类型
object_cols = X.select_dtypes(include=['object']).columns
print(f"处理 {len(object_cols)} 个对象类型列...")

# 丢弃复杂的文本列
text_cols_to_drop = ['std_address', 'full_address_x', 'full_address_y', 'house_name', 'location_summary', 'key_advantages']
text_cols_to_drop = [col for col in text_cols_to_drop if col in X.columns]
if text_cols_to_drop:
    print(f"丢弃文本列: {', '.join(text_cols_to_drop)}")
    X = X.drop(text_cols_to_drop, axis=1)

# 对剩余的对象类型列进行编码
encoder_dict = {}
object_cols = X.select_dtypes(include=['object']).columns
for col in object_cols:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col].astype(str))
    encoder_dict[col] = encoder

# 处理数值特征
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_cols] = X[numeric_cols].fillna(0)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 训练XGBoost模型
print("训练XGBoost模型...")
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# 评估模型
print("评估模型...")
y_pred = model.predict(X_test)

# 计算评估指标
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 计算百分比误差
abs_pct_errors = np.abs((y_test - y_pred) / y_test) * 100
mean_pct_error = np.mean(abs_pct_errors)
median_pct_error = np.median(abs_pct_errors)

# 计算误差分布
error_ranges = {
    "<5%": np.mean(abs_pct_errors < 5),
    "5-10%": np.mean((abs_pct_errors >= 5) & (abs_pct_errors < 10)),
    "10-15%": np.mean((abs_pct_errors >= 10) & (abs_pct_errors < 15)),
    "15-20%": np.mean((abs_pct_errors >= 15) & (abs_pct_errors < 20)),
    ">20%": np.mean(abs_pct_errors >= 20)
}

# 获取特征重要性
feature_importance = []
for i, feature in enumerate(X.columns):
    feature_importance.append({
        "feature": feature,
        "importance": float(model.feature_importances_[i])
    })
feature_importance.sort(key=lambda x: x["importance"], reverse=True)

# 保存结果
print("保存结果...")
# 保存模型
joblib.dump(model, os.path.join(output_dir, "xgboost_model.joblib"))
# 保存编码器
joblib.dump(encoder_dict, os.path.join(output_dir, "encoders.joblib"))

# 保存特征重要性
feature_importance_df = pd.DataFrame(feature_importance)
feature_importance_df.to_csv(os.path.join(output_dir, "xgboost_feature_importance.csv"), index=False)

# 保存评估指标
metrics = {
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2),
    "mean_percentage_error": float(mean_pct_error),
    "median_percentage_error": float(median_pct_error),
    "error_distribution": {
        "error_ranges": error_ranges
    },
    "feature_importance": feature_importance[:20]  # 保存前20个重要特征
}

with open(os.path.join(output_dir, "xgboost_metrics.json"), 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

# 输出结果
print("\nXGBoost模型训练和评估完成!")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
print(f"平均百分比误差: {mean_pct_error:.2f}%")
print(f"中位百分比误差: {median_pct_error:.2f}%")

print("\n误差分布:")
for range_name, value in error_ranges.items():
    print(f"{range_name}: {value*100:.2f}%")

print("\n前10个重要特征:")
for i, feat in enumerate(feature_importance[:10]):
    print(f"{i+1}. {feat['feature']} (重要性: {feat['importance']:.2e})") 