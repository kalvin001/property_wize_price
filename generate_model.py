import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import json
import argparse
from collections import OrderedDict
import sys

# 添加models目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型模块
from models import ModelFactory

# 解析命令行参数
parser = argparse.ArgumentParser(description='生成预测模型')
parser.add_argument('--model_type', type=str, default='xgboost',
                   help='模型类型，支持xgboost, linear, ridge, lasso, elasticnet')
parser.add_argument('--params_file', type=str, default='model/best_params.json',
                   help='模型参数文件路径')
args = parser.parse_args()

print(f"开始使用最佳参数生成{args.model_type}模型...")

# 确保输出目录存在
os.makedirs('model', exist_ok=True)
os.makedirs('frontend/public/data', exist_ok=True)

# 加载最佳参数
if os.path.exists(args.params_file):
    with open(args.params_file, 'r') as f:
        best_params = json.load(f)
    print(f"使用参数文件: {args.params_file}")
else:
    # 默认参数
    if args.model_type == 'xgboost':
        best_params = OrderedDict([
            ('colsample_bytree', 0.7),
            ('gamma', 0.026),
            ('learning_rate', 0.16),
            ('max_depth', 5),
            ('min_child_weight', 4),
            ('n_estimators', 154),
            ('reg_alpha', 0.0005),
            ('reg_lambda', 0.05),
            ('subsample', 0.69)
        ])
    elif args.model_type == 'ridge':
        best_params = {
            'alpha': 1.0
        }
    elif args.model_type == 'lasso':
        best_params = {
            'alpha': 0.01
        }
    elif args.model_type == 'elasticnet':
        best_params = {
            'alpha': 0.01,
            'l1_ratio': 0.5
        }
    else:  # linear
        best_params = {}
    
    print("未找到参数文件，使用默认参数")

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
        print("存在特征选择结果，但本次将使用全部特征")
except Exception as e:
    print(f"无法加载特征选择结果: {e}")

# 使用所有特征
selected_features = feature_cols
print(f"使用所有特征，共 {len(selected_features)} 个")
joblib.dump(selected_features, 'model/selected_features.joblib')

# 使用选择的特征训练模型
print(f"使用选择的特征和最佳参数训练{args.model_type}模型...")
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 创建模型实例
try:
    model = ModelFactory.create_model(args.model_type, **best_params)
    print(f"成功创建{args.model_type}模型")
except Exception as e:
    print(f"创建{args.model_type}模型失败: {e}")
    print("将使用默认的XGBoost模型")
    model = ModelFactory.create_model("xgboost")

# 训练模型
model.train(X_train_selected, y_train)
print("模型训练完成")

# 预测并评估
pred = model.predict(X_test_selected)
metrics = model.evaluate(X_test_selected, y_test)

# 保存模型
model_path = f'model/{args.model_type}_model.joblib'
model.save(model_path)
print(f"模型已保存到: {model_path}")

# 保存最佳参数
with open(f'model/{args.model_type}_best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

# 输出评估结果
print(f"模型评估结果:")
print(f"- RMSE: {metrics['rmse']}")
print(f"- MAE: {metrics['mae']}")
print(f"- R²: {metrics['r2']}")

# 获取特征重要性
feature_importance = model.get_feature_importance()
print("len(feature_importance):", len(feature_importance))

print("\n前10个重要特征:")
for i, (feature, importance) in enumerate(zip(feature_importance['feature'].head(10), 
                                            feature_importance['importance'].head(10))):
    print(f"{i+1}. {feature}: {importance:.4f}")

# 保存模型指标
model_metrics = {
    'performance': {
        'rmse': float(metrics['rmse']),
        'mae': float(metrics['mae']),
        'r2': float(metrics['r2'])
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
print(f"- 模型类型: {args.model_type}")
print(f"- 模型已保存到: {model_path}")
print("- 模型指标已保存到: frontend/public/data/model_metrics.json") 