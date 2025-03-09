import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFECV
import shap
import json
import os
import joblib
from pathlib import Path
import time
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import VerboseCallback
import argparse
import datetime

# 解析命令行参数
parser = argparse.ArgumentParser(description='房产价格预测模型训练与优化')
parser.add_argument('--optimization', type=str, default='bayes', choices=['grid', 'bayes'],
                    help='参数优化方法: grid (网格搜索) 或 bayes (贝叶斯优化)')
parser.add_argument('--feature_selection', action='store_true', default=True,
                    help='是否进行特征选择')
parser.add_argument('--cv_folds', type=int, default=3,
                    help='交叉验证折数')
parser.add_argument('--bayes_iterations', type=int, default=20,
                    help='贝叶斯优化迭代次数')
parser.add_argument('--min_features', type=int, default=10,
                    help='最小选择特征数量')

args = parser.parse_args()

print(f"参数配置:")
print(f"- 优化方法: {args.optimization}")
print(f"- 特征选择: {'是' if args.feature_selection else '否'}")
print(f"- 交叉验证折数: {args.cv_folds}")
print(f"- 贝叶斯优化迭代次数: {args.bayes_iterations}")
print(f"- 最小特征数量: {args.min_features}")

# 创建目录
os.makedirs('model', exist_ok=True)
os.makedirs('frontend/public/data', exist_ok=True)
os.makedirs('model/optimizer_logs', exist_ok=True)  # 创建优化器日志目录

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

# 创建优化日志目录和文件
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
optimizer_log_dir = f"model/optimizer_logs/{timestamp}_{args.optimization}"
os.makedirs(optimizer_log_dir, exist_ok=True)

optimization_log = {
    "method": args.optimization,
    "start_time": datetime.datetime.now().isoformat(),
    "parameters": vars(args),
    "iterations": []
}

# 回调函数：记录每次搜索的参数和结果
class OptimizationCallback:
    def __init__(self, log_file):
        self.log_file = log_file
        self.iteration = 0
        self.start_time = time.time()
        
    def __call__(self, res):
        global optimization_log
        
        elapsed = time.time() - self.start_time
        
        # 提取当前迭代的信息
        current_params = res.x_iters[-1] if len(res.x_iters) > 0 else {}
        current_score = res.func_vals[-1] if len(res.func_vals) > 0 else None
        
        # 创建迭代记录
        if args.optimization == "bayes":
            params_dict = {}
            for i, param_name in enumerate(res.space.dimension_names):
                params_dict[param_name] = current_params[i] if i < len(current_params) else None
        else:
            params_dict = current_params
        
        iteration_record = {
            "iteration": self.iteration,
            "elapsed_seconds": elapsed,
            "parameters": params_dict,
            "score": float(-current_score) if current_score is not None else None
        }
        
        # 添加到日志
        optimization_log["iterations"].append(iteration_record)
        
        # 保存日志
        with open(f"{optimizer_log_dir}/optimization_log.json", 'w') as f:
            json.dump(optimization_log, f, indent=2)
        
        # 打印进度
        if current_score is not None:
            print(f"迭代 {self.iteration}: RMSE = {-current_score:.4f}, 用时: {elapsed:.2f}秒")
        
        self.iteration += 1
        return True

# 进行参数搜索优化
print(f"开始参数优化搜索 (方法: {args.optimization})...")
start_time = time.time()

if args.optimization == "grid":
    # 减少网格搜索参数空间
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_alpha': [0, 1],
        'reg_lambda': [0, 1]
    }
    
    # 基础模型
    base_model = XGBRegressor(random_state=42, n_jobs=-1)
    
    # 设置GridSearchCV
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=args.cv_folds,
        verbose=2,  # 增加详细程度以展示进度
        n_jobs=-1,
        return_train_score=True  # 记录训练分数
    )
else:  # 贝叶斯优化
    # 减少贝叶斯优化参数空间
    param_space = {
        'n_estimators': Integer(50, 300),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'max_depth': Integer(3, 8),
        'subsample': Real(0.6, 0.9),
        'colsample_bytree': Real(0.6, 0.9),
        'reg_alpha': Real(1e-5, 1.0, prior='log-uniform'),
        'reg_lambda': Real(1e-5, 1.0, prior='log-uniform'),
        'min_child_weight': Integer(1, 5),
        'gamma': Real(1e-5, 0.5, prior='log-uniform')
    }
    
    # 基础模型
    base_model = XGBRegressor(random_state=42, n_jobs=-1)
    
    # 创建回调
    optimization_callback = OptimizationCallback(f"{optimizer_log_dir}/optimization_log.json")
    
    # 设置BayesSearchCV
    search = BayesSearchCV(
        estimator=base_model,
        search_spaces=param_space,
        n_iter=args.bayes_iterations,
        scoring='neg_root_mean_squared_error',
        cv=args.cv_folds,
        verbose=3,  # 增加详细程度以展示进度
        n_jobs=-1,
        random_state=42,
        return_train_score=True  # 记录训练分数
    )

# 执行参数搜索
callbacks = [VerboseCallback(1), optimization_callback]
try:
    # 首先尝试使用 callback 参数
    search.fit(X_train, y_train, callback=callbacks)
except TypeError:
    try:
        # 如果失败，尝试使用 callbacks 参数
        search.fit(X_train, y_train, callbacks=callbacks)
    except TypeError:
        # 如果两者都失败，则不使用回调
        print("警告：无法添加回调函数，继续执行参数搜索...")
        search.fit(X_train, y_train)

optimization_end_time = time.time()
optimization_log["end_time"] = datetime.datetime.now().isoformat()
optimization_log["total_seconds"] = optimization_end_time - start_time
optimization_log["best_parameters"] = search.best_params_
optimization_log["best_score"] = float(-search.best_score_)

# 保存最终优化日志
with open(f"{optimizer_log_dir}/optimization_log.json", 'w') as f:
    json.dump(optimization_log, f, indent=2)

# 打印参数搜索结果
print(f"参数搜索完成，耗时：{optimization_end_time - start_time:.2f}秒")
print(f"最佳参数: {search.best_params_}")
print(f"最佳得分: {-search.best_score_}")

# 保存最佳参数
with open('model/best_params.json', 'w') as f:
    json.dump(search.best_params_, f, indent=2)

# 保存搜索历史结果
cv_results = pd.DataFrame(search.cv_results_)
cv_results.to_csv(f"{optimizer_log_dir}/cv_results.csv", index=False)

# 特征选择
selected_features = feature_cols.copy()  # 默认使用所有特征
feature_selection_time = 0

if args.feature_selection:
    print("\n开始特征选择...")
    feature_selection_start_time = time.time()

    # 使用最佳参数创建模型用于特征选择
    model_for_feature_selection = XGBRegressor(**search.best_params_, random_state=42, n_jobs=-1)

    # 递归特征消除与交叉验证
    rfecv = RFECV(
        estimator=model_for_feature_selection,
        step=1,
        cv=args.cv_folds,
        scoring='neg_root_mean_squared_error',
        min_features_to_select=args.min_features,
        n_jobs=-1,
        verbose=1
    )

    rfecv.fit(X_train, y_train)

    # 获取选择的特征
    selected_features = X_train.columns[rfecv.support_].tolist()
    feature_selection_time = time.time() - feature_selection_start_time
    print(f"特征选择完成，从{X_train.shape[1]}个特征中选择了{len(selected_features)}个特征")
    print(f"特征选择耗时：{feature_selection_time:.2f}秒")

    # 保存特征选择结果
    feature_selection_results = {
        "total_features": len(feature_cols),
        "selected_features_count": len(selected_features),
        "selected_features": selected_features,
        "feature_ranking": rfecv.ranking_.tolist(),
        "cv_scores": rfecv.cv_results_['mean_test_score'].tolist() if hasattr(rfecv, 'cv_results_') else []
    }
    
    with open(f"{optimizer_log_dir}/feature_selection_results.json", 'w') as f:
        json.dump(feature_selection_results, f, indent=2)

    # 保存选择的特征
    joblib.dump(selected_features, 'model/selected_features.joblib')
    with open('frontend/public/data/selected_features.json', 'w') as f:
        json.dump(selected_features, f, indent=2)
else:
    print("\n跳过特征选择，使用所有特征...")
    # 保存全部特征
    joblib.dump(selected_features, 'model/selected_features.joblib')
    with open('frontend/public/data/selected_features.json', 'w') as f:
        json.dump(selected_features, f, indent=2)

# 使用选择的特征训练模型
print("\n使用选择的特征和最佳参数训练XGBoost模型...")
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 使用最佳参数训练XGBoost模型
xgb_model = XGBRegressor(**search.best_params_, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_selected, y_train)
xgb_pred = xgb_model.predict(X_test_selected)

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
    'feature': selected_features,
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
shap_sample_indices = np.random.choice(len(X_test_selected), size=min(100, len(X_test_selected)), replace=False)
# 确保使用带特征名称的DataFrame
shap_values = explainer.shap_values(X_test_selected.iloc[shap_sample_indices])

# 保存全局SHAP摘要数据
shap_summary = []
for i, feature in enumerate(selected_features):
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
    sample_idx = X_test_selected.index[idx]
    sample = X_test_selected.iloc[idx]
    # 使用带特征名称的DataFrame进行预测
    pred_price = xgb_model.predict(sample.to_frame().T)[0]
    actual_price = y_test.iloc[idx]
    
    # 获取此样本的SHAP值
    sample_shap = shap_values[np.where(shap_sample_indices == idx)[0][0]]
    
    # 获取影响最大的5个特征
    top_shap_indices = np.argsort(np.abs(sample_shap))[-5:]
    top_features_impact = [{
        'feature': selected_features[i],
        'value': float(sample[selected_features[i]]),
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

# 如果进行了特征选择，比较使用全部特征和选择特征的性能
compare_metrics = {}
if args.feature_selection:
    print("\n比较使用全部特征和选择特征的性能...")
    # 使用全部特征训练一个模型
    xgb_model_all_features = XGBRegressor(**search.best_params_, random_state=42, n_jobs=-1)
    xgb_model_all_features.fit(X_train, y_train)
    xgb_pred_all_features = xgb_model_all_features.predict(X_test)

    # 评估使用全部特征的模型
    xgb_rmse_all_features = np.sqrt(mean_squared_error(y_test, xgb_pred_all_features))
    xgb_mae_all_features = mean_absolute_error(y_test, xgb_pred_all_features)
    xgb_r2_all_features = r2_score(y_test, xgb_pred_all_features)

    print(f"全部特征 - RMSE: {xgb_rmse_all_features}")
    print(f"全部特征 - MAE: {xgb_mae_all_features}")
    print(f"全部特征 - R²: {xgb_r2_all_features}")

    print(f"选择特征 - RMSE: {xgb_rmse}")
    print(f"选择特征 - MAE: {xgb_mae}")
    print(f"选择特征 - R²: {xgb_r2}")
    
    compare_metrics = {
        'all_features': {
            'rmse': float(xgb_rmse_all_features),
            'mae': float(xgb_mae_all_features),
            'r2_score': float(xgb_r2_all_features)
        },
        'selected_features': {
            'rmse': float(xgb_rmse),
            'mae': float(xgb_mae),
            'r2_score': float(xgb_r2)
        }
    }

# 计算交叉验证性能
print("\n进行交叉验证评估...")
cv_scores = cross_val_score(
    xgb_model, 
    X_train_selected, 
    y_train, 
    cv=5, 
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
cv_rmse = -cv_scores.mean()
print(f"交叉验证 RMSE: {cv_rmse}")

# 保存模型评估指标
model_metrics = {
    'rmse': float(xgb_rmse),
    'mae': float(xgb_mae),
    'r2_score': float(xgb_r2),
    'mse': float(xgb_mse),
    'median_ae': float(np.median(np.abs(y_test - xgb_pred))),
    'explained_variance': float(np.var(y_test - xgb_pred) / np.var(y_test)),
    'cv_rmse': float(cv_rmse),
    
    # 保存预测值和实际值（转换为列表以便JSON序列化）
    'predictions': xgb_pred.tolist(),
    'actual': y_test.tolist(),
    
    # 计算中位百分比误差和平均百分比误差
    'median_percentage_error': float(np.median(np.abs((xgb_pred - y_test) / y_test) * 100)),
    'mean_percentage_error': float(np.mean(np.abs((xgb_pred - y_test) / y_test) * 100)),
    
    # 计算误差分布
    'error_distribution': {
        'percentiles': {
            'p10': float(np.percentile(np.abs((xgb_pred - y_test) / y_test) * 100, 10)),
            'p25': float(np.percentile(np.abs((xgb_pred - y_test) / y_test) * 100, 25)),
            'p50': float(np.percentile(np.abs((xgb_pred - y_test) / y_test) * 100, 50)),
            'p75': float(np.percentile(np.abs((xgb_pred - y_test) / y_test) * 100, 75)),
            'p90': float(np.percentile(np.abs((xgb_pred - y_test) / y_test) * 100, 90))
        },
        'error_ranges': {
            '<5%': float(np.mean(np.abs((xgb_pred - y_test) / y_test) * 100 < 5)),
            '5-10%': float(np.mean((np.abs((xgb_pred - y_test) / y_test) * 100 >= 5) & (np.abs((xgb_pred - y_test) / y_test) * 100 < 10))),
            '10-15%': float(np.mean((np.abs((xgb_pred - y_test) / y_test) * 100 >= 10) & (np.abs((xgb_pred - y_test) / y_test) * 100 < 15))),
            '15-20%': float(np.mean((np.abs((xgb_pred - y_test) / y_test) * 100 >= 15) & (np.abs((xgb_pred - y_test) / y_test) * 100 < 20))),
            '>20%': float(np.mean(np.abs((xgb_pred - y_test) / y_test) * 100 >= 20))
        }
    },
    
    # 添加特征重要性
    'feature_importance': [{
        'feature': feature,
        'importance': float(importance)
    } for feature, importance in zip(
        feature_importance['feature'].head(20), 
        feature_importance['importance'].head(20)
    )],
    
    # 添加参数搜索结果
    'optimization': {
        'method': args.optimization,
        'best_params': search.best_params_,
        'best_score': float(-search.best_score_),
        'search_time_seconds': float(optimization_end_time - start_time),
        'log_dir': optimizer_log_dir
    },
    
    # 添加参数配置
    'config': {
        'optimization_method': args.optimization,
        'feature_selection': args.feature_selection,
        'cv_folds': args.cv_folds,
        'bayes_iterations': args.bayes_iterations,
        'min_features': args.min_features
    }
}

# 如果进行了特征选择，添加特征选择结果
if args.feature_selection:
    model_metrics['feature_selection'] = {
        'total_features': len(feature_cols),
        'selected_features_count': len(selected_features),
        'selected_features': selected_features,
        'feature_selection_time_seconds': float(feature_selection_time)
    }
    model_metrics['comparison'] = compare_metrics

with open('frontend/public/data/model_metrics.json', 'w') as f:
    json.dump(model_metrics, f, indent=2)

print("分析完成，结果已保存到 frontend/public/data/ 目录，模型已保存到 model/ 目录")
print(f"优化日志保存在: {optimizer_log_dir}") 