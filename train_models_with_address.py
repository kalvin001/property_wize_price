import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
import json
import argparse
import time
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 导入模型模块
from models import ModelFactory

def train_and_evaluate_model(model_type, data_file, output_dir='model_with_address', 
                           feature_selection=False, test_size=0.2, random_state=42, params=None):
    """
    训练并评估指定类型的模型
    
    Args:
        model_type: 模型类型，如'xgboost', 'lightgbm', 'catboost'
        data_file: 数据文件路径
        output_dir: 输出目录
        feature_selection: 是否使用特征选择
        test_size: 测试集比例
        random_state: 随机种子
        params: 模型参数字典（可选）
    
    Returns:
        模型评估指标
    """
    print(f"开始训练和评估 {model_type} 模型...")
    start_time = time.time()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print(f"读取数据: {data_file}")
    df = pd.read_csv(data_file)
    
    # 排除不应作为特征的列
    exclude_cols = ['prop_id', 'std_address', 'y_label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 数据准备
    print("准备数据...")
    X = df[feature_cols].copy()
    y = df['y_label'].copy()
    
    # 处理缺失值
    X = X.fillna(X.median())
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 将缩放后的数据转换回DataFrame以保留特征名称
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # 创建模型
    print(f"创建{model_type}模型...")
    if params:
        model = ModelFactory.create_model(model_type, **params)
        print(f"使用自定义参数: {params}")
    else:
        model = ModelFactory.create_model(model_type)
        print("使用默认参数")
    
    # 特征选择（可选）
    selected_features = None
    if feature_selection:
        print("执行特征选择...")
        from sklearn.feature_selection import SelectFromModel
        
        # 使用模型的特征重要性进行选择
        selector = SelectFromModel(model.get_base_estimator(), threshold='mean')
        selector.fit(X_train_scaled_df, y_train)
        selected_features = X_train_scaled_df.columns[selector.get_support()].tolist()
        
        print(f"选择了 {len(selected_features)} 个特征 (从 {X_train_scaled_df.shape[1]} 个特征中)")
        
        # 只使用选定的特征
        X_train_scaled_df = X_train_scaled_df[selected_features]
        X_test_scaled_df = X_test_scaled_df[selected_features]
    
    # 训练模型
    print("训练模型...")
    model.train(X_train_scaled_df, y_train)
    
    # 评估模型
    print("评估模型...")
    metrics = model.evaluate(X_test_scaled_df, y_test)
    
    # 使用模型进行预测
    y_pred = model.predict(X_test_scaled_df)
    
    # 计算详细的评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 计算更多详细的指标
    percent_errors = np.abs((y_test - y_pred) / y_test) * 100
    median_ae = np.median(np.abs(y_test - y_pred))
    median_percent_error = np.median(percent_errors)
    mean_percent_error = np.mean(percent_errors)
    
    # 计算误差的分布情况
    error_p10 = np.percentile(percent_errors, 10)
    error_p25 = np.percentile(percent_errors, 25)
    error_p50 = np.percentile(percent_errors, 50)
    error_p75 = np.percentile(percent_errors, 75)
    error_p90 = np.percentile(percent_errors, 90)
    
    # 计算不同误差范围内的样本比例
    error_ranges = {
        "<5%": np.mean(percent_errors < 5),
        "5-10%": np.mean((percent_errors >= 5) & (percent_errors < 10)),
        "10-15%": np.mean((percent_errors >= 10) & (percent_errors < 15)),
        "15-20%": np.mean((percent_errors >= 15) & (percent_errors < 20)),
        ">20%": np.mean(percent_errors >= 20)
    }
    
    # 获取特征重要性
    feature_importance = model.get_feature_importance()
    
    # 检查地址向量特征的重要性
    address_features = [col for col in X_train_scaled_df.columns if 'address_vec' in col]
    address_importance = {}
    
    if address_features:
        print("\n地址向量特征的重要性:")
        for feature in address_features:
            if feature in feature_importance['feature'].values:
                importance = feature_importance.loc[feature_importance['feature'] == feature, 'importance'].values[0]
                address_importance[feature] = importance
                print(f"  - {feature}: {importance:.6f}")
    
    # 计算无特征选择时的性能（使用所有特征）
    all_features_metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2)
    }
    
    # 当前使用选择的特征的性能
    selected_features_metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2)
    }
    
    # 获取模型参数
    if params:
        model_params = params
    else:
        model_params = model.get_params()
    
    # 创建与之前格式一致的详细指标
    detailed_metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2),
        "mse": float(mse),
        "median_ae": float(median_ae),
        "explained_variance": float(1.0 - (np.var(y_test - y_pred) / np.var(y_test))),
        "cv_rmse": float(rmse),  # 这里没有做交叉验证，所以使用相同的rmse
        "predictions": y_pred.tolist()[:10],  # 只存储前10个预测结果，避免文件过大
        "median_percentage_error": float(median_percent_error),
        "mean_percentage_error": float(mean_percent_error),
        "error_distribution": {
            "percentiles": {
                "p10": float(error_p10),
                "p25": float(error_p25),
                "p50": float(error_p50),
                "p75": float(error_p75),
                "p90": float(error_p90)
            },
            "error_ranges": error_ranges
        },
        "feature_importance": [
            {
                "feature": row["feature"],
                "importance": float(row["importance"])
            }
            for _, row in feature_importance.head(20).iterrows()
        ],
        "address_features_importance": address_importance,
        "comparison": {
            "all_features": all_features_metrics,
            "selected_features": selected_features_metrics
        },
        "performance": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        },
        "parameters": model_params
    }
    
    # 保存模型
    model_path = os.path.join(output_dir, f"{model_type}_model.joblib")
    model.save(model_path)
    
    # 保存特征重要性
    importance_path = os.path.join(output_dir, f"{model_type}_feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    
    # 保存详细评估指标
    metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top 20 Feature Importance - {model_type}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type}_feature_importance.png"))
    
    # 计算训练时间
    elapsed_time = time.time() - start_time
    
    print(f"\n{model_type}模型训练和评估完成!")
    print(f"- 训练时间: {elapsed_time:.2f}秒")
    print(f"- RMSE: {metrics['rmse']:.4f}")
    print(f"- MAE: {metrics['mae']:.4f}")
    print(f"- R²: {metrics['r2']:.4f}")
    print(f"- 平均百分比误差: {mean_percent_error:.2f}%")
    print(f"- 中位百分比误差: {median_percent_error:.2f}%")
    print(f"- 模型保存路径: {model_path}")
    print(f"- 特征重要性保存路径: {importance_path}")
    print(f"- 评估指标保存路径: {metrics_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='训练和评估基于地址向量的模型')
    parser.add_argument('--model_types', nargs='+', default=['xgboost'],
                       help='要训练的模型类型列表，如xgboost, lightgbm, catboost')
    parser.add_argument('--data_file', type=str, default='resources/house_samples_engineered_with_address.csv',
                       help='包含地址向量特征的数据文件路径')
    parser.add_argument('--output_dir', type=str, default='model_with_address',
                       help='输出目录')
    parser.add_argument('--feature_selection', action='store_true',
                       help='是否使用特征选择')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练和评估每种模型
    results = {}
    for model_type in args.model_types:
        try:
            print(f"\n{'='*50}")
            print(f"训练和评估 {model_type} 模型 (带地址向量特征)")
            print(f"{'='*50}")
            
            metrics = train_and_evaluate_model(
                model_type=model_type,
                data_file=args.data_file,
                output_dir=args.output_dir,
                feature_selection=args.feature_selection
            )
            
            results[model_type] = metrics
        except Exception as e:
            print(f"训练{model_type}模型时出错: {e}")
    
    # 比较模型性能
    print("\n模型性能比较:")
    print(f"{'模型类型':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'平均百分比误差':<15} {'中位百分比误差':<15}")
    print("-" * 75)
    
    for model_type, metrics in results.items():
        # 从metrics.json文件中读取更详细的指标
        metrics_path = os.path.join(args.output_dir, f"{model_type}_metrics.json")
        with open(metrics_path, 'r') as f:
            detailed_metrics = json.load(f)
        
        mean_pct_error = detailed_metrics.get('mean_percentage_error', 0)
        median_pct_error = detailed_metrics.get('median_percentage_error', 0)
        
        print(f"{model_type:<15} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['r2']:<10.4f} {mean_pct_error:<15.2f}% {median_pct_error:<15.2f}%")
    
    # 保存比较结果
    comparison_path = os.path.join(args.output_dir, "model_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n模型比较结果已保存到: {comparison_path}")

if __name__ == "__main__":
    main() 