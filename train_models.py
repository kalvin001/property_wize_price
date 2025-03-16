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

# 导入模型模块
from models import ModelFactory

def train_and_evaluate_model(model_type, data_file, output_dir='model', test_size=0.2, random_state=42, params=None):
    """
    训练并评估指定类型的模型
    
    Args:
        model_type: 模型类型，如'xgboost', 'linear', 'ridge', 'lasso', 'elasticnet'
        data_file: 数据文件路径
        output_dir: 输出目录
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
    
    # 训练模型
    print("训练模型...")
    model.train(X_train_scaled_df, y_train)
    
    # 评估模型
    print("评估模型...")
    metrics = model.evaluate(X_test_scaled_df, y_test)
    
    # 获取特征重要性
    feature_importance = model.get_feature_importance()
    
    # 保存模型
    model_path = os.path.join(output_dir, f"{model_type}_model.joblib")
    model.save(model_path)
    
    # 保存特征重要性
    importance_path = os.path.join(output_dir, f"{model_type}_feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    
    # 保存评估指标
    metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 计算训练时间
    elapsed_time = time.time() - start_time
    
    print(f"\n{model_type}模型训练和评估完成!")
    print(f"- 训练时间: {elapsed_time:.2f}秒")
    print(f"- RMSE: {metrics['rmse']:.4f}")
    print(f"- MAE: {metrics['mae']:.4f}")
    print(f"- R²: {metrics['r2']:.4f}")
    print(f"- 模型保存路径: {model_path}")
    print(f"- 特征重要性保存路径: {importance_path}")
    print(f"- 评估指标保存路径: {metrics_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='训练和评估不同类型的模型')
    parser.add_argument('--model_types', nargs='+', default=['xgboost', 'linear', 'ridge', 'lasso', 'elasticnet'],
                       help='要训练的模型类型列表')
    parser.add_argument('--data_file', type=str, default='resources/house_samples_features.csv',
                       help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='model',
                       help='输出目录')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练和评估每种模型
    results = {}
    for model_type in args.model_types:
        try:
            print(f"\n{'='*50}")
            print(f"训练和评估 {model_type} 模型")
            print(f"{'='*50}")
            
            metrics = train_and_evaluate_model(
                model_type=model_type,
                data_file=args.data_file,
                output_dir=args.output_dir
            )
            
            results[model_type] = metrics
        except Exception as e:
            print(f"训练{model_type}模型时出错: {e}")
    
    # 比较模型性能
    print("\n模型性能比较:")
    print(f"{'模型类型':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 45)
    
    for model_type, metrics in results.items():
        print(f"{model_type:<15} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['r2']:<10.4f}")
    
    # 保存比较结果
    comparison_path = os.path.join(args.output_dir, "model_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n模型比较结果已保存到: {comparison_path}")

if __name__ == "__main__":
    main() 