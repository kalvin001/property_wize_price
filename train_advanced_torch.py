import pandas as pd
import numpy as np
import argparse
import os
import torch
from models.model_factory import ModelFactory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import json

def train_advanced_torch_model(data_file, output_dir='model', test_size=0.2, random_state=42, 
                               hidden_layers=(256, 256, 128, 128, 64), dropout_rate=0.3, 
                               learning_rate=0.0005, batch_size=64, num_epochs=300, patience=30):
    """
    训练高级PyTorch深度神经网络模型
    
    Args:
        data_file: 数据文件路径
        output_dir: 输出模型目录
        test_size: 测试集比例
        random_state: 随机种子
        hidden_layers: 隐藏层结构，元组形式
        dropout_rate: Dropout比率
        learning_rate: 学习率
        batch_size: 批次大小
        num_epochs: 训练轮数
        patience: 早停耐心值
    
    Returns:
        模型评估指标
    """
    print(f"{'='*50}")
    print(f"开始训练高级PyTorch深度神经网络模型（带残差连接）")
    print(f"{'='*50}")
    print(f"隐藏层结构: {hidden_layers}")
    print(f"学习率: {learning_rate}")
    print(f"批次大小: {batch_size}")
    print(f"最大训练轮数: {num_epochs}")
    print(f"早停耐心值: {patience}")
    
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
    print("创建高级PyTorch神经网络模型...")
    model_params = {
        'hidden_layers': hidden_layers,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'patience': patience
    }
    torch_model = ModelFactory.create_model('torch_advanced_nn', **model_params)
    
    # 训练模型
    print("训练模型...")
    torch_model.train(X_train_scaled_df, y_train)
    
    # 评估模型
    print("评估模型...")
    metrics = torch_model.evaluate(X_test_scaled_df, y_test)
    
    # 保存模型
    model_path = os.path.join(output_dir, "torch_advanced_nn_model.pt")
    torch_model.save(model_path)
    
    # 获取特征重要性
    feature_importance = torch_model.get_feature_importance()
    
    # 保存特征重要性
    importance_path = os.path.join(output_dir, "torch_advanced_nn_feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    
    # 计算训练时间
    elapsed_time = time.time() - start_time
    
    # 输出训练结果
    print(f"\n高级PyTorch神经网络模型训练完成!")
    print(f"- 训练时间: {elapsed_time:.2f}秒")
    print(f"- RMSE: {metrics['rmse']:.4f}")
    print(f"- MAE: {metrics['mae']:.4f}")
    print(f"- R²: {metrics['r2']:.4f}")
    print(f"- 模型保存路径: {model_path}")
    print(f"- 特征重要性保存路径: {importance_path}")
    
    # 保存详细的评估指标
    metrics_path = os.path.join(output_dir, "torch_advanced_nn_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 前端展示目录
    frontend_metrics_path = os.path.join("frontend", "public", "data", "torch_advanced_nn_metrics.json")
    os.makedirs(os.path.dirname(frontend_metrics_path), exist_ok=True)
    with open(frontend_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"- 评估指标保存路径: {metrics_path}")
    print(f"- 前端指标保存路径: {frontend_metrics_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='训练高级PyTorch深度神经网络模型（带残差连接）')
    parser.add_argument('--data_file', type=str, default='resources/house_samples_features.csv', 
                        help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='model', 
                        help='输出目录')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--hidden_layers', type=str, default='256,256,128,128,64', 
                        help='隐藏层结构，逗号分隔的数字')
    parser.add_argument('--dropout_rate', type=float, default=0.3, 
                        help='Dropout比率')
    parser.add_argument('--learning_rate', type=float, default=0.0005, 
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=300, 
                        help='训练轮数')
    parser.add_argument('--patience', type=int, default=30, 
                        help='早停耐心值')
    parser.add_argument('--model_size', type=str, default='medium', 
                        choices=['small', 'medium', 'large', 'xlarge'], 
                        help='预设模型大小')
    
    args = parser.parse_args()
    
    # 根据预设模型大小设置隐藏层结构
    if args.model_size == 'small':
        hidden_layers = (128, 64, 32)
    elif args.model_size == 'medium':
        hidden_layers = (256, 256, 128, 128, 64)
    elif args.model_size == 'large':
        hidden_layers = (512, 384, 256, 192, 128, 64)
    elif args.model_size == 'xlarge':
        hidden_layers = (1024, 768, 512, 384, 256, 192, 128, 64)
    else:
        # 使用自定义隐藏层结构
        hidden_layers = tuple(map(int, args.hidden_layers.split(',')))
    
    # 训练模型
    metrics = train_advanced_torch_model(
        data_file=args.data_file,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        hidden_layers=hidden_layers,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        patience=args.patience
    )
    
    # 输出最终结果
    print(f"\n最终评估结果:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")

if __name__ == "__main__":
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA可用! 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA不可用，使用CPU进行训练")
    main() 