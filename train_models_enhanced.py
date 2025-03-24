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
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# 设置编码
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.6及以下版本不支持reconfigure
        pass

# 防止控制台缓冲区溢出
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# 导入模型模块
from models import ModelFactory

def train_and_evaluate_model(model_type, data_file, output_dir='model', test_size=0.2, random_state=42, params=None, feature_selection=False, top_n_features=None):
    """
    训练并评估指定类型的模型，使用优化后的特征
    
    Args:
        model_type: 模型类型，如'xgboost', 'lightgbm', 'catboost', 'randomforest'等
        data_file: 数据文件路径
        output_dir: 输出目录
        test_size: 测试集比例
        random_state: 随机种子
        params: 模型参数字典（可选）
        feature_selection: 是否进行特征选择
        top_n_features: 如果进行特征选择，使用的特征数量（不指定则自动选择）
    
    Returns:
        模型评估指标
    """
    print(f"开始训练和评估 {model_type} 模型（使用优化后的特征）...")
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

    # 特征选择
    selected_features = feature_cols
    if feature_selection:
        print("执行特征选择...")
        # 初始训练以获取特征重要性
        if model_type in ['xgboost', 'lightgbm', 'catboost', 'randomforest']:
            # 对于树模型，我们可以使用自带的特征重要性
            initial_model = ModelFactory.create_model(model_type)
            initial_model.train(X_train_scaled_df, y_train)
            
            # 获取特征重要性
            feature_importances = initial_model.get_feature_importance()
            
            if top_n_features:
                # 选择前N个重要特征
                selected_features = feature_importances.head(top_n_features)['feature'].tolist()
            else:
                # 使用SelectFromModel进行自动特征选择
                selector = SelectFromModel(initial_model.model, prefit=True)
                selected_mask = selector.get_support()
                selected_features = X_train.columns[selected_mask].tolist()
            
            print(f"从 {len(feature_cols)} 个特征中选择了 {len(selected_features)} 个特征")
            
            # 只使用选定的特征
            X_train_scaled_df = X_train_scaled_df[selected_features]
            X_test_scaled_df = X_test_scaled_df[selected_features]
    
    # 训练模型
    print("训练模型...")
    if model_type == 'xgboost':
        model.fit(X_train_scaled_df, y_train)
    else:
        model.train(X_train_scaled_df, y_train)
    
    # 评估模型
    print("评估模型...")
    if model_type == 'xgboost':
        y_pred = model.predict(X_test_scaled_df)
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    else:
        metrics = model.evaluate(X_test_scaled_df, y_test)
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
        "comparison": {
            "all_features": all_features_metrics,
            "selected_features": selected_features_metrics
        },
        "performance": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        },
        "parameters": model_params,
        "selected_features": selected_features
    }
    
    # 保存模型
    model_path = os.path.join(output_dir, f"{model_type}_enhanced_model.joblib")
    model.save(model_path)
    
    # 保存特征重要性
    importance_path = os.path.join(output_dir, f"{model_type}_enhanced_feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    
    # 保存详细评估指标
    metrics_path = os.path.join(output_dir, f"{model_type}_enhanced_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    # 保存数据集信息
    data_info = {
        "total_samples": len(df),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features_count": len(selected_features),
        "all_features_count": len(feature_cols),
        "features": selected_features,
        "data_file": data_file,
        "test_size": test_size,
        "random_state": random_state,
        "feature_selection": feature_selection,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存数据集信息到model目录
    data_info_path = os.path.join(output_dir, f"{model_type}_enhanced_data_info.json")
    with open(data_info_path, 'w') as f:
        json.dump(data_info, f, indent=2)
    
    # 为了兼容性，同时也将metrics保存到frontend/public/data目录
    frontend_metrics_path = os.path.join("frontend", "public", "data", f"{model_type}_enhanced_metrics.json")
    os.makedirs(os.path.dirname(frontend_metrics_path), exist_ok=True)
    with open(frontend_metrics_path, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    # 计算训练时间
    elapsed_time = time.time() - start_time
    
    print(f"\n{model_type}模型训练和评估完成!")
    print(f"- 训练时间: {elapsed_time:.2f}秒")
    print(f"- RMSE: {metrics['rmse']:.4f}")
    print(f"- MAE: {metrics['mae']:.4f}")
    print(f"- R2: {metrics['r2']:.4f}")
    print(f"- 平均百分比误差: {mean_percent_error:.2f}%")
    print(f"- 中位百分比误差: {median_percent_error:.2f}%")
    print(f"- 模型保存路径: {model_path}")
    print(f"- 特征重要性保存路径: {importance_path}")
    print(f"- 评估指标保存路径: {metrics_path}")
    print(f"- 数据集信息保存路径: {data_info_path}")
    print(f"- 前端指标保存路径: {frontend_metrics_path}")
    
    return metrics

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='训练和评估房价预测模型')
    parser.add_argument('--model_types', nargs='+', default=['xgboost', 'lightgbm', 'catboost', 'randomforest'],
                        help='要训练的模型类型列表')
    parser.add_argument('--data_file', type=str, default='resources/house_samples_engineered_original.csv',
                        help='包含训练数据的CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='model_original',
                        help='保存模型和结果的目录')
    parser.add_argument('--feature_selection', action='store_true',
                        help='是否使用特征选择')
    parser.add_argument('--optimize_hyperparams', action='store_true',
                        help='是否优化超参数')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集大小比例')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取数据
    print(f"读取数据: {args.data_file}")
    try:
        df = pd.read_csv(args.data_file)
    except Exception as e:
        print(f"读取数据时出错: {str(e)}")
        return
    
    # 检查是否存在目标变量
    if 'y_label' not in df.columns:
        print("错误: 数据中没有找到目标变量'y_label'")
        return
    
    # 排除不应作为特征的列
    exclude_cols = ['prop_id', 'std_address', 'y_label', 'full_address_y', 'full_address_x', 'house_name']
    exclude_cols = [col for col in exclude_cols if col in df.columns]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 检查是否存在非数值特征
    object_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"警告: 发现非数值特征列: {object_cols}")
        print("这些列将被排除在训练之外")
        feature_cols = [col for col in feature_cols if col not in object_cols]
    
    # 数据准备
    print("准备数据...")
    X = df[feature_cols].copy()
    y = df['y_label'].copy()
    
    # 确保所有特征都是数值型
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"错误: 列 {col} 是对象类型，无法用于模型训练")
            X = X.drop(columns=[col])
    
    # 检查是否有足够的特征用于训练
    if X.shape[1] == 0:
        print("错误: 没有可用于训练的数值特征")
        return
    
    # 处理缺失值
    X = X.fillna(X.median())
    
    # 特征选择
    if args.feature_selection:
        print("执行特征选择...")
        try:
            # 使用XGBoost的特征重要性进行特征选择
            model = xgb.XGBRegressor(
                n_estimators=200, 
                learning_rate=0.1, 
                max_depth=5, 
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            
            model.fit(X, y)
            
            # 获取特征重要性
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # 选择顶部特征
            top_n = min(50, len(feature_cols))
            top_features = feature_importance['feature'].head(top_n).tolist()
            
            print(f"选择了 {len(top_features)} 个最重要特征")
            X = X[top_features]
        except Exception as e:
            print(f"特征选择过程发生错误: {str(e)}")
            print("将使用所有特征继续")
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 转回DataFrame保留特征名
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # 模型训练和评估
    results = {}
    for model_type in args.model_types:
        print(f"\n==================================================")
        print(f"训练和评估 {model_type} 模型")
        print(f"==================================================")
        
        try:
            # 训练和评估模型
            if model_type == 'xgboost':
                # 略过超参数优化部分，使用默认参数
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    min_child_weight=1,
                    objective='reg:squarederror',
                    random_state=42
                )
                # 训练模型
                print("训练模型...")
                model.fit(X_train_scaled_df, y_train)
                
                # 评估模型
                print("评估模型...")
                y_pred = model.predict(X_test_scaled_df)
                
                # 计算评估指标
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
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
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # 保存详细评估指标
                metrics = {
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2),
                    "mean_percentage_error": float(mean_pct_error),
                    "median_percentage_error": float(median_pct_error),
                    "error_distribution": {
                        "error_ranges": error_ranges
                    },
                    "feature_importance": [
                        {
                            "feature": row["feature"],
                            "importance": float(row["importance"])
                        }
                        for _, row in feature_importance.head(20).iterrows()
                    ]
                }
                
                # 保存模型
                model_path = os.path.join(args.output_dir, f"{model_type}_model.joblib")
                joblib.dump(model, model_path)
                
                # 保存特征重要性
                importance_path = os.path.join(args.output_dir, f"{model_type}_feature_importance.csv")
                feature_importance.to_csv(importance_path, index=False)
                
                # 保存评估指标
                metrics_path = os.path.join(args.output_dir, f"{model_type}_metrics.json")
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                
                # 输出结果
                print(f"\n{model_type}模型训练和评估完成!")
                print(f"- RMSE: {rmse:.4f}")
                print(f"- MAE: {mae:.4f}")
                print(f"- R2: {r2:.4f}")
                print(f"- 平均百分比误差: {mean_pct_error:.2f}%")
                print(f"- 中位百分比误差: {median_pct_error:.2f}%")
                print(f"- 模型保存路径: {model_path}")
                print(f"- 特征重要性保存路径: {importance_path}")
                print(f"- 评估指标保存路径: {metrics_path}")
                
                results[model_type] = {
                    'rmse': float(rmse), 
                    'mae': float(mae), 
                    'r2': float(r2)
                }
            else:
                print(f"跳过 {model_type} 模型训练，本次只训练XGBoost模型")
        except Exception as e:
            print(f"训练{model_type}模型时出错: {str(e)}")
    
    # 比较模型性能
    print("\n模型性能比较:")
    print(f"{'模型类型':<15} {'RMSE':<10} {'MAE':<10} {'R2':<10} {'平均百分比误差':<15} {'中位百分比误差':<15}")
    print("-" * 75)
    
    for model_type, metrics_values in results.items():
        # 从metrics.json文件中读取更详细的指标
        metrics_path = os.path.join(args.output_dir, f"{model_type}_metrics.json")
        with open(metrics_path, 'r', encoding='utf-8') as f:
            detailed_metrics = json.load(f)
        
        mean_pct_error = detailed_metrics.get('mean_percentage_error', 0)
        median_pct_error = detailed_metrics.get('median_percentage_error', 0)
        
        print(f"{model_type:<15} {metrics_values['rmse']:<10.4f} {metrics_values['mae']:<10.4f} {metrics_values['r2']:<10.4f} {mean_pct_error:<15.2f}% {median_pct_error:<15.2f}%")
    
    # 保存比较结果
    comparison_path = os.path.join(args.output_dir, "model_comparison.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n模型比较结果已保存到: {comparison_path}")

if __name__ == "__main__":
    main() 