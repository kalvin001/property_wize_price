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

# 导入模型模块
from models import ModelFactory

def train_and_evaluate_model(model_type, data_file, output_dir='model', test_size=0.2, random_state=42, params=None, deep_features_file=None, use_deep_features=False):
    """
    训练并评估指定类型的模型
    
    Args:
        model_type: 模型类型，如'xgboost', 'linear', 'ridge', 'lasso', 'elasticnet'
        data_file: 数据文件路径
        output_dir: 输出目录
        test_size: 测试集比例
        random_state: 随机种子
        params: 模型参数字典（可选）
        deep_features_file: 深度特征文件路径（可选）
        use_deep_features: 是否使用深度特征
    
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
    
    # 如果指定了使用深度特征且提供了深度特征文件路径
    if use_deep_features and deep_features_file:
        print(f"加载深度特征: {deep_features_file}")
        try:
            df_deep = pd.read_csv(deep_features_file)
            print(f"深度特征形状: {df_deep.shape}")
            
            # 根据ID列合并数据（将深度特征合并到主数据框中）
            if 'prop_id' in df_deep.columns and 'prop_id' in df.columns:
                print("使用prop_id合并深度特征...")
                df = df.merge(df_deep.drop(columns=['std_address'] if 'std_address' in df_deep.columns else []), 
                              on='prop_id', how='left')
            elif 'std_address' in df_deep.columns and 'std_address' in df.columns:
                print("使用std_address合并深度特征...")
                df = df.merge(df_deep.drop(columns=['prop_id'] if 'prop_id' in df_deep.columns else []), 
                              on='std_address', how='left')
            else:
                print("警告: 无法找到合适的ID列进行数据合并，深度特征将不会被使用")
                
            # 填充合并后可能出现的缺失值
            deep_feature_cols = [col for col in df_deep.columns 
                                 if col not in ['prop_id', 'std_address', 'y_label'] 
                                 and col in df.columns]
            
            if deep_feature_cols:
                print(f"填充深度特征中的缺失值，特征列: {deep_feature_cols}")
                for col in deep_feature_cols:
                    df[col] = df[col].fillna(df[col].median())
                    
            print(f"合并后的数据形状: {df.shape}")
        except Exception as e:
            print(f"加载深度特征时出错: {e}")
    
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
    
    # 记录是否使用了深度特征
    feature_info = {
        "total_features": len(feature_cols),
        "deep_features_used": use_deep_features and deep_features_file is not None,
        "deep_features_count": sum(1 for col in feature_cols if col.startswith('key_adv_bert_')) if use_deep_features and deep_features_file else 0
    }
    
    # 创建与之前格式一致的详细指标
    detailed_metrics = { 
        "features_count": len(feature_cols),
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
            for _, row in feature_importance.head(1000).iterrows()
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
        "feature_info": feature_info
    }
    
    # 创建模型名称，标记是否使用了深度特征
    model_name = f"{model_type}_{'with_deep' if use_deep_features and deep_features_file else 'regular'}"
    
    # 保存模型
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    model.save(model_path)
    
    # 保存特征重要性
    importance_path = os.path.join(output_dir, f"{model_name}_feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    
    # 保存详细评估指标
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    # 保存数据集信息
    data_info = {
        "total_samples": len(df),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features_count": len(feature_cols),
        "features": feature_cols,
        "target": "y_label",
        "data_file": data_file,
        "deep_features_file": deep_features_file if use_deep_features else None,
        "deep_features_used": use_deep_features and deep_features_file is not None,
        "test_size": test_size,
        "random_state": random_state,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存数据集信息到model目录
    data_info_path = os.path.join(output_dir, f"{model_name}_data_info.json")
    with open(data_info_path, 'w') as f:
        json.dump(data_info, f, indent=2)
    
    # 为了兼容性，同时也将metrics保存到frontend/public/data目录
    frontend_metrics_path = os.path.join("frontend", "public", "data", f"{model_name}_metrics.json")
    os.makedirs(os.path.dirname(frontend_metrics_path), exist_ok=True)
    with open(frontend_metrics_path, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    # 计算训练时间
    elapsed_time = time.time() - start_time
    
    print(f"\n{model_name}模型训练和评估完成!")
    print(f"- 训练时间: {elapsed_time:.2f}秒")
    print(f"- RMSE: {metrics['rmse']:.4f}")
    print(f"- MAE: {metrics['mae']:.4f}")
    print(f"- R²: {metrics['r2']:.4f}")
    print(f"- 模型保存路径: {model_path}")
    print(f"- 特征重要性保存路径: {importance_path}")
    print(f"- 评估指标保存路径: {metrics_path}")
    print(f"- 数据集信息保存路径: {data_info_path}")
    print(f"- 前端指标保存路径: {frontend_metrics_path}")
    
    return metrics, y_test, y_pred, X_test_scaled_df

def ensemble_models(model_types, data_file, output_dir='model', test_size=0.2, random_state=42, weights=None, method='weighted_average'):
    """
    融合多个模型的预测结果
    
    Args:
        model_types: 要融合的模型类型列表
        data_file: 数据文件路径
        output_dir: 输出目录
        test_size: 测试集比例
        random_state: 随机种子
        weights: 各模型的权重，如果为None则使用基于性能的智能权重
        method: 融合方法，可选值为'weighted_average'或'stacking'
    
    Returns:
        融合模型的评估指标
    """
    print(f"\n{'='*50}")
    print(f"开始多模型融合: {', '.join(model_types)}, 方法: {method}")
    print(f"{'='*50}")
    
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
    
    # 生成各模型的预测结果
    all_predictions = []
    all_train_predictions = []  # 用于stacking方法的训练集预测
    model_metrics = {}
    trained_models = {}  # 保存训练好的模型对象
    
    for model_type in model_types:
        print(f"\n训练和预测 {model_type} 模型...")
        try:
            # 创建模型
            model = ModelFactory.create_model(model_type)
            
            # 训练模型
            model.train(X_train_scaled_df, y_train)
            
            # 对测试集进行预测
            y_pred = model.predict(X_test_scaled_df)
            
            # 获取训练集上的预测（用于stacking）
            y_train_pred = model.predict(X_train_scaled_df)
            
            # 评估模型
            metrics = model.evaluate(X_test_scaled_df, y_test)
            
            # 保存结果
            all_predictions.append(y_pred)
            all_train_predictions.append(y_train_pred)
            model_metrics[model_type] = metrics
            trained_models[model_type] = model
            
            # 保存模型
            model_path = os.path.join(output_dir, f"{model_type}_model.joblib")
            model.save(model_path)
            print(f"模型已保存到: {model_path}")
            
        except Exception as e:
            print(f"处理{model_type}模型时出错: {e}")
    
    if len(all_predictions) == 0:
        print("没有成功训练的模型，无法进行融合")
        return None
    
    # 根据方法选择融合策略
    if method == 'stacking':
        # ===== 堆叠集成（Stacking）方法 =====
        print("\n使用堆叠集成(Stacking)方法...")
        
        # 1. 准备元特征（将每个基模型的预测作为新特征）
        meta_features_train = np.column_stack(all_train_predictions)
        meta_features_test = np.column_stack(all_predictions)
        
        # 2. 创建和训练元学习器（二级模型）
        print("训练元学习器(XGBoost)...")
        from xgboost import XGBRegressor
        meta_learner = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=1.0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state
        )
        meta_learner.fit(meta_features_train, y_train)
        
        # 3. 使用元学习器进行最终预测
        ensemble_pred = meta_learner.predict(meta_features_test)
        
        # 4. 获取元学习器的特征权重（即各模型权重）
        if hasattr(meta_learner, 'feature_importances_'):
            # 对于基于树的模型（如XGBoost）
            importances = meta_learner.feature_importances_
            weights = importances / np.sum(importances)
        else:
            # 对于线性模型（如Ridge）
            weights = meta_learner.coef_ / np.sum(np.abs(meta_learner.coef_))
        
        # 保存元学习器模型
        meta_model_path = os.path.join(output_dir, "stacking_meta_learner.joblib")
        joblib.dump(meta_learner, meta_model_path)
        print(f"元学习器模型已保存到: {meta_model_path}")
        
        # 添加元学习器信息到模型信息中
        ensemble_method_info = {
            "method": "stacking",
            "meta_learner": "XGBoost",
            "meta_learner_params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": random_state
            },
            "base_models": model_types,
            "meta_feature_weights": weights.tolist(),
            "meta_learner_path": meta_model_path
        }
    else:
        # ===== 加权平均方法 =====
        # 计算权重: 权重方案1 - 使用提供的权重
        if weights is not None:
            # 确保权重列表长度与模型数量一致
            if len(weights) != len(all_predictions):
                print(f"提供的权重数量({len(weights)})与模型数量({len(all_predictions)})不匹配，使用性能加权方式")
                weights = None
        
        # 计算权重: 权重方案2 - 基于R² score的权重
        if weights is None:
            print("使用基于性能的智能权重分配...")
            # 获取每个模型的R²值
            r2_scores = [metrics["r2"] for metrics in model_metrics.values()]
            
            # 计算权重方式一: R²的归一化值
            weights_r2 = [r2 / sum(r2_scores) for r2 in r2_scores]
            
            # 计算权重方式二: 误差倒数加权 (1/RMSE)
            rmse_scores = [metrics["rmse"] for metrics in model_metrics.values()]
            weights_rmse_inverse = [1/rmse for rmse in rmse_scores]
            weights_rmse_inverse = [w / sum(weights_rmse_inverse) for w in weights_rmse_inverse]
            
            # 计算权重方式三: 指数加权 (基于R²)
            # 使R²的小差异在权重中得到放大
            r2_exp = [np.exp(r2 * 2) for r2 in r2_scores]  # 指数因子可调整
            weights_r2_exp = [w / sum(r2_exp) for w in r2_exp]
            
            # 默认使用指数加权
            weights = weights_r2_exp
            
            print("不同权重方案:")
            print(f"- R²归一化权重: {[round(w, 4) for w in weights_r2]}")
            print(f"- RMSE倒数权重: {[round(w, 4) for w in weights_rmse_inverse]}")
            print(f"- R²指数权重(已选): {[round(w, 4) for w in weights_r2_exp]}")
        
        # 标准化权重，确保和为1
        weights = [w / sum(weights) for w in weights]
        
        # 加权融合预测结果
        ensemble_pred = np.zeros_like(all_predictions[0])
        for i, pred in enumerate(all_predictions):
            ensemble_pred += weights[i] * pred
            
        ensemble_method_info = {
            "method": "weighted_average",
            "weights": [float(w) for w in weights]
        }
    
    # 计算融合模型的评估指标
    mse = mean_squared_error(y_test, ensemble_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)
    
    # 计算更多详细的指标
    percent_errors = np.abs((y_test - ensemble_pred) / y_test) * 100
    median_ae = np.median(np.abs(y_test - ensemble_pred))
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
    
    # 创建融合模型的详细指标
    ensemble_metrics = {
        "model_name": ensemble_name,
        "features_count": len(feature_cols),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2),
        "mse": float(mse),
        "median_ae": float(median_ae),
        "explained_variance": float(1.0 - (np.var(y_test - ensemble_pred) / np.var(y_test))),
        "cv_rmse": float(rmse),
        "predictions": ensemble_pred.tolist()[:10],
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
        "ensemble_info": {
            "models": model_types,
            "ensemble_method": ensemble_method_info,
            "individual_model_metrics": {model: {"rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"]} 
                                        for model, metrics in model_metrics.items()}
        },
        
        "performance": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
    }
    
    # 保存融合模型的评估指标
    ensemble_prefix = "stacking" if method == "stacking" else "ensemble"
    ensemble_name = f"{ensemble_prefix}_" + "_".join(model_types)
    if len(ensemble_name) > 100:  # 避免文件名过长
        ensemble_name = f"{ensemble_prefix}_models"
        
    metrics_path = os.path.join(output_dir, f"{ensemble_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(ensemble_metrics, f, indent=2)
    
    # 为了兼容性，同时也将metrics保存到frontend/public/data目录
    frontend_metrics_path = os.path.join("frontend", "public", "data", f"{ensemble_name}_metrics.json")
    os.makedirs(os.path.dirname(frontend_metrics_path), exist_ok=True)
    with open(frontend_metrics_path, 'w') as f:
        json.dump(ensemble_metrics, f, indent=2)
    
    # 计算训练时间
    elapsed_time = time.time() - start_time
    
    # 显示融合模型的评估结果
    print(f"\n多模型融合完成! 方法: {method}")
    if method == 'stacking':
        print(f"- 元学习器: XGBoost")
        print(f"- 模型贡献权重: {[round(w, 4) for w in weights]}")
    else:
        print(f"- 融合权重: {[round(w, 4) for w in weights]}")
    print(f"- 融合模型: {', '.join(model_types)}")
    print(f"- 训练时间: {elapsed_time:.2f}秒")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- MAE: {mae:.4f}")
    print(f"- R²: {r2:.4f}")
    print(f"- 评估指标保存路径: {metrics_path}")
    print(f"- 前端指标保存路径: {frontend_metrics_path}")
    
    # 与单个模型进行比较
    print("\n与单个模型比较:")
    print(f"{'模型':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 50)
    
    for model_type, metrics in model_metrics.items():
        print(f"{model_type:<20} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['r2']:<10.4f}")
    
    print(f"{'融合模型 (' + method + ')':<20} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f}")
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def train_with_cross_validation(model_type, data_file, output_dir='model', n_splits=5, random_state=42, params=None, deep_features_file=None, use_deep_features=False):
    """
    使用K折交叉验证训练和评估模型
    
    Args:
        model_type: 模型类型
        data_file: 数据文件路径
        output_dir: 输出目录
        n_splits: 交叉验证折数
        random_state: 随机种子
        params: 模型参数字典（可选）
        deep_features_file: 深度特征文件路径（可选）
        use_deep_features: 是否使用深度特征
        
    Returns:
        交叉验证平均指标
    """
    print(f"\n{'='*50}")
    print(f"开始{n_splits}折交叉验证: {model_type} 模型")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print(f"读取数据: {data_file}")
    df = pd.read_csv(data_file)
    
    # 如果指定了使用深度特征且提供了深度特征文件路径
    if use_deep_features and deep_features_file:
        print(f"加载深度特征: {deep_features_file}")
        try:
            df_deep = pd.read_csv(deep_features_file)
            print(f"深度特征形状: {df_deep.shape}")
            
            # 根据ID列合并数据（将深度特征合并到主数据框中）
            if 'prop_id' in df_deep.columns and 'prop_id' in df.columns:
                print("使用prop_id合并深度特征...")
                df = df.merge(df_deep.drop(columns=['std_address'] if 'std_address' in df_deep.columns else []), 
                              on='prop_id', how='left')
            elif 'std_address' in df_deep.columns and 'std_address' in df.columns:
                print("使用std_address合并深度特征...")
                df = df.merge(df_deep.drop(columns=['prop_id'] if 'prop_id' in df_deep.columns else []), 
                              on='std_address', how='left')
            else:
                print("警告: 无法找到合适的ID列进行数据合并，深度特征将不会被使用")
                
            # 填充合并后可能出现的缺失值
            deep_feature_cols = [col for col in df_deep.columns 
                                 if col not in ['prop_id', 'std_address', 'y_label'] 
                                 and col in df.columns]
            
            if deep_feature_cols:
                print(f"填充深度特征中的缺失值，特征列: {deep_feature_cols}")
                for col in deep_feature_cols:
                    df[col] = df[col].fillna(df[col].median())
                    
            print(f"合并后的数据形状: {df.shape}")
        except Exception as e:
            print(f"加载深度特征时出错: {e}")
    
    # 排除不应作为特征的列
    exclude_cols = ['prop_id', 'std_address', 'y_label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 数据准备
    print("准备数据...")
    X = df[feature_cols].copy()
    y = df['y_label'].copy()
    
    # 处理缺失值
    X = X.fillna(X.median())
    
    # 导入交叉验证工具
    from sklearn.model_selection import KFold
    
    # 创建K折交叉验证分割器
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 记录每个折的评估指标
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\n{'--'*25}")
        print(f"训练和评估第{fold}/{n_splits}折...")
        print(f"{'--'*25}")
        
        # 分割训练集和测试集
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 将缩放后的数据转换回DataFrame以保留特征名称
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # 创建模型
        print(f"创建{model_type}模型（第{fold}折）...")
        if params:
            model = ModelFactory.create_model(model_type, **params)
            print(f"使用自定义参数: {params}")
        else:
            model = ModelFactory.create_model(model_type)
            print("使用默认参数")
        
        # 训练模型
        print(f"训练模型（第{fold}折）...")
        model.train(X_train_scaled_df, y_train, validation_split=0.1)
        
        # 评估模型
        print(f"评估模型（第{fold}折）...")
        metrics = model.evaluate(X_test_scaled_df, y_test)
        
        # 添加折数信息
        metrics['fold'] = fold
        fold_metrics.append(metrics)
        
        # 保存本折的模型（可选）
        if n_splits <= 3:  # 如果折数较少，则保存每个折的模型
            model_path = os.path.join(output_dir, f"{model_type}_fold{fold}_model.joblib")
            model.save(model_path)
            print(f"第{fold}折模型已保存到: {model_path}")
    
    # 计算平均指标
    avg_metrics = {
        'rmse': np.mean([m['rmse'] for m in fold_metrics]),
        'mae': np.mean([m['mae'] for m in fold_metrics]),
        'r2': np.mean([m['r2'] for m in fold_metrics])
    }
    
    # 计算标准差
    std_metrics = {
        'rmse_std': np.std([m['rmse'] for m in fold_metrics]),
        'mae_std': np.std([m['mae'] for m in fold_metrics]),
        'r2_std': np.std([m['r2'] for m in fold_metrics])
    }
    
    # 合并平均指标和标准差
    cv_metrics = {**avg_metrics, **std_metrics}
    
    # 将每折的详细指标添加到结果中
    cv_metrics['fold_metrics'] = fold_metrics
    
    # 保存交叉验证结果
    cv_metrics_path = os.path.join(output_dir, f"{model_type}_cv_metrics.json")
    with open(cv_metrics_path, 'w') as f:
        json.dump(cv_metrics, f, indent=2)
    
    # 训练最终模型（使用全部数据）
    print("\n训练最终模型（使用全部数据）...")
    
    # 标准化所有数据
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # 创建并训练最终模型
    final_model = ModelFactory.create_model(model_type, **params if params else {})
    final_model.train(X_scaled_df, y, validation_split=0.1)
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, f"{model_type}_final_model.joblib")
    final_model.save(final_model_path)
    
    # 计算训练时间
    elapsed_time = time.time() - start_time
    
    # 打印交叉验证结果
    print(f"\n{n_splits}折交叉验证结果:")
    print(f"- 平均RMSE: {avg_metrics['rmse']:.4f} ± {std_metrics['rmse_std']:.4f}")
    print(f"- 平均MAE: {avg_metrics['mae']:.4f} ± {std_metrics['mae_std']:.4f}")
    print(f"- 平均R²: {avg_metrics['r2']:.4f} ± {std_metrics['r2_std']:.4f}")
    print(f"- 总训练时间: {elapsed_time:.2f}秒")
    print(f"- 交叉验证指标已保存到: {cv_metrics_path}")
    print(f"- 最终模型已保存到: {final_model_path}")
    
    return cv_metrics

def main():
    parser = argparse.ArgumentParser(description='训练和评估不同类型的模型')
    # parser.add_argument('--model_types', nargs='+', default=['xgboost', 'linear', 'ridge',  'lasso','lightgbm', 
    #                                                          'catboost', 'torch_nn',  ],
    #                                                          #'property_similarity_knn','geographic_knn','weighted_knn','knn' 'elasticnet',
    #                     help='要训练的模型类型列表')

    parser.add_argument('--model_types', nargs='+', default=['xgboost', 'lightgbm', 'catboost'],   help='要训练的模型类型列表') #, 'torch_nn'
    
    #parser.add_argument('--model_types', nargs='+', default=['torch_nn'],   help='要训练的模型类型列表')
    #parser.add_argument('--model_types', nargs='+', default=['lightgbm'],   help='要训练的模型类型列表')
    #parser.add_argument('--data_file', type=str, default='resources/house_samples_features.csv',  help='数据文件路径')
    #parser.add_argument('--data_file', type=str, default='resources/house_samples_engineered_enhanced.csv',  help='数据文件路径')
    parser.add_argument('--data_file', type=str, default='resources/house_samples_features_v2.csv',  help='常规特征数据文件路径')
    #parser.add_argument('--data_file', type=str, default='resources/house_samples_engineered.csv',  help='数据文件路径')
    #parser.add_argument('--data_file', type=str, default='resources/house_samples_deep_features.csv',  help='常规特征数据文件路径')

    #parser.add_argument('--data_file', type=str, default='resources/house_samples_engineered.csv',  help='数据文件路径')

    
    
    parser.add_argument('--output_dir', type=str, default='model', help='输出目录')
    parser.add_argument('--ensemble', action='store_false', help='是否进行模型融合')
    parser.add_argument('--ensemble_weights', nargs='+', type=float, help='模型融合权重，需与模型数量一致')
    parser.add_argument('--ensemble_method', type=str, choices=['weighted_average', 'stacking'], 
                        default='weighted_average', help='融合方法: weighted_average或stacking')
    
    # 添加深度特征相关的参数
    parser.add_argument('--deep_features_file', type=str, default='resources/house_samples_deep_features.csv', 
                        help='深度特征数据文件路径')
    parser.add_argument('--use_deep_features', action='store_true', help='是否使用深度特征')
    parser.add_argument('--deep_only', action='store_true', help='是否仅使用深度特征进行训练')
    parser.add_argument('--compare_with_without_deep', action='store_true', 
                        help='对比使用深度特征和不使用深度特征的模型性能')
                        
    # 添加交叉验证相关参数
    parser.add_argument('--cv', action='store_true', help='是否使用交叉验证')
    parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果启用了交叉验证，使用交叉验证方式训练模型
    if args.cv:
        cv_results = {}
        
        for model_type in args.model_types:
            try:
                cv_metrics = train_with_cross_validation(
                    model_type=model_type,
                    data_file=args.data_file,
                    output_dir=args.output_dir,
                    n_splits=args.n_splits,
                    random_state=args.random_state,
                    deep_features_file=args.deep_features_file if args.use_deep_features else None,
                    use_deep_features=args.use_deep_features
                )
                
                cv_results[model_type] = cv_metrics
                
            except Exception as e:
                print(f"交叉验证{model_type}模型时出错: {e}")
        
        # 比较不同模型的交叉验证结果
        if len(cv_results) > 1:
            print("\n不同模型的交叉验证结果比较:")
            print(f"{'模型类型':<15} {'平均RMSE':<15} {'平均MAE':<15} {'平均R²':<15}")
            print("-" * 60)
            
            for model_type, metrics in cv_results.items():
                print(f"{model_type:<15} {metrics['rmse']:.4f} ± {metrics['rmse_std']:.4f} {metrics['mae']:.4f} ± {metrics['mae_std']:.4f} {metrics['r2']:.4f} ± {metrics['r2_std']:.4f}")
        
        return  # 交叉验证完成后退出
    
    # 检查是否需要同时训练使用和不使用深度特征的模型进行比较
    if args.compare_with_without_deep:
        print("\n开始对比实验: 使用深度特征 vs 不使用深度特征")
        results = {}
        
        # 对每种模型类型，分别训练使用和不使用深度特征的版本
        for model_type in args.model_types:
            try:
                print(f"\n{'='*50}")
                print(f"训练和评估 {model_type} 模型 (不使用深度特征)")
                print(f"{'='*50}")
                
                # 训练不使用深度特征的模型
                metrics_without_deep, _, _, _ = train_and_evaluate_model(
                    model_type=model_type,
                    data_file=args.data_file,
                    output_dir=args.output_dir,
                    use_deep_features=False
                )
                
                print(f"\n{'='*50}")
                print(f"训练和评估 {model_type} 模型 (使用深度特征)")
                print(f"{'='*50}")
                
                # 训练使用深度特征的模型
                metrics_with_deep, _, _, _ = train_and_evaluate_model(
                    model_type=model_type,
                    data_file=args.data_file,
                    output_dir=args.output_dir,
                    use_deep_features=True,
                    deep_features_file=args.deep_features_file
                )
                
                # 保存结果
                results[f"{model_type}_without_deep"] = metrics_without_deep
                results[f"{model_type}_with_deep"] = metrics_with_deep
                
            except Exception as e:
                print(f"训练{model_type}模型时出错: {e}")
        
        # 比较结果
        print("\n深度特征对比实验结果:")
        print(f"{'模型类型':<20} {'RMSE(无深度)':<15} {'RMSE(有深度)':<15} {'改进率':<10} {'R²(无深度)':<12} {'R²(有深度)':<12}")
        print("-" * 90)
        
        for model_type in args.model_types:
            without_key = f"{model_type}_without_deep"
            with_key = f"{model_type}_with_deep"
            
            if without_key in results and with_key in results:
                rmse_without = results[without_key]['rmse']
                rmse_with = results[with_key]['rmse']
                rmse_improvement = (rmse_without - rmse_with) / rmse_without * 100
                
                r2_without = results[without_key]['r2']
                r2_with = results[with_key]['r2']
                
                print(f"{model_type:<20} {rmse_without:<15.4f} {rmse_with:<15.4f} {rmse_improvement:+.2f}% {r2_without:<12.4f} {r2_with:<12.4f}")
        
        # 保存比较结果
        comparison_deep_path = os.path.join(args.output_dir, "deep_feature_comparison.json")
        with open(comparison_deep_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n深度特征对比结果已保存到: {comparison_deep_path}")
        
    else:
        # 正常训练模式
        # 训练和评估每种模型
        results = {}
        for model_type in args.model_types:
            try:
                print(f"\n{'='*50}")
                if args.deep_only:
                    print(f"仅使用深度特征训练和评估 {model_type} 模型")
                    
                    # 读取仅包含深度特征的数据
                    deep_df = pd.read_csv(args.deep_features_file)
                    # 需要将deep_df添加y_label列
                    if 'y_label' not in deep_df.columns:
                        print("向深度特征数据添加y_label列...")
                        regular_df = pd.read_csv(args.data_file)
                        id_col = 'std_address' if 'std_address' in deep_df.columns else 'prop_id'
                        deep_df = deep_df.merge(regular_df[['y_label', id_col]], on=id_col, how='left')
                        
                        # 保存合并后的深度特征文件
                        deep_only_file = os.path.join(os.path.dirname(args.deep_features_file), 
                                                    'house_samples_deep_only.csv')
                        deep_df.to_csv(deep_only_file, index=False)
                        print(f"已保存仅深度特征的数据文件: {deep_only_file}")
                        
                    # 仅使用深度特征训练
                    metrics, _, _, _ = train_and_evaluate_model(
                        model_type=model_type,
                        data_file=deep_only_file,  # 使用包含y_label的深度特征文件
                        output_dir=args.output_dir,
                        use_deep_features=False  # 不需要再合并特征
                    )
                else:
                    print(f"训练和评估 {model_type} 模型")
                    print(f"{'='*50}")
                    
                    metrics, _, _, _ = train_and_evaluate_model(
                        model_type=model_type,
                        data_file=args.data_file,
                        output_dir=args.output_dir,
                        use_deep_features=args.use_deep_features,
                        deep_features_file=args.deep_features_file if args.use_deep_features else None
                    )
                
                results[model_type] = metrics
            except Exception as e:
                print(f"训练{model_type}模型时出错: {e}")
        
        # 比较模型性能
        print("\n模型性能比较:")
        print(f"{'模型类型':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'平均百分比误差':<15} {'中位百分比误差':<15}")
        print("-" * 75)
        
        for model_type, metrics in results.items():
            # 构建模型名称
            model_name = f"{model_type}_{'with_deep' if args.use_deep_features else 'regular'}"
            if args.deep_only:
                model_name = f"{model_type}_deep_only"
                
            # 从metrics.json文件中读取更详细的指标
            metrics_path = os.path.join(args.output_dir, f"{model_name}_metrics.json")
            
            try:
                with open(metrics_path, 'r') as f:
                    detailed_metrics = json.load(f)
                
                mean_pct_error = detailed_metrics.get('mean_percentage_error', 0)
                median_pct_error = detailed_metrics.get('median_percentage_error', 0)
                
                print(f"{model_type:<15} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['r2']:<10.4f} {mean_pct_error:<15.2f}% {median_pct_error:<15.2f}%")
            except Exception as e:
                print(f"读取{model_name}指标时出错: {e}")
                print(f"{model_type:<15} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['r2']:<10.4f} {'N/A':<15} {'N/A':<15}")
        
        # 保存比较结果
        comparison_path = os.path.join(args.output_dir, "model_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n模型比较结果已保存到: {comparison_path}")
        
    # # 如果启用了模型融合，执行模型融合
    # if args.ensemble or True:  # 默认执行模型融合
    #     print("\n执行模型融合...")
    #     
    #     # 首先执行加权平均融合
    #     weighted_ensemble_metrics = ensemble_models(
    #         model_types=args.model_types,
    #         data_file=args.data_file,
    #         output_dir=args.output_dir,
    #         weights=args.ensemble_weights,
    #         method='weighted_average'
    #     )
    #     
    #     if weighted_ensemble_metrics:
    #         results['weighted_ensemble'] = weighted_ensemble_metrics
    #     
    #     # 然后执行堆叠融合
    #     stacking_ensemble_metrics = ensemble_models(
    #         model_types=args.model_types,
    #         data_file=args.data_file,
    #         output_dir=args.output_dir,
    #         method='stacking'
    #     )
    #     
    #     if stacking_ensemble_metrics:
    #         results['stacking_ensemble'] = stacking_ensemble_metrics
    #         
    #     # 更新比较结果文件
    #     with open(comparison_path, 'w') as f:
    #         json.dump(results, f, indent=2)
    #     
    #     print(f"\n更新后的模型比较结果已保存到: {comparison_path}")
    # 
    # # 显示最佳模型
    # best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    # print(f"\n最佳模型: {best_model[0]}")
    # print(f"- RMSE: {best_model[1]['rmse']:.4f}")
    # print(f"- MAE: {best_model[1]['mae']:.4f}")
    # print(f"- R²: {best_model[1]['r2']:.4f}")

if __name__ == "__main__":
    main() 