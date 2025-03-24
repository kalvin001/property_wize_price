import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import lightgbm as lgb
import joblib
import os
from models.model_interface import ModelInterface
import itertools
from sklearn.model_selection import KFold, cross_val_score

class LightGBMModel(ModelInterface):
    """
    LightGBM模型实现
    """
    
    def __init__(self, **params):
        """
        初始化LightGBM模型
        
        Args:
            **params: 模型参数
        """
        self.model = None
        self._model_path = None
        self._feature_names = None
        self._metadata = {
            "model_type": "lightgbm",
            "params": params
        }
        
        # 默认参数
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 3.0,
            'lambda_l2': 3.0,
            'verbose': -1
        }
        
        # 应用用户提供的参数，覆盖默认参数
        self.params = self.default_params.copy()
        self.params.update(params)
    
    def train(self, X: pd.DataFrame, y: pd.Series, enable_param_search: bool = False, 
             param_grid: Optional[Dict[str, List[Any]]] = None, cv: int = 5, 
             search_metric: str = 'rmse', max_combinations: int = 20) -> None:
        if enable_param_search and param_grid is None:
            param_grid = {
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.8, 0.9]
            }
        
        """
        训练模型
        
        Args:
            X: 特征DataFrame
            y: 目标变量Series
            enable_param_search: 是否启用参数搜索，默认为False
            param_grid: 参数网格，格式为 {参数名: [可能的值列表]}
            cv: 交叉验证折数
            search_metric: 参数搜索评估指标，可选'rmse'、'mae'、'r2'
            max_combinations: 最大参数组合数
            
        示例:
            ```python
            # 创建模型实例
            model = LightGBMModel()
            
            # 定义参数网格
            param_grid = {
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.8, 0.9]
            }
            
            # 方式1: 单独训练模型（不搜索参数）
            model.train(X_train, y_train)
            
            # 方式2: 使用参数搜索训练模型
            model.train(X_train, y_train, enable_param_search=True, param_grid=param_grid,
                       cv=5, search_metric='rmse', max_combinations=20)
            ```
        """
        # 保存特征名称
        self._feature_names = X.columns.tolist()
        
        # 如果启用参数搜索且提供了参数网格
        if enable_param_search and param_grid:
            print("启用参数搜索...")
            best_params, best_metrics = self.param_search(X, y, param_grid, 
                                                        cv=cv, metric=search_metric, 
                                                        max_combinations=max_combinations)
            
            # param_search 方法已经训练了模型，所以不需要再次训练
            return
        
        # 常规训练流程（参数搜索未开启或未提供参数网格）
        print("使用当前参数训练模型...")
        # 准备训练数据
        lgb_train = lgb.Dataset(X, y)
        
        # 训练模型
        self.model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=10000, 
            verbose_eval=False
        )
        
        # 更新元数据
        self._metadata.update({
            "feature_names": self._feature_names,
            "n_features": len(self._feature_names),
            "trained": True
        })
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 特征DataFrame
            
        Returns:
            预测结果数组
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 确保输入特征与训练时一致
        if self._feature_names is not None:
            missing_features = [f for f in self._feature_names if f not in X.columns]
            extra_features = [f for f in X.columns if f not in self._feature_names]
            
            if missing_features:
                raise ValueError(f"输入特征缺少以下特征: {missing_features}")
            
            # 只使用训练时的特征，并保持相同顺序
            X = X[self._feature_names]
        
        # 预测
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征DataFrame
            y: 目标变量Series
            
        Returns:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 预测
        y_pred = self.predict(X)
        
        # 计算指标
        mse = ((y - y_pred) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(y - y_pred).mean()
        r2 = 1 - (((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        
        # 更新元数据
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
        
        self._metadata["metrics"] = metrics
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        保存模型到指定路径
        
        Args:
            path: 模型保存路径
        """
        # 准备要保存的数据
        data = {
            "model": self.model,
            "metadata": self._metadata,
            "feature_names": self._feature_names
        }
        
        # 保存模型
        joblib.dump(data, path)
        
        # 更新模型路径
        self._model_path = path
        
        # 保存元数据
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        try:
            import json
            with open(meta_path, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            print(f"保存元数据失败: {e}")
    
    @classmethod
    def load(cls, path: str) -> 'LightGBMModel':
        """
        从指定路径加载模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        # 加载模型数据
        data = joblib.load(path)
        
        # 创建模型实例
        instance = cls()
        
        # 设置模型属性
        if isinstance(data, dict):
            instance.model = data.get("model")
            instance._metadata = data.get("metadata", {})
            instance._feature_names = data.get("feature_names")
        else:
            # 兼容旧版本
            instance.model = data
            instance._metadata = {"model_type": "lightgbm"}
        
        # 设置模型路径
        instance._model_path = path
        
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 获取特征重要性
        importances = self.model.feature_importance(importance_type='gain')
        
        # 创建特征重要性DataFrame
        if self._feature_names is not None:
            feature_importance = pd.DataFrame({
                'feature': self._feature_names,
                'importance': importances
            })
        else:
            feature_importance = pd.DataFrame({
                'feature': [f'f{i}' for i in range(len(importances))],
                'importance': importances
            })
        
        # 按重要性排序
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns:
            模型参数字典
        """
        return self.params.copy()
    
    def set_params(self, **params) -> None:
        """
        设置模型参数
        
        Args:
            **params: 模型参数
        """
        self.params.update(params)
        self._metadata["params"] = self.params.copy()
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        获取模型元数据
        
        Returns:
            模型元数据字典
        """
        return self._metadata.copy()
    
    @property
    def model_path(self) -> Optional[str]:
        """
        获取模型路径
        
        Returns:
            模型路径
        """
        return self._model_path
    
    @model_path.setter
    def model_path(self, path: str) -> None:
        """
        设置模型路径
        
        Args:
            path: 模型路径
        """
        self._model_path = path
    
    @property
    def feature_names(self) -> Optional[List[str]]:
        """
        获取特征名称
        
        Returns:
            特征名称列表
        """
        return self._feature_names.copy() if self._feature_names else None
    
    def param_search(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, List[Any]], 
                   cv: int = 5, metric: str = 'rmse', max_combinations: int = 20) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        简单参数搜索，自动选择最优参数
        
        Args:
            X: 特征DataFrame
            y: 目标变量Series
            param_grid: 参数网格，格式为 {参数名: [可能的值列表]}
            cv: 交叉验证折数
            metric: 评估指标，可选'rmse'、'mae'、'r2'
            max_combinations: 最大参数组合数
            
        Returns:
            best_params: 最优参数字典
            best_metrics: 最优参数下的评估指标
            
        示例:
            ```python
            # 创建模型实例
            model = LightGBMModel()
            
            # 定义参数网格
            param_grid = {
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.8, 0.9],
                'bagging_fraction': [0.8, 0.9]
            }
            
            # 执行参数搜索
            best_params, best_metrics = model.param_search(X_train, y_train, param_grid, 
                                                         cv=5, metric='rmse', max_combinations=20)
            
            # 查看最佳参数
            print("最佳参数:", best_params)
            print("最佳性能:", best_metrics)
            
            # 使用最佳模型进行预测
            predictions = model.predict(X_test)
            ```
        """
        # 生成所有参数组合
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        # 限制组合数量
        if len(combinations) > max_combinations:
            import random
            random.seed(42)  # 设置随机种子，确保结果可重现
            combinations = random.sample(combinations, max_combinations)
        
        print(f"将评估 {len(combinations)} 个参数组合...")
        
        # 初始化结果跟踪
        best_score = float('inf') if metric in ['rmse', 'mae'] else float('-inf')
        best_params = None
        best_metrics = {}
        
        # 设置交叉验证
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # 测试每个参数组合
        for i, combo in enumerate(combinations):
            # 构建当前参数
            current_params = self.default_params.copy()
            for j, key in enumerate(keys):
                current_params[key] = combo[j]
            
            print(f"评估参数组合 {i+1}/{len(combinations)}: {', '.join([f'{k}={v}' for k, v in zip(keys, combo)])}")
            
            # 创建临时模型实例
            temp_model = LightGBMModel(**current_params)
            
            # 进行交叉验证
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 训练模型
                temp_model.train(X_train, y_train)
                
                # 评估模型
                metrics = temp_model.evaluate(X_val, y_val)
                scores.append(metrics[metric])
            
            # 计算平均得分
            avg_score = np.mean(scores)
            print(f"平均 {metric}: {avg_score:.4f}")
            
            # 更新最佳参数（如果更好）
            is_better = (metric in ['rmse', 'mae'] and avg_score < best_score) or \
                        (metric not in ['rmse', 'mae'] and avg_score > best_score)
            
            if is_better:
                best_score = avg_score
                best_params = current_params
                print(f"找到新的最佳参数: {metric} = {best_score:.4f}")
        
        # 使用最佳参数训练最终模型
        if best_params:
            print(f"使用最佳参数训练最终模型...")
            self.params = best_params
            self.train(X, y)
            
            # 获取完整评估指标
            best_metrics = self.evaluate(X, y)
            print(f"最终模型性能: {', '.join([f'{k}={v:.4f}' for k, v in best_metrics.items()])}")
            
            # 更新元数据
            self._metadata["best_params"] = best_params
            self._metadata["param_search"] = {
                "metric": metric,
                "cv": cv,
                "combinations_tested": len(combinations)
            }
        
        return best_params, best_metrics 