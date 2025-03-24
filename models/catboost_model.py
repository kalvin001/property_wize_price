import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from catboost import CatBoostRegressor
import joblib
import os
from models.model_interface import ModelInterface

class CatBoostModel(ModelInterface):
    """
    CatBoost模型实现
    """
    
    def __init__(self, **params):
        """
        初始化CatBoost模型
        
        Args:
            **params: 模型参数
        """
        self.model = None
        self._model_path = None
        self._feature_names = None
        self._metadata = {
            "model_type": "catboost",
            "params": params
        }
        
        # 默认参数
        self.default_params = {
            'loss_function': 'RMSE',
            'iterations': 10000,
            'learning_rate': 0.05,
            'depth': 6,
            'random_seed': 42,
            'l2_leaf_reg': 3,
            'verbose': False
        }
        
        # 应用用户提供的参数，覆盖默认参数
        self.params = self.default_params.copy()
        self.params.update(params)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        训练模型
        
        Args:
            X: 特征DataFrame
            y: 目标变量Series
        """
        # 保存特征名称
        self._feature_names = X.columns.tolist()
        
        # 创建模型
        self.model = CatBoostRegressor(**self.params)
        
        # 训练模型
        self.model.fit(X, y, verbose=False)
        
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
    def load(cls, path: str) -> 'CatBoostModel':
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
            instance._metadata = {"model_type": "catboost"}
        
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
        importances = self.model.get_feature_importance()
        
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