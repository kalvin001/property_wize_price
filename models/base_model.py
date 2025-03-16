import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import json

from models.model_interface import ModelInterface

class BaseModel(ModelInterface):
    """
    基础模型类，实现模型接口中共有的方法
    """
    
    def __init__(self, name: str = "base_model", **kwargs):
        """
        初始化基础模型
        
        Args:
            name: 模型名称
            **kwargs: 其他初始化参数
        """
        self.name = name
        self.model = None
        self.feature_names = None
        self.metadata = {
            "model_type": self.__class__.__name__,
            "params": kwargs
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        训练模型 - 在子类中实现
        """
        self.feature_names = X_train.columns.tolist()
        self.metadata["feature_names"] = self.feature_names
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用模型进行预测 - 在子类中实现
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_test: 测试数据特征
            y_test: 测试数据标签
            
        Returns:
            包含评估指标的字典，如 RMSE, MAE, R² 等
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        # 更新模型元数据
        self.metadata["metrics"] = metrics
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        保存模型到指定路径
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
            
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型和元数据
        model_data = {
            "model": self.model,
            "metadata": self.metadata,
            "feature_names": self.feature_names
        }
        
        joblib.dump(model_data, path)
        
        # 保存模型元数据为JSON（用于前端展示）
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        从指定路径加载模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
            
        model_data = joblib.load(path)
        
        # 创建模型实例
        instance = cls()
        instance.model = model_data["model"]
        instance.metadata = model_data["metadata"]
        instance.feature_names = model_data["feature_names"]
        
        return instance
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取模型参数 - 在子类中实现
        """
        return {}
    
    def set_params(self, **params) -> None:
        """
        设置模型参数 - 在子类中实现
        """
        pass 