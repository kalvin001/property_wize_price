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
    
    def __init__(self, name: str = "base_model", model_path: str = None, **kwargs):
        """
        初始化基础模型
        
        Args:
            name: 模型名称
            model_path: 模型保存路径
            **kwargs: 其他初始化参数
        """
        super().__init__(model_path=model_path)
        self.name = name
        self.model = None
        self._feature_names = None
        self._metadata = {
            "model_type": self.__class__.__name__,
            "params": kwargs
        }
    
    @property
    def model_type(self) -> str:
        """
        获取模型类型
        
        Returns:
            模型类型字符串
        """
        return self._metadata.get("model_type", self.__class__.__name__)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        训练模型 - 在子类中实现
        """
        self._feature_names = X_train.columns.tolist()
        self._metadata["feature_names"] = self._feature_names
    
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
            X_test: 测试集特征
            y_test: 测试集标签
            
        Returns:
            包含各种评估指标的字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 进行预测
        y_pred = self.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 计算MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # 保存评估结果
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape)
        }
        
        # 更新元数据
        self._metadata["metrics"] = metrics
        
        return metrics
    
    def save(self, path: Optional[str] = None) -> str:
        """
        保存模型
        
        Args:
            path: 保存路径，如果为None则使用默认路径
            
        Returns:
            模型保存的文件路径
        """
        if path is None:
            if self._model_path:
                path = self._model_path
            else:
                # 默认保存路径
                path = f"model/{self.name}.joblib"
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 将模型和元数据一起保存
        save_data = {
            "model": self.model,
            "metadata": self._metadata,
            "feature_names": self._feature_names
        }
        
        joblib.dump(save_data, path)
        
        # 同时保存元数据为JSON格式
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(self._metadata, f, indent=2)
        
        # 更新模型路径
        self._model_path = path
        
        return path
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        加载模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        try:
            # 加载模型数据
            load_data = joblib.load(path)
            
            # 创建模型实例
            model_instance = cls()
            
            # 恢复模型属性
            if isinstance(load_data, dict):
                model_instance.model = load_data.get("model")
                model_instance._metadata = load_data.get("metadata", {})
                model_instance._feature_names = load_data.get("feature_names", [])
            else:
                # 兼容旧格式 - 直接加载模型
                model_instance.model = load_data
            
            # 设置模型路径
            model_instance._model_path = path
            
            return model_instance
        except Exception as e:
            raise ValueError(f"加载模型失败: {str(e)}")
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns:
            模型参数字典
        """
        return self._metadata.get("params", {})
    
    def set_params(self, **params) -> None:
        """
        设置模型参数
        
        Args:
            **params: 要设置的参数
        """
        if "params" not in self._metadata:
            self._metadata["params"] = {}
        
        self._metadata["params"].update(params) 