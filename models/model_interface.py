from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import joblib
import os

class ModelInterface(ABC):
    """
    模型接口抽象类，定义所有模型需要实现的方法
    """
    
    def __init__(self, model_path: str = None, metadata: Dict[str, Any] = None):
        """
        初始化模型接口
        
        Args:
            model_path: 模型保存路径
            metadata: 模型元数据
        """
        self._model_path = model_path
        self._metadata = metadata or {}
        self._feature_names = None
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        获取模型元数据
        
        Returns:
            模型元数据字典
        """
        return self._metadata
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        """
        设置模型元数据
        
        Args:
            value: 模型元数据字典
        """
        self._metadata = value
    
    @property
    def feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            特征名称列表
        """
        return self._feature_names
    
    @feature_names.setter
    def feature_names(self, value: List[str]) -> None:
        """
        设置特征名称列表
        
        Args:
            value: 特征名称列表
        """
        self._feature_names = value
    
    @property
    def model_path(self) -> str:
        """
        获取模型路径
        
        Returns:
            模型路径
        """
        return self._model_path
    
    @model_path.setter
    def model_path(self, value: str) -> None:
        """
        设置模型路径
        
        Args:
            value: 模型路径
        """
        self._model_path = value
    
    @property
    def model_type(self) -> str:
        """
        获取模型类型
        
        Returns:
            模型类型字符串
        """
        return self._metadata.get("model_type", "未知")
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        训练模型
        
        Args:
            X_train: 训练数据特征
            y_train: 训练数据标签
            **kwargs: 其他训练参数
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 预测数据特征
            
        Returns:
            预测结果数组
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_test: 测试数据特征
            y_test: 测试数据标签
            
        Returns:
            包含评估指标的字典，如 RMSE, MAE, R² 等
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            包含特征名称和重要性分数的DataFrame
        """
        pass
    
    @abstractmethod
    def save(self, path: Optional[str] = None) -> str:
        """
        保存模型
        
        Args:
            path: 保存路径，如果为None则使用默认路径
            
        Returns:
            保存的文件路径
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'ModelInterface':
        """
        从指定路径加载模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns:
            模型参数字典
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> None:
        """
        设置模型参数
        
        Args:
            **params: 模型参数
        """
        pass

    @property
    def model_path(self):
        """
        获取模型路径
        
        Returns:
            模型路径
        """
        pass 