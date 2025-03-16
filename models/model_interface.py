from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class ModelInterface(ABC):
    """
    模型接口抽象类，定义所有模型需要实现的方法
    """
    
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
    def save(self, path: str) -> None:
        """
        保存模型到指定路径
        
        Args:
            path: 保存路径
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