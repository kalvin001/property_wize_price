import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import joblib
import json
from collections import OrderedDict
from sklearn.neighbors import KNeighborsRegressor

from models.base_model import BaseModel

class KNNModel(BaseModel):
    """
    KNN模型实现类
    """
    
    def __init__(self, name: str = "knn_model", **kwargs):
        """
        初始化KNN模型
        
        Args:
            name: 模型名称
            **kwargs: KNN模型参数
        """
        super().__init__(name=name, **kwargs)
        
        # 设置默认参数
        default_params = {
            'n_neighbors': 5,
            'weights': 'distance',  # 改为distance更合理，根据距离加权
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,  # 欧几里得距离
            'n_jobs': -1
        }
        
        # 更新参数
        self.model_params = default_params.copy()
        self.model_params.update(kwargs)
        
        # 初始化模型
        self.model = KNeighborsRegressor(**self.model_params)
        
        # 更新元数据
        self._metadata["model_type"] = "KNN"
        self._metadata["params"] = self.model_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        训练KNN模型
        
        Args:
            X_train: 训练数据特征
            y_train: 训练数据标签
            **kwargs: 其他训练参数
        """
        # 调用父类的train方法设置feature_names
        super().train(X_train, y_train, **kwargs)
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 更新元数据
        self._metadata["training_params"] = kwargs
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用KNN模型进行预测
        
        Args:
            X: 预测数据特征
            
        Returns:
            预测结果数组
        """
        # 先调用父类的predict方法进行检查
        super().predict(X)
        
        # 确保数据只包含模型训练时使用的特征
        if self._feature_names is not None:
            missing_features = [f for f in self._feature_names if f not in X.columns]
            if missing_features:
                # 如果有缺失的特征，填充为0
                for feature in missing_features:
                    X[feature] = 0
            
            # 只使用模型训练时的特征，并保持相同的顺序
            X = X[self._feature_names]
        
        # 模型预测
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        注意: KNN模型没有内在的特征重要性测量方法。此实现返回的是一个均匀分布的特征重要性（所有特征重要性相同）。
        对于KNN模型，可以考虑使用置换重要性或其他基于距离的方法来估计特征重要性。
        
        Returns:
            特征重要性数据框，包含feature和importance两列
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法获取特征重要性")
        
        # 如果没有特征名称，使用默认名称
        if self._feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.model.n_features_in_)]
        else:
            feature_names = self._feature_names
        
        # 创建特征重要性列表 - 对于KNN模型，所有特征重要性相同
        importance_list = []
        importance_value = 1.0 / len(feature_names)  # 均匀分配重要性
        
        for feature in feature_names:
            importance_list.append({
                "feature": feature,
                "importance": importance_value
            })
        
        # 将列表转换为DataFrame
        importance_df = pd.DataFrame(importance_list)
        
        return importance_df
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取KNN模型参数
        
        Returns:
            模型参数字典
        """
        if self.model is None:
            return self.model_params
        return self.model.get_params()
    
    def set_params(self, **params) -> None:
        """
        设置KNN模型参数
        
        Args:
            **params: 模型参数
        """
        if self.model is None:
            self.model_params.update(params)
            self.model = KNeighborsRegressor(**self.model_params)
        else:
            self.model.set_params(**params)
            self.model_params.update(params)
        
        # 更新元数据
        self._metadata["params"] = self.model_params 