import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import joblib
import json
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from models.base_model import BaseModel

class LinearModel(BaseModel):
    """
    线性模型实现类，支持LinearRegression, Ridge, Lasso和ElasticNet
    """
    
    # 支持的线性模型类型
    MODEL_TYPES = {
        "linear": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elasticnet": ElasticNet
    }
    
    def __init__(self, name: str = "linear_model", model_type: str = "linear", **kwargs):
        """
        初始化线性模型
        
        Args:
            name: 模型名称
            model_type: 线性模型类型，支持'linear', 'ridge', 'lasso', 'elasticnet'
            **kwargs: 模型参数
        """
        super().__init__(name=name, model_type=model_type, **kwargs)
        
        # 验证模型类型
        model_type = model_type.lower()
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"不支持的模型类型: {model_type}，支持的类型: {list(self.MODEL_TYPES.keys())}")
        
        self._linear_model_type = model_type
        
        # 设置默认参数
        default_params = {}
        if model_type == "ridge":
            default_params = {"alpha": 1.0}
        elif model_type == "lasso":
            default_params = {"alpha": 1.0}
        elif model_type == "elasticnet":
            default_params = {"alpha": 1.0, "l1_ratio": 0.5}
        
        # 更新参数
        self.model_params = default_params.copy()
        self.model_params.update(kwargs)
        
        # 初始化模型
        self.model = self.MODEL_TYPES[model_type](**self.model_params)
        
        # 更新元数据
        self._metadata["model_type"] = f"Linear-{model_type}"
        self._metadata["params"] = self.model_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        训练线性模型
        
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
        
        # 对于线性模型，保存系数作为特征重要性
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_
            # 如果系数是一维数组，直接使用
            if len(coefficients.shape) == 1:
                self._metadata["coefficients"] = {
                    feature: float(coef) for feature, coef in zip(self.feature_names, coefficients)
                }
            # 如果是二维数组，取第一行
            elif len(coefficients.shape) == 2:
                self._metadata["coefficients"] = {
                    feature: float(coef) for feature, coef in zip(self.feature_names, coefficients[0])
                }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用线性模型进行预测
        
        Args:
            X: 预测数据特征
            
        Returns:
            预测结果数组
        """
        # 先调用父类的predict方法进行检查
        super().predict(X)
        
        # 确保数据只包含模型训练时使用的特征
        if self.feature_names is not None:
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                # 如果有缺失的特征，填充为0
                for feature in missing_features:
                    X[feature] = 0
            
            # 只使用模型训练时的特征，并保持相同的顺序
            X = X[self.feature_names]
        
        # 模型预测
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性（对于线性模型，使用系数作为特征重要性）
        
        Returns:
            特征重要性数据框，包含feature和importance两列
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法获取特征重要性")
        
        # 获取模型系数
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_
        else:
            raise ValueError("模型没有系数属性，无法获取特征重要性")
        
        # 如果没有特征名称，使用默认名称
        if self._feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        else:
            feature_names = self._feature_names
        
        # 确保系数是一维数组
        if len(coefficients.shape) > 1:
            coefficients = coefficients[0]
        
        # 创建特征重要性DataFrame
        importance_list = []
        for feature, coef in zip(feature_names, coefficients):
            importance_list.append({
                "feature": feature,
                "importance": abs(coef)  # 使用系数的绝对值作为重要性指标
            })
        
        # 将列表转换为DataFrame并按重要性降序排序
        importance_df = pd.DataFrame(importance_list)
        importance_df = importance_df.sort_values("importance", ascending=False)
        
        return importance_df
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取线性模型参数
        
        Returns:
            模型参数字典
        """
        if self.model is None:
            return self.model_params
        return self.model.get_params()
    
    def set_params(self, **params) -> None:
        """
        设置线性模型参数
        
        Args:
            **params: 模型参数
        """
        if self.model is None:
            self.model_params.update(params)
            model_class = self.MODEL_TYPES[self._linear_model_type]
            self.model = model_class(**self.model_params)
        else:
            self.model.set_params(**params)
            self.model_params.update(params)
        
        # 更新元数据
        self._metadata["params"] = self.model_params 