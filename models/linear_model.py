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
        
        self.model_type = model_type
        
        # 设置默认参数
        default_params = {}
        
        # 根据模型类型设置特定的默认参数
        if model_type == "ridge":
            default_params = {
                'alpha': 1.0,
                'random_state': 42
            }
        elif model_type == "lasso":
            default_params = {
                'alpha': 0.1,
                'random_state': 42
            }
        elif model_type == "elasticnet":
            default_params = {
                'alpha': 0.1,
                'l1_ratio': 0.5,
                'random_state': 42
            }
        
        # 更新参数
        self.model_params = default_params.copy()
        self.model_params.update(kwargs)
        
        # 初始化模型
        model_class = self.MODEL_TYPES[model_type]
        self.model = model_class(**self.model_params)
        
        # 更新元数据
        self.metadata["model_type"] = f"Linear-{model_type}"
        self.metadata["params"] = self.model_params
    
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
        self.metadata["training_params"] = kwargs
        
        # 对于线性模型，保存系数作为特征重要性
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_
            # 如果系数是一维数组，直接使用
            if len(coefficients.shape) == 1:
                self.metadata["coefficients"] = {
                    feature: float(coef) for feature, coef in zip(self.feature_names, coefficients)
                }
            # 如果是二维数组，取第一行
            elif len(coefficients.shape) == 2:
                self.metadata["coefficients"] = {
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
        获取线性模型的特征重要性（系数的绝对值）
        
        Returns:
            包含特征名称和重要性分数的DataFrame
        """
        if self.model is None or not hasattr(self.model, 'coef_'):
            raise ValueError("模型尚未训练或不支持特征系数")
            
        # 获取系数
        coefficients = self.model.coef_
        
        # 如果系数是一维数组，直接使用
        if len(coefficients.shape) == 1:
            coefs = coefficients
        # 如果是二维数组，取第一行
        elif len(coefficients.shape) == 2:
            coefs = coefficients[0]
        
        # 如果没有特征名称，使用默认名称
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefs))]
        else:
            feature_names = self.feature_names
            
        # 创建特征重要性DataFrame，使用系数的绝对值表示重要性
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coefs),
            'coefficient': coefs
        })
        
        # 按重要性降序排序
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
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
            model_class = self.MODEL_TYPES[self.model_type]
            self.model = model_class(**self.model_params)
        else:
            self.model.set_params(**params)
            self.model_params.update(params)
        
        # 更新元数据
        self.metadata["params"] = self.model_params 