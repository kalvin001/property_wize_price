import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import joblib
import json
from collections import OrderedDict
from xgboost import XGBRegressor

from models.base_model import BaseModel

class XGBoostModel(BaseModel):
    """
    XGBoost模型实现类
    """
    
    def __init__(self, name: str = "xgboost_model", **kwargs):
        """
        初始化XGBoost模型
        
        Args:
            name: 模型名称
            **kwargs: XGBoost模型参数
        """
        super().__init__(name=name, **kwargs)
        
        # 设置默认参数
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # 更新参数
        self.model_params = default_params.copy()
        self.model_params.update(kwargs)
        
        # 初始化模型
        self.model = XGBRegressor(**self.model_params)
        
        # 更新元数据
        self.metadata["model_type"] = "XGBoost"
        self.metadata["params"] = self.model_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        训练XGBoost模型
        
        Args:
            X_train: 训练数据特征
            y_train: 训练数据标签
            **kwargs: 其他训练参数，如early_stopping_rounds, eval_set等
        """
        # 调用父类的train方法设置feature_names
        super().train(X_train, y_train, **kwargs)
        
        # 获取训练参数
        train_params = kwargs.copy()
        
        # 准备评估集
        eval_set = train_params.pop('eval_set', None)
        if eval_set is None and kwargs.get('early_stopping_rounds') is not None:
            # 如果设置了early_stopping_rounds但没有eval_set，使用10%的训练数据作为验证集
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
            eval_set = [(X_val, y_val)]
            # 更新训练数据
            X_train, y_train = X_tr, y_tr
        
        # 训练模型
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            **train_params
        )
        
        # 更新元数据
        self.metadata["training_params"] = train_params
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            self.metadata["best_iteration"] = self.model.best_iteration
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用XGBoost模型进行预测
        
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
        获取XGBoost模型的特征重要性
        
        Returns:
            包含特征名称和重要性分数的DataFrame
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("模型尚未训练或不支持特征重要性")
            
        # 获取特征重要性
        importances = self.model.feature_importances_
        
        # 如果没有特征名称，使用默认名称
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names
            
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # 按重要性降序排序
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取XGBoost模型参数
        
        Returns:
            模型参数字典
        """
        if self.model is None:
            return self.model_params
        return self.model.get_params()
    
    def set_params(self, **params) -> None:
        """
        设置XGBoost模型参数
        
        Args:
            **params: 模型参数
        """
        if self.model is None:
            self.model_params.update(params)
            self.model = XGBRegressor(**self.model_params)
        else:
            self.model.set_params(**params)
            self.model_params.update(params)
        
        # 更新元数据
        self.metadata["params"] = self.model_params 