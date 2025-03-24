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
        
        # 设置默认参数 - 优化后更强大的默认参数
        default_params = {
            'n_estimators': 10000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # 更快的直方图算法
            'booster': 'gbtree'
        }
        
        # 更新参数
        self.model_params = default_params.copy()
        self.model_params.update(kwargs)
        
        # 初始化模型
        self.model = XGBRegressor(**self.model_params)
        
        # 更新元数据
        self._metadata["model_type"] = "XGBoost"
        self._metadata["params"] = self.model_params
    
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
        self._metadata["training_params"] = train_params
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            self._metadata["best_iteration"] = self.model.best_iteration
    
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
        获取特征重要性
        
        Returns:
            特征重要性数据框，包含feature和importance两列
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法获取特征重要性")
        
        # 获取特征重要性
        importance_type = 'gain'  # 可选: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
        feature_importances = self.model.get_booster().get_score(importance_type=importance_type)
        
        # 将特征重要性转换为DataFrame
        if not feature_importances:
            # 如果没有获取到特征重要性，可能是因为模型太简单或特征未使用
            # 创建一个空的DataFrame
            return pd.DataFrame(columns=["feature", "importance"])
            
        # 创建重要性列表
        importance_list = []
        for feature, importance in feature_importances.items():
            # 在XGBoost中，特征名称可能是f0, f1, f2等格式
            # 如果有原始特征名称，尝试映射回去
            feature_name = feature
            if feature.startswith('f') and feature[1:].isdigit():
                # 尝试将f0, f1等映射回原始特征名称
                try:
                    index = int(feature[1:])
                    if self._feature_names and index < len(self._feature_names):
                        feature_name = self._feature_names[index]
                except:
                    pass
                    
            importance_list.append({
                "feature": feature_name,
                "importance": importance
            })
        
        # 将列表转换为DataFrame并按重要性降序排序
        importance_df = pd.DataFrame(importance_list)
        importance_df = importance_df.sort_values("importance", ascending=False)
        
        return importance_df
    
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
        self._metadata["params"] = self.model_params 