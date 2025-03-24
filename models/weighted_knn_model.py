import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import joblib
import json
from collections import OrderedDict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from models.knn_model import KNNModel

class WeightedKNNModel(KNNModel):
    """
    基于多特征权重的KNN模型实现类
    对不同特征赋予不同权重计算距离
    """
    
    def __init__(self, name: str = "weighted_knn_model", **kwargs):
        """
        初始化WeightedKNN模型
        
        Args:
            name: 模型名称
            **kwargs: KNN模型参数
        """
        # 提取KNeighborsRegressor不需要的自定义参数
        self.feature_weights = kwargs.pop('feature_weights', {})
        self.auto_weights = kwargs.pop('auto_weights', True)
        self.standardize = kwargs.pop('standardize', True)
        
        # 默认设置一些常用特征的权重
        if not self.feature_weights:
            self.feature_weights = {
                'internal_area': 2.0,     # 面积是很重要的特征
                'bedrooms': 1.5,          # 卧室数量较重要
                'bathrooms': 1.2,         # 浴室数量
                'latitude': 1.5,          # 纬度（位置）
                'longitude': 1.5,         # 经度（位置）
                'age': 1.0,               # 房龄
                'locality_id': 1.0        # 区域ID
            }
            
        # 设置默认值
        if 'n_neighbors' not in kwargs:
            kwargs['n_neighbors'] = 6
        if 'weights' not in kwargs:
            kwargs['weights'] = 'distance'
            
        # 初始化父类
        super().__init__(name=name, **kwargs)
        
        # 初始化标准化器
        self.scaler = StandardScaler() if self.standardize else None
        
        # 更新元数据
        self._metadata["model_type"] = "WeightedKNN"
        self._metadata["params"].update({
            "feature_weights": self.feature_weights,
            "auto_weights": self.auto_weights,
            "standardize": self.standardize
        })
    
    def weighted_distance(self, X1, X2, weights):
        """
        计算加权欧几里得距离
        
        Args:
            X1: 第一组样本特征
            X2: 第二组样本特征
            weights: 特征权重数组
            
        Returns:
            加权距离矩阵
        """
        # 应用特征权重
        X1_weighted = X1 * np.sqrt(weights)
        X2_weighted = X2 * np.sqrt(weights)
        
        # 计算欧几里得距离
        return pairwise_distances(X1_weighted, X2_weighted, metric='euclidean')
    
    def calculate_feature_weights(self, X_train):
        """
        自动计算特征权重
        
        实现一个简单的启发式算法，对面积、位置等重要特征给予更高权重
        
        Args:
            X_train: 训练数据特征
            
        Returns:
            特征权重字典
        """
        weights = {}
        
        # 遍历所有特征
        for feature in X_train.columns:
            # 对特征名称进行判断，根据业务规则设置初始权重
            feature_lower = feature.lower()
            
            # 面积特征给予较高权重
            if 'area' in feature_lower or 'size' in feature_lower:
                weights[feature] = 2.0
            # 地理位置特征给予较高权重
            elif 'location' in feature_lower or 'lat' in feature_lower or 'lon' in feature_lower or 'coord' in feature_lower:
                weights[feature] = 1.5
            # 房龄、房间数等特征给予中等权重
            elif 'age' in feature_lower or 'room' in feature_lower or 'bath' in feature_lower:
                weights[feature] = 1.0
            # 其他特征给予较低权重
            else:
                weights[feature] = 0.5
        
        return weights
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        训练WeightedKNN模型
        
        Args:
            X_train: 训练数据特征
            y_train: 训练数据标签
            **kwargs: 其他训练参数
        """
        # 调用父类的train方法设置feature_names
        super(KNNModel, self).train(X_train, y_train, **kwargs)
        
        # 如果启用自动权重计算且没有提供权重，则计算特征权重
        if self.auto_weights and not self.feature_weights:
            self.feature_weights = self.calculate_feature_weights(X_train)
            self._metadata["params"]["feature_weights"] = self.feature_weights
        
        # 如果提供了权重字典，但不包含所有特征，则为缺失的特征设置默认权重
        if self.feature_weights:
            for feature in X_train.columns:
                if feature not in self.feature_weights:
                    self.feature_weights[feature] = 1.0
        else:
            # 如果没有提供权重且不使用自动权重，则对所有特征使用相同权重
            self.feature_weights = {feature: 1.0 for feature in X_train.columns}
        
        # 创建权重数组，与特征列顺序一致
        self.weight_array = np.array([self.feature_weights.get(feature, 1.0) for feature in X_train.columns])
        
        # 数据标准化
        if self.standardize:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train.values
        
        # 使用自定义度量构建KNN模型
        self.model = KNeighborsRegressor(
            n_neighbors=self.model_params['n_neighbors'],
            weights=self.model_params['weights'],
            algorithm='brute',  # 使用暴力算法与自定义距离
            leaf_size=self.model_params['leaf_size'],
            n_jobs=self.model_params['n_jobs'],
            metric='precomputed'  # 使用预计算的距离矩阵
        )
        
        # 计算训练集的距离矩阵
        distance_matrix = self.weighted_distance(X_train_scaled, X_train_scaled, self.weight_array)
        
        # 使用距离矩阵训练模型
        self.model.fit(distance_matrix, y_train)
        
        # 保存训练数据，用于后续预测
        self._metadata["train_data"] = X_train_scaled.tolist()
        
        # 更新元数据
        self._metadata["training_params"] = kwargs
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用WeightedKNN模型进行预测
        
        Args:
            X: 预测数据特征
            
        Returns:
            预测结果数组
        """
        # 先调用父类的predict方法进行检查
        super(KNNModel, self).predict(X)
        
        # 确保数据只包含模型训练时使用的特征
        if self._feature_names is not None:
            missing_features = [f for f in self._feature_names if f not in X.columns]
            if missing_features:
                # 如果有缺失的特征，填充为0
                for feature in missing_features:
                    X[feature] = 0
            
            # 只使用模型训练时的特征，并保持相同的顺序
            X = X[self._feature_names]
        
        # 数据标准化
        if self.standardize and self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # 从元数据中获取训练数据
        train_data = np.array(self._metadata.get("train_data", []))
        if len(train_data) == 0:
            raise ValueError("模型尚未训练或训练数据未保存")
        
        # 计算测试数据与训练数据之间的加权距离矩阵
        distance_matrix = self.weighted_distance(X_scaled, train_data, self.weight_array)
        
        # 使用距离矩阵进行预测
        return self.model.predict(distance_matrix)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        对于WeightedKNN模型，特征重要性由特征权重决定
        
        Returns:
            特征重要性数据框，包含feature和importance两列
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法获取特征重要性")
        
        # 如果没有特征名称，使用默认名称
        if self._feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.weight_array))]
        else:
            feature_names = self._feature_names
        
        # 创建特征重要性列表
        importance_list = []
        
        # 确保我们有特征权重
        if hasattr(self, 'weight_array') and len(self.weight_array) > 0:
            # 归一化权重作为重要性
            total_weight = np.sum(self.weight_array)
            if total_weight > 0:
                normalized_weights = self.weight_array / total_weight
            else:
                normalized_weights = np.ones_like(self.weight_array) / len(self.weight_array)
            
            for feature, importance in zip(feature_names, normalized_weights):
                importance_list.append({
                    "feature": feature,
                    "importance": float(importance)
                })
        else:
            # 如果没有权重，则所有特征重要性相同
            importance_value = 1.0 / len(feature_names)
            for feature in feature_names:
                importance_list.append({
                    "feature": feature,
                    "importance": importance_value
                })
        
        # 将列表转换为DataFrame
        importance_df = pd.DataFrame(importance_list)
        
        return importance_df 