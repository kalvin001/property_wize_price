import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import joblib
import json
import math
from collections import OrderedDict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import pairwise_distances

from models.knn_model import KNNModel

class GeographicKNNModel(KNNModel):
    """
    基于地理距离的KNN模型实现类
    使用Haversine距离计算经纬度之间的实际地理距离
    """
    
    def __init__(self, name: str = "geographic_knn_model", **kwargs):
        """
        初始化GeographicKNN模型
        
        Args:
            name: 模型名称
            **kwargs: KNN模型参数
        """
        # 提取KNeighborsRegressor不需要的自定义参数
        self.lat_col = kwargs.pop('lat_col', 'prop_y')
        self.lon_col = kwargs.pop('lon_col', 'prop_x')
        
        # 设置默认值 
        if 'n_neighbors' not in kwargs:
            kwargs['n_neighbors'] = 7  # 地理距离模型适合稍多的邻居
        if 'weights' not in kwargs:
            kwargs['weights'] = 'distance'  # 距离加权

        # 初始化父类
        super().__init__(name=name, **kwargs)
        
        # 更新元数据
        self._metadata["model_type"] = "GeographicKNN"
        self._metadata["params"].update({
            "lat_col": self.lat_col,
            "lon_col": self.lon_col
        })
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        使用Haversine公式计算两点间的实际距离（公里）
        
        Args:
            lat1: 第一点纬度
            lon1: 第一点经度
            lat2: 第二点纬度
            lon2: 第二点经度
            
        Returns:
            两点之间的距离（公里）
        """
        # 将经纬度转换为弧度
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # 地球半径，单位为公里
        
        return c * r
    
    def haversine_distance_matrix(self, X1, X2=None):
        """
        计算两组经纬度点之间的Haversine距离矩阵
        
        Args:
            X1: 第一组经纬度点，格式为[[lat1, lon1], [lat2, lon2], ...]
            X2: 第二组经纬度点，格式同X1，如果为None则计算X1与自身的距离
            
        Returns:
            距离矩阵
        """
        if X2 is None:
            X2 = X1
        
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        dm = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                dm[i, j] = self.haversine_distance(
                    X1[i, 0], X1[i, 1], 
                    X2[j, 0], X2[j, 1]
                )
        
        return dm
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        训练GeographicKNN模型
        
        Args:
            X_train: 训练数据特征
            y_train: 训练数据标签
            **kwargs: 其他训练参数
        """
        # 调用父类的train方法设置feature_names
        super(KNNModel, self).train(X_train, y_train, **kwargs)
        
        # 确保经纬度列存在
        if self.lat_col not in X_train.columns or self.lon_col not in X_train.columns:
            raise ValueError(f"训练数据中缺少经纬度列: {self.lat_col}, {self.lon_col}")
        
        # 提取经纬度特征
        self.geo_features = X_train[[self.lat_col, self.lon_col]].values
        
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
        distance_matrix = self.haversine_distance_matrix(self.geo_features)
        
        # 使用距离矩阵训练模型
        self.model.fit(distance_matrix, y_train)
        
        # 保存训练数据中的经纬度值，用于后续预测
        self._metadata["train_geo_data"] = self.geo_features.tolist()
        
        # 更新元数据
        self._metadata["training_params"] = kwargs
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用GeographicKNN模型进行预测
        
        Args:
            X: 预测数据特征
            
        Returns:
            预测结果数组
        """
        # 先调用父类的predict方法进行检查
        super(KNNModel, self).predict(X)
        
        # 确保经纬度列存在
        if self.lat_col not in X.columns or self.lon_col not in X.columns:
            raise ValueError(f"预测数据中缺少经纬度列: {self.lat_col}, {self.lon_col}")
        
        # 提取预测数据中的经纬度特征
        test_geo_features = X[[self.lat_col, self.lon_col]].values
        
        # 从元数据中获取训练数据的经纬度
        train_geo_features = np.array(self._metadata.get("train_geo_data", []))
        if len(train_geo_features) == 0:
            raise ValueError("模型尚未训练或训练数据的经纬度未保存")
        
        # 计算测试数据与训练数据之间的距离矩阵
        distance_matrix = self.haversine_distance_matrix(test_geo_features, train_geo_features)
        
        # 使用距离矩阵进行预测
        return self.model.predict(distance_matrix)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        注意: 由于本模型仅使用经纬度特征，因此仅返回经纬度列的重要性。
        
        Returns:
            特征重要性数据框，包含feature和importance两列
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法获取特征重要性")
        
        # 对于地理KNN模型，经纬度特征的重要性是相同的
        importance_list = [
            {"feature": self.lat_col, "importance": 0.5},
            {"feature": self.lon_col, "importance": 0.5}
        ]
        
        # 将列表转换为DataFrame
        importance_df = pd.DataFrame(importance_list)
        
        return importance_df 