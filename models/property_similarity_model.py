import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import joblib
import json
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import haversine_distances
import math

from models.base_model import BaseModel

class PropertySimilarityModel(BaseModel):
    """
    房产相似度计算模型
    基于KNN算法，使用自定义相似度计算逻辑筛选相似房产
    """
    
    def __init__(self, name: str = "property_similarity_model", **kwargs):
        """
        初始化房产相似度计算模型
        
        Args:
            name: 模型名称
            **kwargs: 模型参数
        """
        super().__init__(name=name, **kwargs)
        
        # 设置默认参数
        default_params = {
            'n_neighbors': 10,  # 默认获取10个最相似的房产
            'max_distance': 1.0,  # 最大距离1公里
            'max_bed_diff': 2,  # 最大卧室差异
            'max_days': 365,  # 最大天数差异（一年）
            'min_price_ratio': 0.7,  # 最小价格比例
            'max_price_ratio': 1.5,  # 最大价格比例
            'self_similarity_score': 25  # 目标房源自身的相似度得分
        }
        
        # 更新参数
        self.model_params = default_params.copy()
        self.model_params.update(kwargs)
        
        # 更新元数据
        self._metadata["model_type"] = "PropertySimilarity"
        self._metadata["params"] = self.model_params
        
        # 添加训练状态标志
        self._is_trained = False
        # 保存训练数据
        self._train_X = None
        self._train_y = None
    
    def calculate_distance(self, row1: pd.Series, row2: pd.Series) -> float:
        """
        计算两个房产之间的距离得分
        
        Args:
            row1: 第一个房产的特征
            row2: 第二个房产的特征
            
        Returns:
            距离得分
        """
        # 计算直线距离（公里）
        # 使用Haversine公式计算地理坐标之间的距离
        lat1, lon1 = row1['prop_y'], row1['prop_x']
        lat2, lon2 = row2['prop_y'], row2['prop_x']
        
        # 将经纬度转换为弧度
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # 地球半径（公里）
        earth_radius = 6371.0
        
        # Haversine公式
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = earth_radius * c
        
        # 计算额外的距离得分
        extra_distance = 0.0
        
        # 不在同一条街道加0.2分
        if row1['street_id'] != row2['street_id']:
            extra_distance += 0.2
        
        # 不在同一suburb加0.2分
        if row1['locality_id'] != row2['locality_id']:
            extra_distance += 0.2
        
        # 不在同一sector加0.2分
        if row1['sector_code'] != row2['sector_code']:
            extra_distance += 0.2
        
        # 总距离得分 = 直线距离 + 额外得分
        total_distance = distance + extra_distance
        
        return total_distance
    
    def calculate_date_diff(self, date1: str, date2: str) -> float:
        """
        计算两个日期之间的差异天数
        
        Args:
            date1: 第一个日期字符串（格式：YYYY-MM-DD）
            date2: 第二个日期字符串（格式：YYYY-MM-DD）
            
        Returns:
            日期差异天数
        """
        try:
            d1 = datetime.strptime(date1, "%Y-%m-%d")
            d2 = datetime.strptime(date2, "%Y-%m-%d")
            
            # 计算天数差异的绝对值
            days_diff = abs((d2 - d1).days)
            return days_diff
        except ValueError:
            # 如果日期格式不正确，返回最大值
            return float('inf')
    
    def calculate_similarity_score(self, target_property: pd.Series, similar_property: pd.Series, target_price: float) -> float:
        """
        计算目标房产与相似房产之间的相似度得分 - 优化版
        
        相似度得分计算因子权重调整:
        - 距离权重: 15.0 (增加，因为地理位置是房产价值的重要因素)
        - 日期差权重: 0.2 (增加，因为时间因素对价格影响较大)
        - 卧室数差权重: 7.0 (增加，卧室数量是基本特征)
        - 卫生间差权重: 3.0 (增加，卫生间数量也很重要)
        - 地价差权重: 15.0 (增加，地价是房产价值的基础)
        - 占地面积差权重: 12.0 (增加，占地面积是关键特征)
                   
        Args:
            target_property: 目标房产的特征
            similar_property: 相似房产的特征
            target_price: 目标房产的估价
            
        Returns:
            相似度得分（越小越相似）
        """
        # 如果是同一个房产，返回预设的自身相似度得分
        if target_property.equals(similar_property):
            return self.model_params['self_similarity_score']
        
        # 计算距离得分 - 权重提高到15.0
        distance_score = self.calculate_distance(target_property, similar_property) * 15.0
        
        # 计算日期差得分 - 权重提高到0.2
        date_diff = self.calculate_date_diff(target_property['sold_contract_date'], similar_property['sold_contract_date'])
        date_score = date_diff * 0.2
        
        # 计算卧室数差得分 - 权重提高到7.0
        bed_diff = abs(target_property['prop_bed'] - similar_property['prop_bed'])
        bed_score = bed_diff * 7.0
        
        # 计算卫生间差得分 - 权重提高到3.0
        bath_diff = abs(target_property['prop_bath'] - similar_property['prop_bath'])
        bath_score = bath_diff * 3.0
        
        # 计算车位差得分 - 保持不变
        carpark_diff = abs(target_property['prop_carpark'] - similar_property['prop_carpark'])
        carpark_score = min(carpark_diff * 0.5, 1)
        
        # 计算地价差得分 - 权重提高到15.0
        land_value_diff = abs(target_property['land_value'] - similar_property['land_value'])
        similar_price = similar_property.get('price', target_price)  # 如果没有价格信息，使用目标价格
        land_value_score = (land_value_diff / max(similar_price, 1)) * 15.0  # 避免除以零
        
        # 计算房屋特征未标注分 - 不变
        feature_score = 10 if similar_property.get('unreliable_features', False) else 0
        
        # 计算占地面积差得分 - 权重提高到12.0
        land_size_diff = abs(target_property['land_size'] - similar_property['land_size'])
        land_size_score = (land_size_diff / max(target_property['land_size'], 1)) * 12.0  # 避免除以零
        
        # 添加建筑面积差得分 (如果存在)
        building_size_score = 0
        if 'building_size' in target_property and 'building_size' in similar_property:
            building_size_diff = abs(target_property['building_size'] - similar_property['building_size'])
            building_size_score = (building_size_diff / max(target_property['building_size'], 1)) * 10.0  # 避免除以零
        
        # 添加楼层差得分 (如果是公寓且存在楼层信息)
        floor_score = 0
        if ('floor' in target_property and 'floor' in similar_property and 
            target_property['prop_type'] == 'Apartment' and similar_property['prop_type'] == 'Apartment'):
            floor_diff = abs(target_property['floor'] - similar_property['floor'])
            floor_score = floor_diff * 5.0
        
        # 总相似度得分
        total_score = (
            distance_score + 
            date_score + 
            bed_score + 
            bath_score + 
            carpark_score + 
            land_value_score + 
            feature_score + 
            land_size_score + 
            building_size_score + 
            floor_score
        )
        
        return total_score
    
    def filter_similar_properties(self, target_property: pd.Series, properties: pd.DataFrame, target_price: float) -> pd.DataFrame:
        """
        根据条件筛选相似房产 - 性能优化版
        
        筛选条件：
        - 在同一个council
        - 属于同一个zoning
        - 卧室数差距不超过2
        - 都是House类别
        - 直线距离不超过1km
        - 成交时间距现在不超过一年
        - 成交价在目标房源估价的70%-150%范围内
        
        Args:
            target_property: 目标房产的特征
            properties: 候选房产数据框
            target_price: 目标房源估价
            
        Returns:
            筛选后的相似房产数据框
        """
        # 使用boolean索引而不是复制数据框
        # 筛选条件1：在同一个council
        mask_locality = properties['locality_id'] == target_property['locality_id']
        
        # 筛选条件2：属于同一个zoning
        mask_zoning = properties['prop_zoning'] == target_property['prop_zoning']
        
        # 筛选条件3：卧室数差距严格小于2（按要求排除差距>=2的）
        bed_diff = (properties['prop_bed'] - target_property['prop_bed']).abs()
        mask_bed = bed_diff < 2
        
        # 筛选条件4：都是House类别
        mask_type = properties['prop_type'] == 'House'
        
        # 筛选条件7：成交价在目标房源估价的70%-150%范围内
        min_price = target_price * self.model_params['min_price_ratio']
        max_price = target_price * self.model_params['max_price_ratio']
        mask_price = (properties['price'] >= min_price) & (properties['price'] <= max_price)
        
        # 组合所有基本筛选条件
        mask_combined = mask_locality & mask_zoning & mask_bed & mask_type & mask_price
        
        # 使用组合掩码过滤数据，避免多次复制
        filtered_properties = properties.loc[mask_combined].copy()
        
        # 如果没有通过基本筛选的房产，直接返回空DataFrame
        if len(filtered_properties) == 0:
            return filtered_properties
        
        # 计算直线距离（仅对基本筛选后的数据进行计算）
        distances = []
        for _, row in filtered_properties.iterrows():
            distance = self.calculate_distance(target_property, row)
            distances.append(distance)
        
        filtered_properties['distance'] = distances
        mask_distance = filtered_properties['distance'] <= self.model_params['max_distance']
        filtered_properties = filtered_properties.loc[mask_distance]
        
        # 如果没有通过距离筛选的房产，直接返回空DataFrame
        if len(filtered_properties) == 0:
            return filtered_properties
        
        # 计算日期差异（仅对基本筛选和距离筛选后的数据进行计算）
        current_date = datetime.now().strftime("%Y-%m-%d")
        date_diffs = []
        for _, row in filtered_properties.iterrows():
            date_diff = self.calculate_date_diff(current_date, row['sold_contract_date'])
            date_diffs.append(date_diff)
        
        filtered_properties['date_diff'] = date_diffs
        mask_date = filtered_properties['date_diff'] <= self.model_params['max_days']
        filtered_properties = filtered_properties.loc[mask_date]
        
        return filtered_properties
    
    def find_similar_properties(self, target_property: pd.Series, properties: pd.DataFrame, target_price: float) -> pd.DataFrame:
        """
        寻找相似房产并计算相似度得分 - 性能优化版
        
        Args:
            target_property: 目标房产的特征
            properties: 候选房产数据框
            target_price: 目标房源估价
            
        Returns:
            相似房产数据框，包含相似度得分
        """
        # 筛选符合条件的房产
        filtered_properties = self.filter_similar_properties(target_property, properties, target_price)
        
        # 如果没有符合条件的房产，返回空DataFrame
        if len(filtered_properties) == 0:
            return pd.DataFrame()
        
        # 计算相似度得分
        similarity_scores = []
        for _, row in filtered_properties.iterrows():
            score = self.calculate_similarity_score(target_property, row, target_price)
            similarity_scores.append(score)
        
        # 添加相似度得分列
        filtered_properties.loc[:, 'similarity_score'] = similarity_scores
        
        # 按相似度得分排序（升序，得分越小越相似）
        sorted_properties = filtered_properties.sort_values('similarity_score')
        
        # 只返回前n_neighbors个最相似的房产
        n_neighbors = min(self.model_params['n_neighbors'], len(sorted_properties))
        top_properties = sorted_properties.head(n_neighbors)
        
        return top_properties
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        训练模型 - 在相似度模型中，实际上是存储训练数据，以便在预测时使用
        
        Args:
            X: 训练特征
            y: 训练目标值
        """
        print(f"开始训练房产相似度模型...")
        print(f"训练样本数量: {len(X)}")
        print(f"特征数量: {len(X.columns)}")
        
        # 保存训练数据以便后续使用
        self._train_X = X.copy()
        self._train_y = y.copy()
        
        # 合并特征和目标值，形成完整的房产数据库
        self._properties_db = X.copy()
        self._properties_db['price'] = y.copy()
        
        # 标记模型为已训练
        self._is_trained = True
        
        # 更新元数据
        self._metadata["train_samples"] = len(X)
        self._metadata["train_features"] = list(X.columns)
        self._metadata["train_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"房产相似度模型训练完成!")
        print(f"房产数据库中共有 {len(self._properties_db)} 条记录")
        print(f"模型参数设置: ")
        for key, value in self.model_params.items():
            print(f"  - {key}: {value}")
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        评估模型
        
        对于相似度模型，我们通过比较每个房产的真实价格与其找到的相似房产的平均价格来评估
        每100个样本打印一次平均预测误差
        
        Args:
            X: 测试特征
            y: 测试目标值
            
        Returns:
            评估指标字典
        """
        if not self._is_trained:
            raise RuntimeError("模型尚未训练")
        
        # 预测值和误差
        y_pred = []
        errors = []
        percent_errors = []
        
        # 计算总样本数和每10%的样本数
        total_samples = len(X)
        progress_step = max(1, total_samples // 10)
        
        print(f"开始评估模型，共{total_samples}个测试样本...")
        
        # 对每个测试样本找到相似的房产
        for i, (idx, target_property) in enumerate(X.iterrows()):
            # 显示进度
            if (i + 1) % progress_step == 0 or i == 0 or i == total_samples - 1:
                progress_percent = min(100, int((i + 1) / total_samples * 100))
                print(f"评估进度: {progress_percent}% ({i+1}/{total_samples})")
            
            target_price = y.iloc[i]
            
            # 创建不包含当前测试样本的临时数据库
            # 我们需要通过索引来排除当前测试样本
            temp_db = self._properties_db.copy()
            
            # 如果测试样本的索引存在于训练数据库中，则排除它
            if idx in temp_db.index:
                temp_db = temp_db.drop(idx)
            
            # 寻找相似房产
            similar_properties = self.find_similar_properties(
                target_property,
                temp_db,
                target_price
            )
            
            # 如果找到相似房产，使用其平均价格作为预测值
            if len(similar_properties) > 0:
                pred_price = similar_properties['price'].mean()
            else:
                # 如果没有找到相似房产，使用训练集中位数价格
                pred_price = self._train_y.median()
            
            y_pred.append(pred_price)
            
            # 计算误差
            error = abs(target_price - pred_price)
            errors.append(error)
            
            # 计算百分比误差
            percent_error = (error / target_price) * 100 if target_price > 0 else 0
            percent_errors.append(percent_error)
            
            # 每100个样本打印一次平均预测误差
            if (i + 1) % 100 == 0:
                avg_error = sum(errors) / len(errors)
                median_error = np.median(errors)
                avg_percent_error = sum(percent_errors) / len(percent_errors)
                median_percent_error = np.median(percent_errors)
                
                # 计算最近100个样本的平均误差和中位数误差
                recent_errors = errors[-100:]
                recent_percent_errors = percent_errors[-100:]
                recent_avg_error = sum(recent_errors) / len(recent_errors)
                recent_median_error = np.median(recent_errors)
                recent_avg_percent_error = sum(recent_percent_errors) / len(recent_percent_errors)
                recent_median_percent_error = np.median(recent_percent_errors)
                
                print(f"已评估 {i+1} 个样本:")
                print(f"  - 总平均绝对误差: {avg_error:.2f}")
                print(f"  - 总中位数绝对误差: {median_error:.2f}")
                print(f"  - 总平均百分比误差: {avg_percent_error:.2f}%")
                print(f"  - 总中位数百分比误差: {median_percent_error:.2f}%")
                print(f"  - 最近100个样本平均绝对误差: {recent_avg_error:.2f}")
                print(f"  - 最近100个样本中位数绝对误差: {recent_median_error:.2f}")
                print(f"  - 最近100个样本平均百分比误差: {recent_avg_percent_error:.2f}%")
                print(f"  - 最近100个样本中位数百分比误差: {recent_median_percent_error:.2f}%")
        
        print("完成所有测试样本评估，正在计算最终评估指标...")
        
        # 计算评估指标
        y_true = y.values
        y_pred = np.array(y_pred)
        
        # 计算均方误差和均方根误差
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # 计算平均绝对误差
        mae = np.mean(np.abs(y_true - y_pred))
        
        # 计算中位数绝对误差
        median_ae = np.median(np.abs(y_true - y_pred))
        
        # 计算决定系数R²
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # 计算平均百分比误差
        mape = np.mean(np.abs((y_true - y_pred) / y_true) * 100)
        
        # 计算中位数百分比误差
        mdape = np.median(np.abs((y_true - y_pred) / y_true) * 100)
        
        # 返回评估指标
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "median_ae": median_ae,
            "r2": r2,
            "mape": mape,
            "mdape": mdape
        }
        
        print(f"评估完成!")
        print(f"- RMSE: {rmse:.4f}")
        print(f"- MAE (平均绝对误差): {mae:.4f}")
        print(f"- MedAE (中位数绝对误差): {median_ae:.4f}")
        print(f"- MAPE (平均百分比误差): {mape:.2f}%")
        print(f"- MdAPE (中位数百分比误差): {mdape:.2f}%")
        print(f"- R²: {r2:.4f}")
        
        # 打印误差分布
        percentiles = [10, 25, 50, 75, 90]
        percent_error_values = np.percentile(percent_errors, percentiles)
        print("\n误差分布:")
        for i, p in enumerate(percentiles):
            print(f"- {p}% 的样本误差低于: {percent_error_values[i]:.2f}%")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测目标房产的价格
        
        Args:
            X: 目标房产特征
            
        Returns:
            预测价格数组
        """
        if not self._is_trained:
            raise RuntimeError("模型尚未训练")
        
        # 预测值（使用相似房产的平均价格作为预测值）
        y_pred = []
        
        # 计算总样本数和每10%的样本数
        total_samples = len(X)
        progress_step = max(1, total_samples // 10)
        
        print(f"开始预测，共{total_samples}个样本...")
        
        # 对每个测试样本找到相似的房产
        for i, (idx, target_property) in enumerate(X.iterrows()):
            # 显示进度
            if (i + 1) % progress_step == 0 or i == 0 or i == total_samples - 1:
                progress_percent = min(100, int((i + 1) / total_samples * 100))
                print(f"预测进度: {progress_percent}% ({i+1}/{total_samples})")
            
            # 使用初始估计价格(中位数价格)作为目标价格进行相似性过滤
            initial_price_estimate = self._train_y.median()
            
            # 创建不包含当前样本的临时数据库
            temp_db = self._properties_db.copy()
            
            # 如果预测样本的索引存在于训练数据库中，则排除它
            if idx in temp_db.index:
                temp_db = temp_db.drop(idx)
            
            # 寻找相似房产
            similar_properties = self.find_similar_properties(
                target_property,
                temp_db,
                initial_price_estimate
            )
            
            # 如果找到相似房产，使用其平均价格作为预测值
            if len(similar_properties) > 0:
                pred_price = similar_properties['price'].mean()
            else:
                # 如果没有找到相似房产，使用训练集中位数价格
                pred_price = initial_price_estimate
            
            y_pred.append(pred_price)
        
        print(f"预测完成! 共预测 {total_samples} 个样本")
        
        return np.array(y_pred)
    
    def predict_single(self, target_property: pd.Series, properties: pd.DataFrame, target_price: float) -> pd.DataFrame:
        """
        为单个目标房产寻找相似房产
        
        Args:
            target_property: 目标房产的特征
            properties: 候选房产数据框
            target_price: 目标房源估价
            
        Returns:
            相似房产数据框
        """
        return self.find_similar_properties(target_property, properties, target_price)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        注意: 该模型使用自定义相似度逻辑，不存在传统意义的特征重要性。
        此方法返回的是各个相似度计算组成部分的权重。
        
        Returns:
            特征重要性数据框，包含feature和importance两列
        """
        # 定义各个相似度计算组成部分及其权重
        components = [
            {"feature": "距离", "importance": 15.0},
            {"feature": "日期差", "importance": 0.2},
            {"feature": "卧室数差", "importance": 7.0},
            {"feature": "卫生间差", "importance": 3.0},
            {"feature": "车位差", "importance": 0.5},
            {"feature": "地价差", "importance": 15.0},
            {"feature": "房屋特征未标注", "importance": 10.0},
            {"feature": "占地面积差", "importance": 12.0},
            {"feature": "建筑面积差", "importance": 10.0},
            {"feature": "楼层差(公寓)", "importance": 5.0}
        ]
        
        # 将列表转换为DataFrame
        importance_df = pd.DataFrame(components)
        
        return importance_df 