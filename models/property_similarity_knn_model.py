import pandas as pd
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional
import os
import joblib
import json
from collections import OrderedDict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
from functools import partial

from models.knn_model import KNNModel

class PropertySimilarityKNNModel(KNNModel):
    """
    基于房产相似性的KNN模型实现类
    使用price_analysis.py中的方法计算房产相似性
    """
    
    def __init__(self, name: str = "property_similarity_knn_model", **kwargs):
        """
        初始化PropertySimilarityKNN模型
        
        Args:
            name: 模型名称
            **kwargs: KNN模型参数
        """
        # 提取KNeighborsRegressor不需要的自定义参数
        self.lat_col = kwargs.pop('lat_col', 'prop_y')
        self.lon_col = kwargs.pop('lon_col', 'prop_x')
        self.area_col = kwargs.pop('area_col', 'internal_area')
        self.type_col = kwargs.pop('type_col', 'prop_type')
        self.locality_id_col = kwargs.pop('locality_id_col', 'locality_id')
        self.x_col = kwargs.pop('x_col', 'prop_x')
        self.y_col = kwargs.pop('y_col', 'prop_y')
        self.distance_weight = kwargs.pop('distance_weight', 0.6)  # 增加距离权重
        self.area_weight = kwargs.pop('area_weight', 0.4)  # 减少面积权重
        self.standardize = kwargs.pop('standardize', True)
        self.precompute_matrix = kwargs.pop('precompute_matrix', False)  # 默认不预计算完整矩阵
        
        # 设置默认值
        if 'n_neighbors' not in kwargs:
            kwargs['n_neighbors'] = 8  # 房产相似度模型可以使用更多邻居
        if 'weights' not in kwargs:
            kwargs['weights'] = 'distance'  # 距离加权
        
        # 初始化父类
        super().__init__(name=name, **kwargs)
        
        # 初始化标准化器
        self.scaler = StandardScaler() if self.standardize else None
        
        # 初始化数据存储
        self.X_train_data = None
        
        # 更新元数据
        self._metadata["model_type"] = "PropertySimilarityKNN"
        self._metadata["params"].update({
            "lat_col": self.lat_col,
            "lon_col": self.lon_col,
            "area_col": self.area_col,
            "type_col": self.type_col,
            "locality_id_col": self.locality_id_col,
            "x_col": self.x_col,
            "y_col": self.y_col,
            "distance_weight": self.distance_weight,
            "area_weight": self.area_weight,
            "standardize": self.standardize,
            "precompute_matrix": self.precompute_matrix
        })
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
    
    def calculate_euclidean_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        计算欧几里得距离
        
        Args:
            x1: 第一点x坐标
            y1: 第一点y坐标
            x2: 第二点x坐标
            y2: 第二点y坐标
            
        Returns:
            欧几里得距离
        """
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    def calculate_property_similarity(self, prop1, prop2) -> float:
        """
        计算两个房产之间的相似度
        
        Args:
            prop1: 第一个房产数据（Series或dict）
            prop2: 第二个房产数据（Series或dict）
            
        Returns:
            相似度得分（值越小表示相似度越高）
        """
        # 如果房产类型不同，设置一个很大的值表示不相似
        if self.type_col in prop1 and self.type_col in prop2:
            if prop1[self.type_col] != prop2[self.type_col]:
                return 1000
        
        # 如果locality_id存在且不同，增加不相似度
        locality_penalty = 0
        if self.locality_id_col in prop1 and self.locality_id_col in prop2:
            loc1 = prop1.get(self.locality_id_col)
            loc2 = prop2.get(self.locality_id_col)
            if loc1 and loc2 and loc1 != loc2:
                locality_penalty = 10
        
        # 判断使用哪种距离计算方法
        use_latlon = all(col in prop1 and col in prop2 for col in [self.lat_col, self.lon_col])
        use_xy = all(col in prop1 and col in prop2 for col in [self.x_col, self.y_col])
        
        # 计算地理距离
        distance = 0
        if use_latlon:
            try:
                lat1 = float(prop1[self.lat_col])
                lon1 = float(prop1[self.lon_col])
                lat2 = float(prop2[self.lat_col])
                lon2 = float(prop2[self.lon_col])
                
                if all([lat1, lon1, lat2, lon2]):  # 确保经纬度有效
                    distance = self.calculate_distance(lat1, lon1, lat2, lon2)
            except (ValueError, TypeError):
                # 如果经纬度计算失败，尝试使用坐标
                if use_xy:
                    distance = self.calculate_euclidean_distance(
                        prop1[self.x_col], 
                        prop1[self.y_col],
                        prop2[self.x_col], 
                        prop2[self.y_col]
                    ) * 0.1  # 转换系数
        elif use_xy:
            # 使用坐标计算欧几里得距离
            distance = self.calculate_euclidean_distance(
                prop1[self.x_col], 
                prop1[self.y_col],
                prop2[self.x_col], 
                prop2[self.y_col]
            ) * 0.1  # 转换系数
        
        # 计算面积差异百分比
        area_diff = 0
        if self.area_col in prop1 and self.area_col in prop2:
            area1 = float(prop1[self.area_col] or 0)
            area2 = float(prop2[self.area_col] or 0)
            if area1 > 0 and area2 > 0:
                area_diff = abs(area1 - area2) / max(area1, area2) * 100
        
        # 计算综合相似度得分（距离权重和面积权重）
        # 距离和面积差异越小，相似度越高（得分越低）
        similarity_score = (
            self.distance_weight * distance + 
            self.area_weight * (area_diff / 10) + 
            locality_penalty
        )
        
        return similarity_score
    
    def custom_distance_metric(self, x, y):
        """
        自定义距离度量函数，用于KNN算法
        
        Args:
            x: 样本1的特征向量
            y: 样本2的特征向量
            
        Returns:
            两个样本之间的距离
        """
        # x和y是从X_train或X_test中提取的特征向量
        # 需要将它们转换回原始的房产数据格式
        # 使用indices_map将索引映射到X_train_data中的位置
        
        # 通过索引找到原始房产数据
        idx1 = int(x[0])  # 假设第一个特征是索引
        idx2 = int(y[0])  # 假设第一个特征是索引
        
        # 计算两个房产之间的相似度
        similarity = self.calculate_property_similarity(
            self.X_train_data.iloc[idx1], 
            self.X_train_data.iloc[idx2]
        )
        
        return similarity
    
    def calculate_similarity_matrix(self, props_df: pd.DataFrame) -> np.ndarray:
        """
        计算房产之间的相似度矩阵
        
        Args:
            props_df: 房产数据框
            
        Returns:
            相似度矩阵（值越小表示相似度越高）
        """
        n = len(props_df)
        similarity_matrix = np.zeros((n, n))
        
        # 确保必要的列存在
        required_cols = [self.type_col, self.area_col]
        
        # 根据可用信息选择距离计算方法
        use_latlon = all(col in props_df.columns for col in [self.lat_col, self.lon_col])
        use_xy = all(col in props_df.columns for col in [self.x_col, self.y_col])
        
        start_time = time.time()
        print(f"开始计算{n}个房产之间的相似度矩阵...")
        
        # 使用tqdm来显示进度
        total_steps = n * (n + 1) // 2  # 只计算上三角矩阵
        with tqdm(total=total_steps, desc="计算相似度矩阵") as pbar:
            processed = 0
            for i in range(n):
                for j in range(i, n):  # 相似度矩阵是对称的，只需计算上三角
                    # 如果是同一个房产，相似度设为0（完全相似）
                    if i == j:
                        similarity_matrix[i, j] = 0
                        pbar.update(1)
                        processed += 1
                        continue
                    
                    # 计算两个房产之间的相似度
                    similarity_score = self.calculate_property_similarity(
                        props_df.iloc[i], 
                        props_df.iloc[j]
                    )
                    
                    # 填充相似度矩阵（对称矩阵）
                    similarity_matrix[i, j] = similarity_matrix[j, i] = similarity_score
                    
                    # 更新进度条
                    pbar.update(1)
                    processed += 1
                
                # 每处理50个房产，显示一次进度
                if (i + 1) % 50 == 0 or i == n - 1:
                    elapsed = time.time() - start_time
                    progress = (i + 1) / n
                    estimated_total = elapsed / progress if progress > 0 else 0
                    remaining = max(0, estimated_total - elapsed)
                    print(f"已处理 {i+1}/{n} 个房产 ({progress:.1%}), 已用时间: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s")
        
        elapsed = time.time() - start_time
        print(f"相似度矩阵计算完成，总耗时: {elapsed:.2f}秒")
        return similarity_matrix
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        训练PropertySimilarityKNN模型
        
        Args:
            X_train: 训练数据特征
            y_train: 训练数据标签
            **kwargs: 其他训练参数
        """
        start_time = time.time()
        print(f"开始训练PropertySimilarityKNN模型，数据集大小: {len(X_train)}行 x {len(X_train.columns)}列")
        
        # 调用父类的train方法设置feature_names
        super(KNNModel, self).train(X_train, y_train, **kwargs)
        
        # 验证必要的列
        required_cols = [self.area_col, self.type_col]
        missing_cols = [col for col in required_cols if col not in X_train.columns]
        if missing_cols:
            raise ValueError(f"训练数据中缺少必要的列: {missing_cols}")
        
        # 检查地理位置信息
        has_latlon = all(col in X_train.columns for col in [self.lat_col, self.lon_col])
        has_xy = all(col in X_train.columns for col in [self.x_col, self.y_col])
        
        if not (has_latlon or has_xy):
            raise ValueError("训练数据中缺少地理位置信息，需要提供经纬度或坐标")
        
        print("数据验证通过，开始预处理...")
        preprocess_start = time.time()
        
        # 数据预处理和标准化
        if self.standardize:
            # 只标准化数值列，不包括分类特征
            numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # 排除要特殊处理的列
            special_cols = [self.type_col, self.locality_id_col]
            numeric_cols = [col for col in numeric_cols if col not in special_cols]
            
            if numeric_cols:
                print(f"开始标准化 {len(numeric_cols)} 个数值列...")
                self.scaler = StandardScaler()
                X_train_numeric = X_train[numeric_cols].copy()
                X_train_numeric = self.scaler.fit_transform(X_train_numeric)
                
                # 将标准化后的数据放回DataFrame
                for i, col in enumerate(numeric_cols):
                    X_train[col] = X_train_numeric[:, i]
                print("标准化完成")
        
        preprocess_time = time.time() - preprocess_start
        print(f"预处理完成，耗时: {preprocess_time:.2f}秒")
        
        # 保存训练数据，用于计算相似度
        # 为了节省空间，只保存需要的列
        save_cols = [col for col in X_train.columns 
                    if col in [self.type_col, self.area_col, self.lat_col, self.lon_col, 
                              self.x_col, self.y_col, self.locality_id_col]]
        self.X_train_data = X_train[save_cols].copy()
        
        # 保存训练标签，用于后续预测
        self.y_train = y_train.copy()
        
        # 根据设置决定是否预计算相似度矩阵
        if self.precompute_matrix:
            print("开始计算相似度矩阵...")
            sim_start = time.time()
            similarity_matrix = self.calculate_similarity_matrix(X_train)
            sim_time = time.time() - sim_start
            print(f"相似度矩阵计算完成，耗时: {sim_time:.2f}秒")
            
            # 使用自定义距离矩阵构建KNN模型
            print(f"构建KNN模型（使用预计算矩阵），邻居数: {self.model_params['n_neighbors']}...")
            model_start = time.time()
            self.model = KNeighborsRegressor(
                n_neighbors=self.model_params['n_neighbors'],
                weights=self.model_params['weights'],
                algorithm='brute',  # 使用暴力算法与自定义距离
                leaf_size=self.model_params['leaf_size'],
                n_jobs=self.model_params['n_jobs'],
                metric='precomputed'  # 使用预计算的距离矩阵
            )
            
            # 使用距离矩阵训练模型
            print("正在拟合模型...")
            self.model.fit(similarity_matrix, y_train)
            model_time = time.time() - model_start
            print(f"模型拟合完成，耗时: {model_time:.2f}秒")
        else:
            # 使用自定义距离函数构建KNN模型
            print(f"构建KNN模型（使用动态计算相似度），邻居数: {self.model_params['n_neighbors']}...")
            model_start = time.time()
            
            # 创建索引特征，用于在自定义距离函数中找回原始数据
            X_train_idx = pd.DataFrame({
                'index': np.arange(len(X_train))
            })
            
            self.model = KNeighborsRegressor(
                n_neighbors=self.model_params['n_neighbors'],
                weights=self.model_params['weights'],
                algorithm='brute',  # 动态计算必须使用暴力算法
                leaf_size=self.model_params['leaf_size'],
                n_jobs=1,  # 自定义度量不支持并行
                metric=self.calculate_property_similarity  # 直接使用属性相似度函数
            )
            
            # 使用索引训练模型
            print("正在拟合模型...")
            self.model.fit(X_train_idx, y_train)
            model_time = time.time() - model_start
            print(f"模型拟合完成，耗时: {model_time:.2f}秒")
        
        # 保存训练数据，用于后续预测
        self._metadata["train_data"] = self.X_train_data.to_dict('records')
        
        # 更新元数据
        self._metadata["training_params"] = kwargs
        
        total_time = time.time() - start_time
        print(f"模型训练完成，总耗时: {total_time:.2f}秒")
        print(f"- 预处理: {preprocess_time:.2f}秒 ({preprocess_time/total_time:.1%})")
        if self.precompute_matrix:
            print(f"- 相似度矩阵: {sim_time:.2f}秒 ({sim_time/total_time:.1%})")
        print(f"- 模型拟合: {model_time:.2f}秒 ({model_time/total_time:.1%})")
    
    def predict(self, X: pd.DataFrame, y_true: pd.Series = None) -> np.ndarray:
        """
        使用PropertySimilarityKNN模型进行预测
        
        Args:
            X: 预测数据特征
            y_true: 可选的真实标签，用于计算误差
            
        Returns:
            预测结果数组
        """
        start_time = time.time()
        print(f"开始预测，数据集大小: {len(X)}行 x {len(X.columns)}列")
        
        # 先调用父类的predict方法进行检查
        super(KNNModel, self).predict(X)
        
        # 验证必要的列
        required_cols = [self.area_col, self.type_col]
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"预测数据中缺少必要的列: {missing_cols}")
        
        # 检查地理位置信息
        has_latlon = all(col in X.columns for col in [self.lat_col, self.lon_col])
        has_xy = all(col in X.columns for col in [self.x_col, self.y_col])
        
        if not (has_latlon or has_xy):
            raise ValueError("预测数据中缺少地理位置信息，需要提供经纬度或坐标")
        
        print("数据验证通过，开始预处理...")
        preprocess_start = time.time()
        
        # 检查真实值是否可用
        has_true_values = y_true is not None
        if has_true_values:
            print(f"检测到真实值，将计算和显示每个样本的预测误差（共{len(y_true)}个样本）")
            print(f"y_true的索引范围: {min(y_true.index)}-{max(y_true.index)}")
            # 打印表头
            print("\n" + "="*75)
            print(f"|| {'样本索引':<10} | {'真实值':<10} | {'预测值':<10} | {'绝对误差':<10} | {'误差百分比':<15} | {'有效邻居':<10} ||")
            print("=" * 75)
        else:
            print(f"警告：未提供真实值(y_true)，将无法计算误差统计")
        
        # 数据预处理和标准化
        if self.standardize and self.scaler:
            # 只标准化数值列，不包括分类特征
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # 排除要特殊处理的列
            special_cols = [self.type_col, self.locality_id_col]
            numeric_cols = [col for col in numeric_cols if col not in special_cols]
            
            if numeric_cols:
                print(f"开始标准化 {len(numeric_cols)} 个数值列...")
                X_numeric = X[numeric_cols].copy()
                X_numeric = self.scaler.transform(X_numeric)
                
                # 将标准化后的数据放回DataFrame
                for i, col in enumerate(numeric_cols):
                    X[col] = X_numeric[:, i]
                print("标准化完成")
        
        preprocess_time = time.time() - preprocess_start
        print(f"预处理完成，耗时: {preprocess_time:.2f}秒")
        
        # 从元数据中获取训练数据
        print("获取训练数据...")
        if self.X_train_data is None:
            train_data = self._metadata.get("train_data", [])
            if not train_data:
                raise ValueError("模型尚未训练或训练数据未保存")
            self.X_train_data = pd.DataFrame(train_data)
        
        print(f"已加载训练数据，大小: {len(self.X_train_data)}行 x {len(self.X_train_data.columns)}列")
        
        # 累计误差统计用的数组
        if has_true_values:
            all_abs_errors = []
            all_percent_errors = []
        
        # 根据设置决定如何进行预测
        if self.precompute_matrix:
            # 使用预计算矩阵进行预测
            # ...代码省略...
            
            # 如果提供了真实值，计算并打印整体误差统计
            if has_true_values:
                self._calculate_and_print_errors(predictions, y_true, None)
        else:
            # 为每条预测数据动态计算与训练数据的相似度
            print("使用动态计算相似度进行预测...")
            predict_start = time.time()
            
            # 只保存需要的列，用于计算相似度
            save_cols = [col for col in X.columns 
                       if col in [self.type_col, self.area_col, self.lat_col, self.lon_col, 
                                 self.x_col, self.y_col, self.locality_id_col]]
            X_pred = X[save_cols].copy()
            
            # 执行预测 - 优化版本，避免连接数据
            predictions = []
            # 对每个样本保存详细预测结果
            detailed_results = []
            
            # 统计变量
            warning_count = 0  # 警告计数
            warning_samples = []  # 收集需要警告的样本
            avg_valid_neighbors = []  # 收集有效邻居数量
            sample_batch_count = 0  # 样本批次计数
            
            k = self.model_params['n_neighbors']  # 邻居数量
            batch_size = 10  # 每批次样本数
            
            print(f"开始为{len(X_pred)}个样本预测...")

            # 不使用tqdm，改用普通的循环，确保每个样本的输出都能被看到
            total_samples = len(X_pred)
            for i, row in X_pred.iterrows():
                # 打印进度信息
                sample_batch_count += 1
                if sample_batch_count % 10 == 0 or sample_batch_count == 1:
                    elapsed = time.time() - predict_start
                    progress = sample_batch_count / total_samples
                    remaining = (elapsed / progress - elapsed) if progress > 0 else 0
                    print(f"\n>>> 预测进度: {sample_batch_count}/{total_samples} ({progress:.1%}), "
                          f"已用时间: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s")
                
                # 提取当前预测样本的特征
                row_data = row.to_dict()
                
                # 计算与所有训练样本的相似度
                similarities = []
                for j, train_row in self.X_train_data.iterrows():
                    sim = self.calculate_property_similarity(row_data, train_row)
                    similarities.append((sim, j))
                
                # 排序并找出最近的k个邻居
                similarities.sort()  # 按相似度升序排序
                k_nearest = similarities[:k]
                
                # 获取k个邻居的标签
                neighbor_indices = [int(idx) for _, idx in k_nearest]
                
                # 确保索引在有效范围内
                valid_indices = [idx for idx in neighbor_indices if 0 <= idx < len(self.y_train)]
                
                # 收集统计信息
                avg_valid_neighbors.append(len(valid_indices))
                if len(valid_indices) != k:
                    warning_count += 1
                    warning_samples.append((i, len(valid_indices)))
                    
                if len(valid_indices) == 0:
                    # 如果没有有效邻居，使用训练数据的平均值
                    pred = self.y_train.mean()
                    predictions.append(pred)
                    
                    # 记录详细结果
                    sample_result = {
                        "index": i,
                        "predicted": pred,
                        "similar_props": 0,
                        "nearest_neighbors": []
                    }
                    detailed_results.append(sample_result)
                    
                    # 如果有真实值，立即显示该样本的预测结果
                    if has_true_values:
                        # 检查索引是否存在于y_true中
                        if i in y_true.index:
                            true_val = y_true.loc[i]
                            abs_error = abs(true_val - pred)
                            percent_error = (abs_error / true_val) * 100
                            
                            # 累计误差
                            all_abs_errors.append(abs_error)
                            all_percent_errors.append(percent_error)
                            
                            # 打印该样本的预测结果（确保能看到输出）
                            print(f"|| {i:<10} | {true_val:<10.2f} | {pred:<10.2f} | {abs_error:<10.2f} | {percent_error:<15.2f}% | {0:<10}/{k} ||")
                        else:
                            print(f"|| {i:<10} | {'无真实值':<10} | {pred:<10.2f} | {'N/A':<10} | {'N/A':<15} | {0:<10}/{k} ||")
                    
                    # 每10个样本显示一次批次统计 (邻居情况)
                    if sample_batch_count % batch_size == 0:
                        self._print_batch_statistics(sample_batch_count, warning_count, warning_samples, avg_valid_neighbors, k)
                        
                        # 重置批次统计变量
                        warning_count = 0
                        warning_samples = []
                        avg_valid_neighbors = []
                        
                        # 显示当前累计的误差统计
                        if has_true_values and len(all_abs_errors) > 0:
                            self._print_current_error_statistics(all_abs_errors, all_percent_errors, sample_batch_count)
                    
                    continue
                
                # 获取邻居的标签值
                neighbor_labels = [self.y_train.iloc[idx] for idx in valid_indices]
                
                # 保存最近邻居的相似性和标签，用于详细输出
                nearest_neighbors = []
                for (sim, idx) in k_nearest:
                    if 0 <= idx < len(self.y_train):
                        nearest_neighbors.append({
                            "similarity": sim,
                            "value": float(self.y_train.iloc[idx]),
                            "index": int(idx)
                        })
                
                # 根据权重计算预测值
                if self.model_params['weights'] == 'uniform':
                    # 平均
                    pred = np.mean(neighbor_labels)
                else:
                    # 距离加权
                    valid_k_nearest = [(sim, idx) for sim, idx in k_nearest if idx in valid_indices]
                    weights = [1/(sim+1e-5) for sim, _ in valid_k_nearest]
                    pred = np.average(neighbor_labels, weights=weights)
                
                predictions.append(pred)
                
                # 记录详细结果
                sample_result = {
                    "index": i,
                    "predicted": pred,
                    "similar_props": len(valid_indices),
                    "nearest_neighbors": nearest_neighbors
                }
                detailed_results.append(sample_result)
                
                # 如果有真实值，立即显示该样本的预测结果
                if has_true_values:
                    # 检查索引是否存在于y_true中
                    if i in y_true.index:
                        true_val = y_true.loc[i]
                        abs_error = abs(true_val - pred)
                        percent_error = (abs_error / true_val) * 100
                        
                        # 累计误差
                        all_abs_errors.append(abs_error)
                        all_percent_errors.append(percent_error)
                        
                        # 打印该样本的预测结果（确保能看到输出）
                        print(f"|| {i:<10} | {true_val:<10.2f} | {pred:<10.2f} | {abs_error:<10.2f} | {percent_error:<15.2f}% | {len(valid_indices):<10}/{k} ||")
                    else:
                        print(f"|| {i:<10} | {'无真实值':<10} | {pred:<10.2f} | {'N/A':<10} | {'N/A':<15} | {len(valid_indices):<10}/{k} ||")
                
                # 每10个样本显示一次批次统计 (邻居情况)
                if sample_batch_count % batch_size == 0:
                    self._print_batch_statistics(sample_batch_count, warning_count, warning_samples, avg_valid_neighbors, k)
                    
                    # 重置批次统计变量
                    warning_count = 0
                    warning_samples = []
                    avg_valid_neighbors = []
                    
                    # 显示当前累计的误差统计
                    if has_true_values and len(all_abs_errors) > 0:
                        self._print_current_error_statistics(all_abs_errors, all_percent_errors, sample_batch_count)
            
            # 打印最后一批次的统计信息（如果有）
            if warning_count > 0 or len(avg_valid_neighbors) > 0:
                self._print_batch_statistics(sample_batch_count, warning_count, warning_samples, avg_valid_neighbors, k)
            
            # 打印最终的表尾
            if has_true_values:
                print("=" * 75)
            
            predictions = np.array(predictions)
            predict_time = time.time() - predict_start
            print(f"预测完成，耗时: {predict_time:.2f}秒")
        
        total_time = time.time() - start_time
        print(f"总预测过程完成，耗时: {total_time:.2f}秒")
        print(f"- 预处理: {preprocess_time:.2f}秒 ({preprocess_time/total_time:.1%})")
        if self.precompute_matrix:
            print(f"- 相似度矩阵: {sim_time:.2f}秒 ({sim_time/total_time:.1%})")
        print(f"- 模型预测: {predict_time:.2f}秒 ({predict_time/total_time:.1%})")
        
        # 如果提供了真实值，计算并打印最终的整体误差统计
        if has_true_values:
            print("\n===== 最终整体误差统计 =====")
            self._calculate_and_print_errors(predictions, y_true, detailed_results if not self.precompute_matrix else None)
        
        return predictions
    
    def _print_batch_statistics(self, batch_end: int, warning_count: int, warning_samples: list, 
                               avg_valid_neighbors: list, k: int) -> None:
        """
        打印批次统计信息
        
        Args:
            batch_end: 当前批次的结束索引
            warning_count: 警告数量
            warning_samples: 警告样本列表
            avg_valid_neighbors: 有效邻居数量列表
            k: 应有的邻居数量
        """
        start_idx = max(0, batch_end - len(avg_valid_neighbors))
        batch_range = f"{start_idx+1}-{batch_end}"
        
        if not avg_valid_neighbors:
            return
            
        avg_neighbors = sum(avg_valid_neighbors) / len(avg_valid_neighbors)
        
        print(f"\n--- 样本 {batch_range} 批次统计 ---")
        print(f"平均有效邻居数: {avg_neighbors:.2f}/{k}")
        
        if warning_count > 0:
            print(f"有 {warning_count}/{len(avg_valid_neighbors)} 个样本邻居数不足")
            print("邻居不足的样本:")
            for sample_idx, valid_count in warning_samples[:5]:  # 只显示前5个
                print(f"  样本 {sample_idx}: {valid_count}/{k} 个有效邻居")
            
            if len(warning_samples) > 5:
                print(f"  ... 还有 {len(warning_samples) - 5} 个邻居不足的样本未显示")
        else:
            print("所有样本都有足够的邻居")
        print("----------------------------\n")
    
    def _calculate_and_print_errors(self, predictions: np.ndarray, y_true: pd.Series, detailed_results: list = None) -> None:
        """
        计算并打印预测误差统计
        
        Args:
            predictions: 预测值数组
            y_true: 真实值序列
            detailed_results: 可选的详细预测结果列表
        """
        # 确保长度匹配
        if len(predictions) != len(y_true):
            print(f"警告: 预测值数量 ({len(predictions)}) 与真实值数量 ({len(y_true)}) 不匹配")
            return
        
        # 计算误差
        abs_errors = np.abs(y_true.values - predictions)
        percent_errors = (abs_errors / y_true.values) * 100
        
        # 统计指标
        mean_abs_error = np.mean(abs_errors)
        median_abs_error = np.median(abs_errors)
        mean_percent_error = np.mean(percent_errors)
        median_percent_error = np.median(percent_errors)
        
        # 误差分布
        p10 = np.percentile(percent_errors, 10)
        p25 = np.percentile(percent_errors, 25)
        p50 = np.percentile(percent_errors, 50)
        p75 = np.percentile(percent_errors, 75)
        p90 = np.percentile(percent_errors, 90)
        
        # 误差范围统计
        error_ranges = {
            "<5%": np.mean(percent_errors < 5) * 100,
            "5-10%": np.mean((percent_errors >= 5) & (percent_errors < 10)) * 100,
            "10-15%": np.mean((percent_errors >= 10) & (percent_errors < 15)) * 100,
            "15-20%": np.mean((percent_errors >= 15) & (percent_errors < 20)) * 100,
            ">20%": np.mean(percent_errors >= 20) * 100
        }
        
        # 打印统计结果
        print("\n===== 预测误差统计 =====")
        print(f"样本数量: {len(predictions)}")
        print(f"平均绝对误差 (MAE): {mean_abs_error:.2f}")
        print(f"中位绝对误差: {median_abs_error:.2f}")
        print(f"平均相对误差: {mean_percent_error:.2f}%")
        print(f"中位相对误差: {median_percent_error:.2f}%")
        
        print("\n误差百分比分布:")
        print(f"10%分位数: {p10:.2f}%")
        print(f"25%分位数: {p25:.2f}%")
        print(f"50%分位数 (中位数): {p50:.2f}%")
        print(f"75%分位数: {p75:.2f}%")
        print(f"90%分位数: {p90:.2f}%")
        
        print("\n误差范围分布:")
        for range_name, percentage in error_ranges.items():
            print(f"{range_name}: {percentage:.2f}%")
        
        # 打印每个样本的详细误差，但为了避免输出过多，只打印前10个及误差最大的10个
        if detailed_results is not None:
            # 添加误差信息到详细结果
            for i, (pred, true) in enumerate(zip(predictions, y_true)):
                if i < len(detailed_results):
                    detailed_results[i]["true"] = float(true)
                    detailed_results[i]["abs_error"] = float(abs_errors[i])
                    detailed_results[i]["percent_error"] = float(percent_errors[i])
            
            # 按误差百分比排序
            sorted_results = sorted(detailed_results, key=lambda x: x.get("percent_error", 0), reverse=True)
            
            # 打印误差最大的10个样本
            print("\n===== 误差最大的10个样本 =====")
            for i, result in enumerate(sorted_results[:10]):
                print(f"样本 {result['index']}: 预测值={result['predicted']:.2f}, 真实值={result['true']:.2f}, "
                      f"误差={result['abs_error']:.2f}, 误差百分比={result['percent_error']:.2f}%")
                
                # 打印最近邻居信息
                if len(result.get("nearest_neighbors", [])) > 0:
                    print("  最近邻居:")
                    for j, neighbor in enumerate(result["nearest_neighbors"][:3]):  # 只打印前3个邻居
                        print(f"  - 邻居{j+1}: 相似度={neighbor['similarity']:.4f}, 值={neighbor['value']:.2f}")
            
            # 打印前10个样本的详细信息
            print("\n===== 前10个样本预测详情 =====")
            for i, result in enumerate(detailed_results[:10]):
                if "true" in result:
                    print(f"样本 {result['index']}: 预测值={result['predicted']:.2f}, 真实值={result['true']:.2f}, "
                          f"误差={result['abs_error']:.2f}, 误差百分比={result['percent_error']:.2f}%")
                    
                    # 打印最近邻居信息
                    if len(result.get("nearest_neighbors", [])) > 0:
                        print("  最近邻居:")
                        for j, neighbor in enumerate(result["nearest_neighbors"][:3]):  # 只打印前3个邻居
                            print(f"  - 邻居{j+1}: 相似度={neighbor['similarity']:.4f}, 值={neighbor['value']:.2f}")
                else:
                    print(f"样本 {result['index']}: 预测值={result['predicted']:.2f}, 无真实值")
    
    def _print_current_error_statistics(self, abs_errors: list, percent_errors: list, sample_count: int) -> None:
        """
        打印当前累计的误差统计
        
        Args:
            abs_errors: 绝对误差列表
            percent_errors: 百分比误差列表
            sample_count: 当前样本数
        """
        if not abs_errors:
            return
            
        # 转换为numpy数组便于计算
        abs_errors_array = np.array(abs_errors)
        percent_errors_array = np.array(percent_errors)
        
        # 统计指标
        mean_abs_error = np.mean(abs_errors_array)
        median_abs_error = np.median(abs_errors_array)
        mean_percent_error = np.mean(percent_errors_array)
        median_percent_error = np.median(percent_errors_array)
        
        print(f"\n--- 累计误差统计 (前{len(abs_errors)}个有效样本) ---")
        print(f"平均绝对误差: {mean_abs_error:.2f}, 中位绝对误差: {median_abs_error:.2f}")
        print(f"平均相对误差: {mean_percent_error:.2f}%, 中位相对误差: {median_percent_error:.2f}%")
        print("-" * 65)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        对于PropertySimilarityKNN模型，返回基于权重的特征重要性
        
        Returns:
            特征重要性数据框，包含feature和importance两列
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法获取特征重要性")
        
        # 基于权重计算特征重要性
        importance_list = [
            {"feature": "地理距离", "importance": self.distance_weight},
            {"feature": "面积差异", "importance": self.area_weight}
        ]
        
        # 如果使用了地区ID，添加地区特征
        if self.locality_id_col in self._feature_names:
            # 调整权重以保证总和为1
            total_weight = self.distance_weight + self.area_weight
            locality_importance = 0.1
            scale_factor = (1 - locality_importance) / total_weight
            
            importance_list = [
                {"feature": "地理距离", "importance": self.distance_weight * scale_factor},
                {"feature": "面积差异", "importance": self.area_weight * scale_factor},
                {"feature": "地区", "importance": locality_importance}
            ]
        
        # 将列表转换为DataFrame
        importance_df = pd.DataFrame(importance_list)
        
        return importance_df 