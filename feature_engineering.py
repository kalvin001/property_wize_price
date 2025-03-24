import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats
import argparse
from pathlib import Path

def load_data(data_file):
    """加载原始数据"""
    print(f"正在加载数据: {data_file}")
    df = pd.read_csv(data_file)
    print(f"数据加载完成，共有 {df.shape[0]} 行 {df.shape[1]} 列")
    return df

def handle_missing_values(df):
    """处理缺失值"""
    print("处理缺失值...")
    
    # 统计每列的缺失值比例
    missing_ratio = df.isna().mean().sort_values(ascending=False)
    high_missing_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
    
    print(f"缺失值比例超过50%的列有 {len(high_missing_cols)} 个，将被删除")
    df = df.drop(columns=high_missing_cols)
    
    # 对于数值列，使用中位数填充缺失值
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # 对于分类列，使用众数填充缺失值
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df
def create_new_features(df):
    """创建新特征，避免使用目标变量y_label"""
    print("创建新特征...")
    
    # 移除所有使用y_label的特征派生
    
    # 1. 房产特征比率
    if 'internal_area' in df.columns and 'land_size' in df.columns:
        df['building_coverage_ratio'] = df['internal_area'] / df['land_size'].replace(0, np.nan)
        # 添加对数变换
        df['log_internal_area'] = np.log1p(df['internal_area'])
        df['log_land_size'] = np.log1p(df['land_size'])
        df['log_building_coverage_ratio'] = df['log_internal_area'] - df['log_land_size']
    
    if 'prop_bed' in df.columns and 'prop_bath' in df.columns:
        df['bed_bath_ratio'] = df['prop_bed'] / df['prop_bath'].replace(0, np.nan)
    
    # 2. 房间总数特征
    room_cols = ['prop_bed', 'prop_bath', 'prop_carpark']
    if all(col in df.columns for col in room_cols):
        df['total_rooms'] = df[room_cols].sum(axis=1)
        # 添加房间组合特征
        df['bed_bath_product'] = df['prop_bed'] * df['prop_bath']
        df['rooms_squared'] = df['total_rooms'] ** 2
    
    # 3. 综合质量得分
    quality_cols = ['luxury_level_score', 'natural_light', 'quietness', 'view_quality']
    if all(col in df.columns for col in quality_cols):
        df['quality_score'] = df[quality_cols].mean(axis=1)
        # 添加质量指数的平方和立方特征
        df['quality_score_squared'] = df['quality_score'] ** 2
        df['quality_score_cubed'] = df['quality_score'] ** 3
    
    # 4. 构建年龄相关特征
    if 'construction_year_guess' in df.columns:
        current_year = 2023  # 当前年份
        df['building_age_years'] = current_year - df['construction_year_guess']
        
        # 创建年龄段分类
        df['age_category'] = pd.cut(
            df['building_age_years'], 
            bins=[-1, 5, 10, 20, 30, 50, 100, 200], 
            labels=['新建', '近新', '较新', '中等', '较旧', '旧', '古老']
        )
        
        # 添加建筑年龄的非线性变换
        df['log_building_age'] = np.log1p(df['building_age_years'])
        df['building_age_squared'] = df['building_age_years'] ** 2
        
        # 增加建筑年龄与面积的交互特征
        if 'internal_area' in df.columns:
            df['age_area_interaction'] = df['building_age_years'] * df['internal_area']
    
    # 5. 位置便利性指数
    location_cols = ['near_park', 'corner_property']
    if all(col in df.columns for col in location_cols):
        df['location_convenience'] = df[location_cols].sum(axis=1)
    
    # 6. 功能空间特征
    function_cols = ['study_home_office_cnt', 'entertaining_space_cnt', 'living_space_cnt']
    if all(col in df.columns for col in function_cols):
        df['functional_spaces'] = df[function_cols].sum(axis=1)
        # 添加功能空间的平方特征
        df['functional_spaces_squared'] = df['functional_spaces'] ** 2
    
    # 7. 特殊设施指数
    facility_cols = ['indoor_swimming_pool_cnt', 'outdoor_swimming_pool_cnt', 
                    'tennis_court_cnt', 'gym_cnt', 'indoor_spa_cnt', 'outdoor_spa_cnt']
    if all(col in df.columns for col in facility_cols):
        df['luxury_facilities'] = df[facility_cols].sum(axis=1)
        # 增加高级设施的二次指标
        df['luxury_facilities_squared'] = df['luxury_facilities'] ** 2
        
        # 增加高级设施密度指标
        if 'internal_area' in df.columns:
            df['luxury_facility_density'] = df['luxury_facilities'] / df['internal_area'].replace(0, np.nan)
    
    # 8. 户型结构比例
    if 'prop_bed' in df.columns and 'total_rooms' in df.columns:
        df['bedroom_ratio'] = df['prop_bed'] / df['total_rooms'].replace(0, np.nan)
    
    # 9. 高级区域统计特征
    if 'locality_post' in df.columns and 'land_value' in df.columns:
        # 使用邮编区域的地价统计特征
        locality_land_stats = df.groupby('locality_post')['land_value'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
        locality_land_stats.columns = ['locality_post', 'locality_land_mean', 'locality_land_median', 
                                     'locality_land_std', 'locality_land_min', 'locality_land_max']
        df = df.merge(locality_land_stats, on='locality_post', how='left')
        
        # 计算地价与区域统计值的比率特征
        df['land_to_locality_mean_ratio'] = df['land_value'] / df['locality_land_mean'].replace(0, np.nan)
        df['land_to_locality_max_ratio'] = df['land_value'] / df['locality_land_max'].replace(0, np.nan)
        
        # 计算地价在区域中的离差
        df['land_value_locality_deviation'] = (df['land_value'] - df['locality_land_mean']) / df['locality_land_std'].replace(0, np.nan)
    
    # 10. 区域复合指数
    if 'sector_median' in df.columns and 'internal_area' in df.columns:
        df['sector_value_per_sqm'] = df['sector_median'] / df['internal_area'].replace(0, np.nan)
    
    # 11. 生活空间舒适度
    if 'internal_area' in df.columns and 'prop_bed' in df.columns:
        df['space_per_bedroom'] = df['internal_area'] / df['prop_bed'].replace(0, np.nan)
        # 添加非线性变换
        df['log_space_per_bedroom'] = np.log1p(df['space_per_bedroom'])
    
    # 12. 综合特性指数 - 结合多个维度
    if all(col in df.columns for col in ['quality_score', 'location_convenience', 'luxury_facilities']):
        df['property_composite_index'] = (
            df['quality_score'].fillna(0) * 0.4 + 
            df['location_convenience'].fillna(0) * 0.3 + 
            df['luxury_facilities'].fillna(0) * 0.3
        )
        # 添加非线性变换
        df['property_composite_index_squared'] = df['property_composite_index'] ** 2
    
    # 13. 娱乐空间比例
    entertainment_cols = ['biggest_entertainment_area', 'outdoor_dining_area']
    if all(col in df.columns for col in entertainment_cols) and 'internal_area' in df.columns:
        df['entertainment_space_ratio'] = df[entertainment_cols].sum(axis=1) / df['internal_area'].replace(0, np.nan)
    
    # 14. 地理坐标特征
    if 'prop_x' in df.columns and 'prop_y' in df.columns:
        # 计算与区域中心的距离
        df['distance_from_center'] = np.sqrt(df['prop_x']**2 + df['prop_y']**2)
        # 添加距离的对数变换
        df['log_distance_from_center'] = np.log1p(df['distance_from_center'])
        
        # 将经纬度转换为基数形式的角度特征
        df['sin_latitude'] = np.sin(df['prop_y'])
        df['cos_latitude'] = np.cos(df['prop_y'])
        df['sin_longitude'] = np.sin(df['prop_x'])
        df['cos_longitude'] = np.cos(df['prop_x'])
        
        # 极坐标表示
        df['property_angle'] = np.arctan2(df['prop_y'], df['prop_x'])
        
        # 添加坐标聚类特征
        from sklearn.cluster import KMeans
        n_clusters = 10  # 可根据实际数据调整
        
        # 获取有效坐标的样本
        coord_df = df[['prop_x', 'prop_y']].dropna()
        
        if len(coord_df) > n_clusters:  # 确保有足够的样本进行聚类
            # 执行KMeans聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            coord_df['geo_cluster'] = kmeans.fit_predict(coord_df)
            
            # 将聚类结果合并回原始DataFrame
            geo_clusters = coord_df['geo_cluster'].reset_index()
            df = df.merge(geo_clusters, left_index=True, right_on='index', how='left')
            df.drop(columns=['index'], inplace=True)
            
            # 获取聚类中心
            cluster_centers = kmeans.cluster_centers_
            
            # 计算到最近聚类中心的距离
            for i, center in enumerate(cluster_centers):
                center_x, center_y = center
                df[f'distance_to_cluster_{i}'] = np.sqrt((df['prop_x'] - center_x)**2 + 
                                                        (df['prop_y'] - center_y)**2)
            
            # 计算到最近聚类中心的距离
            distance_columns = [f'distance_to_cluster_{i}' for i in range(n_clusters)]
            df['min_cluster_distance'] = df[distance_columns].min(axis=1)
    
    # 15. 房屋属性组合特征
    if 'prop_bed' in df.columns and 'prop_bath' in df.columns and 'prop_carpark' in df.columns:
        # 床卫车组合得分
        df['bed_bath_car_score'] = df['prop_bed'] * 0.5 + df['prop_bath'] * 0.3 + df['prop_carpark'] * 0.2
        # 添加非线性变换
        df['bed_bath_car_score_squared'] = df['bed_bath_car_score'] ** 2
        
        # 添加房型特征编码
        df['property_type_code'] = (df['prop_bed'] * 100 + df['prop_bath'] * 10 + df['prop_carpark']).astype(int)
    
    # 16. 土地利用效率
    if 'internal_area' in df.columns and 'land_size' in df.columns:
        df['land_utilization_ratio'] = df['internal_area'] / df['land_size'].replace(0, np.nan)
        # 非线性变换
        df['land_utilization_ratio_squared'] = df['land_utilization_ratio'] ** 2
    
    # 17. 房间平均面积
    if 'internal_area' in df.columns and 'total_rooms' in df.columns:
        df['avg_room_area'] = df['internal_area'] / df['total_rooms'].replace(0, np.nan)
        # 对数变换
        df['log_avg_room_area'] = np.log1p(df['avg_room_area'])
    
    # 18. 房屋层数密度
    if 'internal_area' in df.columns and 'levels_cnt' in df.columns:
        df['area_per_level'] = df['internal_area'] / df['levels_cnt'].replace(0, np.nan)
        # 增加与建筑年龄的交互特征
        if 'building_age_years' in df.columns:
            df['age_level_interaction'] = df['building_age_years'] * df['levels_cnt']
    
    # 19. 便利设施密度
    facility_cols = ['entertaining_space_cnt', 'study_home_office_cnt', 'kitchen_cnt']
    if all(col in df.columns for col in facility_cols) and 'internal_area' in df.columns:
        facility_sum = df[facility_cols].sum(axis=1)
        df['facilities_density'] = facility_sum / df['internal_area'].replace(0, np.nan)
    
    # 20. 主要功能空间比例
    area_cols = ['mainbuilding_bedrooms_area', 'living_space_area', 'kitchen_area']
    if all(col in df.columns for col in area_cols) and 'internal_area' in df.columns:
        df['main_functional_area_ratio'] = df[area_cols].sum(axis=1) / df['internal_area'].replace(0, np.nan)
    
    # 21. 新增：高级地理位置特征
    if 'locality_post' in df.columns and 'locality_suburb' in df.columns:
        # 计算每个郊区的属性数量（作为受欢迎程度的代理指标）
        suburb_popularity = df.groupby('locality_suburb').size().reset_index(name='suburb_property_count')
        df = df.merge(suburb_popularity, on='locality_suburb', how='left')
        
        # 计算每个郊区的平均地价
        if 'land_value' in df.columns:
            suburb_land_value = df.groupby('locality_suburb')['land_value'].mean().reset_index(name='suburb_avg_land_value')
            df = df.merge(suburb_land_value, on='locality_suburb', how='left')
            
            # 计算地价与郊区平均值的比例
            df['land_to_suburb_ratio'] = df['land_value'] / df['suburb_avg_land_value'].replace(0, np.nan)
    
    # 22. 新增：特征交互和多项式特征
    important_features = ['land_value', 'internal_area', 'prop_bed', 'building_age_years']
    if all(feature in df.columns for feature in important_features):
        # 两两特征交互
        for i, feat1 in enumerate(important_features):
            for feat2 in important_features[i+1:]:
                df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
    
    # 23. 新增：价格区间特征
    if 'land_value' in df.columns:
        # 创建土地价值区间分类
        df['land_value_bracket'] = pd.qcut(df['land_value'], q=10, labels=False, duplicates='drop')
    
    # 24. 新增：季节性特征
    if 'sale_month' in df.columns:
        # 将月份转换为季节
        df['season'] = ((df['sale_month'] % 12) // 3).map({0: '冬', 1: '春', 2: '夏', 3: '秋'})
        
        # 创建月份的周期性特征
        df['month_sin'] = np.sin(2 * np.pi * df['sale_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['sale_month'] / 12)
    
    # 25. 新增：豪华度特征的多元组合
    luxury_features = ['luxury_level_score', 'luxury_facilities', 'quality_score']
    if all(feature in df.columns for feature in luxury_features):
        # 计算豪华特征的加权几何平均数
        df['luxury_composite'] = (
            (df['luxury_level_score'] + 1) ** 0.5 * 
            (df['luxury_facilities'] + 1) ** 0.3 * 
            (df['quality_score'] + 1) ** 0.2
        ) - 1
    
    return df
def encode_categorical_features(df):
    """编码分类特征"""
    print("编码分类特征...")
    
    # 得到所有分类特征
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 移除不需要编码的列：如地址、ID等
    exclude_cols = ['std_address', 'house_name', 'full_address_y']
    cat_cols = [col for col in cat_cols if col not in exclude_cols]
    
    print(f"需要编码的分类特征有 {len(cat_cols)} 个")
    
    # 对每个分类特征进行独热编码
    for col in cat_cols:
        if df[col].nunique() < 10:  # 只对基数较低的特征进行独热编码
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
        else:
            # 对高基数特征进行标签编码
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
            df.drop(columns=[col], inplace=True)
    
    return df

def engineer_features(data_file, output_file, skip_outlier=False):
    """特征工程主函数"""
    # 加载数据
    df = load_data(data_file)
    
    # 备份不需处理的列
    id_cols = ['prop_id']
    target_col = 'y_label'
    
    # 处理缺失值
    df = handle_missing_values(df)
    
    # 创建新特征
    df = create_new_features(df)
    
    # #处理异常值
    # if not skip_outlier:
    #     df = handle_outliers(df, target_col=target_col)
    
    # 编码分类特征
    df = encode_categorical_features(df)
    
    # 移除低方差特征
    low_var_threshold = 0.01
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # 排除ID列和目标列
    numeric_cols = [col for col in numeric_cols if col not in id_cols + [target_col]]
    
    for col in numeric_cols:
        if df[col].var() < low_var_threshold:
            df = df.drop(columns=[col])
            print(f"移除低方差特征: {col}")
    
    # 保存结果
    print(f"保存结果到: {output_file}")
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_file, index=False)
    print(f"特征工程完成! 处理后的数据有 {df.shape[0]} 行 {df.shape[1]} 列")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='对房价数据进行特征工程')
    parser.add_argument('--data_file', type=str, default='resources/house_samples_features.csv',
                       help='原始数据文件路径')
    parser.add_argument('--output_file', type=str, default='resources/house_samples_engineered.csv',
                       help='处理后数据文件保存路径')
    parser.add_argument('--skip_outlier', action='store_true',
                       help='是否跳过异常值处理')
    args = parser.parse_args()
    
    # 执行特征工程
    engineer_features(args.data_file, args.output_file, args.skip_outlier)

if __name__ == "__main__":
    main() 