import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ModelFactory, KNNModel, GeographicKNNModel, WeightedKNNModel, PropertySimilarityKNNModel

def load_sample_data(file_path='data/property_sample.csv'):
    """
    加载房产样本数据
    """
    try:
        # 尝试加载数据
        df = pd.read_csv(file_path)
        print(f"加载了 {len(df)} 条房产数据")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        
        # 如果文件不存在，创建一个简单的测试数据集
        print("创建测试数据集...")
        np.random.seed(42)
        n_samples = 1000
        
        # 创建模拟房产数据
        data = {
            'prop_id': [f'P{i:05d}' for i in range(n_samples)],
            'prop_type': np.random.choice(['apartment', 'house', 'townhouse', 'villa'], n_samples),
            'internal_area': np.random.uniform(50, 300, n_samples),
            'latitude': np.random.uniform(-37.85, -37.75, n_samples),
            'longitude': np.random.uniform(144.95, 145.05, n_samples),
            'prop_x': np.random.uniform(144.95, 145.05, n_samples),
            'prop_y': np.random.uniform(-37.85, -37.75, n_samples),
            'age': np.random.uniform(0, 50, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'locality_id': np.random.randint(1, 21, n_samples),
            'std_address': [f'{i} Test Street, Melbourne' for i in range(n_samples)]
        }
        
        # 基于特征生成价格（模拟真实世界的关系）
        prices = []
        for i in range(n_samples):
            # 基础价格基于面积、房间数和地理位置
            base_price = data['internal_area'][i] * 5000
            
            # 根据房产类型调整
            type_factor = {
                'apartment': 0.9,
                'house': 1.0,
                'townhouse': 1.1,
                'villa': 1.2
            }.get(data['prop_type'][i], 1.0)
            
            # 根据房间数量调整
            room_factor = 1.0 + (data['bedrooms'][i] * 0.05) + (data['bathrooms'][i] * 0.03)
            
            # 根据年龄调整（较新的房产更贵）
            age_factor = 1.0 - (data['age'][i] * 0.005)
            
            # 根据位置调整（模拟某些区域更贵）
            loc_factor = 1.0 + ((data['locality_id'][i] % 5) * 0.1)
            
            # 综合价格因素
            price = base_price * type_factor * room_factor * age_factor * loc_factor
            
            # 添加一些随机噪声（±10%）
            price *= np.random.uniform(0.9, 1.1)
            
            prices.append(price)
        
        data['price'] = prices
        data['y_label'] = prices  # 标签列
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存到文件
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"测试数据已保存到 {file_path}")
        
        return df

def prepare_data(df):
    """
    准备训练和测试数据
    """
    # 确保必要的列存在
    required_cols = ['prop_type', 'internal_area', 'prop_x', 'prop_y', 'latitude', 'longitude']
    
    for col in required_cols:
        if col not in df.columns:
            print(f"警告: 数据中缺少列 {col}，将使用随机值")
            if col in ['prop_x', 'prop_y', 'latitude', 'longitude']:
                df[col] = np.random.uniform(0, 100, len(df))
            elif col == 'internal_area':
                df[col] = np.random.uniform(50, 300, len(df))
            elif col == 'prop_type':
                df[col] = np.random.choice(['apartment', 'house', 'townhouse'], len(df))
    
    # 确保有标签列
    if 'price' not in df.columns and 'y_label' not in df.columns:
        print("警告: 数据中缺少价格列，将使用随机值")
        df['price'] = df['internal_area'] * np.random.uniform(4000, 6000, len(df))
    
    label_col = 'y_label' if 'y_label' in df.columns else 'price'
    
    # 选择特征列
    feature_cols = [col for col in df.columns if col not in [label_col, 'prop_id', 'std_address']]
    
    # 分割数据
    X = df[feature_cols]
    y = df[label_col]
    
    # 处理分类特征
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col] = pd.Categorical(X[col]).codes
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_cols

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 计算MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape
    }

def plot_results(metrics, model_types):
    """
    可视化不同模型的性能比较
    """
    metrics_df = pd.DataFrame(metrics)
    metrics_df['model'] = model_types
    metrics_df = metrics_df.set_index('model')
    
    # 绘制RMSE比较图
    plt.figure(figsize=(10, 6))
    metrics_df['rmse'].plot(kind='bar', color='skyblue')
    plt.title('各模型的RMSE比较')
    plt.ylabel('RMSE值 (越低越好)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/knn_rmse_comparison.png')
    
    # 绘制R²比较图
    plt.figure(figsize=(10, 6))
    metrics_df['r2'].plot(kind='bar', color='lightgreen')
    plt.title('各模型的R²比较')
    plt.ylabel('R²值 (越高越好)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/knn_r2_comparison.png')
    
    # 绘制MAPE比较图
    plt.figure(figsize=(10, 6))
    metrics_df['mape'].plot(kind='bar', color='salmon')
    plt.title('各模型的MAPE比较')
    plt.ylabel('MAPE值 (%) (越低越好)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/knn_mape_comparison.png')
    
    print("图表已保存到results目录")
    return metrics_df

def find_similar_properties(model, prop_id, X, all_data, n=5):
    """
    使用模型找到最相似的房产
    """
    if prop_id not in all_data['prop_id'].values:
        print(f"未找到ID为 {prop_id} 的房产")
        return []
    
    # 获取目标房产数据
    target_data = all_data[all_data['prop_id'] == prop_id]
    target_idx = target_data.index[0]
    
    # 如果是PropertySimilarityKNN模型，可以直接使用其内部计算方法
    if isinstance(model, PropertySimilarityKNNModel):
        # 提取当前行
        row = target_data.iloc[0]
        
        # 使用模型中的方法查找相似房产
        similar_props = model.calculate_similarity_matrix(all_data)
        
        # 获取与目标房产的相似度
        similarities = similar_props[target_idx]
        
        # 排除自身
        mask = np.ones(len(similarities), dtype=bool)
        mask[target_idx] = False
        filtered_similarities = similarities[mask]
        filtered_indices = np.arange(len(similarities))[mask]
        
        # 获取最相似的n个房产（相似度值最小的）
        top_n_idx = filtered_indices[np.argsort(filtered_similarities)[:n]]
        
        # 构建结果列表
        similar_properties = []
        for idx in top_n_idx:
            prop = all_data.iloc[idx]
            similarity_score = 100 - min(100, filtered_similarities[np.where(filtered_indices == idx)[0][0]])
            similar_properties.append({
                "id": prop['prop_id'],
                "address": prop.get('std_address', f"房产 {prop['prop_id']}"),
                "area": float(prop.get('internal_area', 0)),
                "price": float(prop.get('price', prop.get('y_label', 0))),
                "type": str(prop.get('prop_type', '')),
                "similarity": max(0, int(similarity_score)),
                "distance_km": float(filtered_similarities[np.where(filtered_indices == idx)[0][0]])
            })
    else:
        # 对于其他模型，使用一个简单的欧几里得距离计算相似度
        print("使用欧几里得距离计算相似度")
        
        # 获取目标房产的特征
        target_features = X.iloc[target_idx].values.reshape(1, -1)
        
        # 计算所有房产与目标房产的欧几里得距离
        distances = []
        for i in range(len(X)):
            if i != target_idx:  # 排除自身
                dist = np.linalg.norm(X.iloc[i].values - target_features)
                distances.append((i, dist))
        
        # 按距离排序并获取前n个
        top_n = sorted(distances, key=lambda x: x[1])[:n]
        
        # 构建结果列表
        similar_properties = []
        for idx, dist in top_n:
            prop = all_data.iloc[idx]
            # 计算相似度得分（距离越小，相似度越高）
            similarity_score = 100 - min(100, dist * 10)  # 简单映射到0-100
            similar_properties.append({
                "id": prop['prop_id'],
                "address": prop.get('std_address', f"房产 {prop['prop_id']}"),
                "area": float(prop.get('internal_area', 0)),
                "price": float(prop.get('price', prop.get('y_label', 0))),
                "type": str(prop.get('prop_type', '')),
                "similarity": max(0, int(similarity_score)),
                "distance_km": float(dist)
            })
    
    return similar_properties

def main():
    """
    主函数
    """
    print("加载房产数据...")
    df = load_sample_data()
    
    print("准备数据...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    print("训练和评估不同的KNN模型...")
    models = {
        "基础KNN": KNNModel(name="basic_knn", n_neighbors=5),
        "地理距离KNN": GeographicKNNModel(
            name="geographic_knn", 
            n_neighbors=5,
            lat_col='latitude',
            lon_col='longitude'
        ),
        "加权特征KNN": WeightedKNNModel(
            name="weighted_knn", 
            n_neighbors=5,
            feature_weights={
                'internal_area': 2.0,
                'latitude': 1.5,
                'longitude': 1.5,
                'bedrooms': 1.2,
                'bathrooms': 1.0
            }
        ),
        "房产相似度KNN": PropertySimilarityKNNModel(
            name="property_similarity_knn", 
            n_neighbors=5,
            distance_weight=0.6,
            area_weight=0.4
        )
    }
    
    # 训练和评估每个模型
    metrics_list = []
    model_types = []
    
    for model_name, model in models.items():
        print(f"\n训练 {model_name}...")
        model.train(X_train, y_train)
        
        print(f"评估 {model_name}...")
        metrics = evaluate_model(model, X_test, y_test)
        
        print(f"{model_name} 评估结果:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        metrics_list.append(metrics)
        model_types.append(model_name)
        
        # 保存模型
        model_path = os.path.join('model', f"{model.name}.joblib")
        model.save(model_path)
        print(f"模型已保存到 {model_path}")
    
    # 可视化比较结果
    metrics_df = plot_results(metrics_list, model_types)
    
    # 为每个模型查找相似房产示例
    test_prop_id = df['prop_id'].iloc[0]
    print(f"\n为房产 {test_prop_id} 查找相似房产:")
    
    for model_name, model in models.items():
        print(f"\n使用 {model_name} 查找相似房产:")
        similar_props = find_similar_properties(model, test_prop_id, X_test, df)
        
        print(f"相似房产列表 (排序: 相似度):")
        for i, prop in enumerate(similar_props):
            print(f"  {i+1}. ID: {prop['id']}, 地址: {prop['address']}, " 
                  f"面积: {prop['area']:.1f}, 价格: {prop['price']:.0f}, "
                  f"相似度: {prop['similarity']}%")

if __name__ == "__main__":
    main() 