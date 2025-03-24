import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA  # 添加PCA导入
import pickle
import base64
import torch
from transformers import BertTokenizer, BertModel
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

def text_to_bert_embeddings(texts, model_name='bert-base-uncased', max_length=128):
    """
    使用BERT模型将文本转换为向量嵌入
    
    参数:
        texts: 文本列表
        model_name: 使用的BERT模型名称
        max_length: 最大序列长度
        
    返回:
        numpy数组形式的嵌入向量
    """
    # 加载预训练的BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 将模型移动到GPU（如果可用）
    model = model.to(device)
    
    # 将模型设置为评估模式
    model.eval()
    
    # 存储所有文本的嵌入向量
    all_embeddings = []
    
    # 分批处理文本以避免内存问题
    batch_size = 128
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # 对文本进行编码
        encoded_input = tokenizer(
            batch_texts, 
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取输入IDs和注意力掩码，并移动到GPU
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        
        # 不计算梯度
        with torch.no_grad():
            # 前向传播得到模型输出
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # 使用CLS标记的最后隐藏状态作为文本的表示，并移回CPU转换为numpy
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
    
    # 将所有批次的嵌入向量连接起来
    return np.vstack(all_embeddings)

def preprocess_features(df: pd.DataFrame, 
                        categorical_cols: list = None, 
                        numeric_cols: list = None, 
                        skip_encoding_cols: list = None,
                        fill_na_numeric='median', 
                        fill_na_categorical='missing'):
    """
    通用特征预处理函数：
    - 对数值特征缺失值进行填充(默认中位数)
    - 对类别特征缺失值进行填充(默认字符串'missing')
    - 对类别特征进行Label Encoding（可通过skip_encoding_cols指定不编码的列）
    
    参数：
        df: 输入数据DataFrame
        categorical_cols: 指定类别型特征列名的列表，如果不指定则自动推断
        numeric_cols: 指定数值型特征列名的列表，如果不指定则自动推断
        skip_encoding_cols: 指定不进行编码的列名称列表（这些列若为类别型，将保持填充后的string类型）
        fill_na_numeric: 数值列缺失值填充方式，默认'median'，可选择'mean'或其他固定值
        fill_na_categorical: 类别列缺失值填充值，默认'missing'，可为'string'或'mode'等
    
    返回：
        df_processed: 处理后的DataFrame
        encoders: dict, {列名: LabelEncoder对象}，不编码的列不会出现在字典中。
    """
    
    df_processed = df.copy()
    
    # 如果未指定特征类型列，则自动推断
    if categorical_cols is None:
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    if numeric_cols is None:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # 如果未指定skip_encoding_cols，则默认为空列表
    if skip_encoding_cols is None:
        skip_encoding_cols = []
        
    # 数值列缺失值填充
    for col in numeric_cols:
        if fill_na_numeric == 'median':
            fill_value = df_processed[col].median()
        elif fill_na_numeric == 'mean':
            fill_value = df_processed[col].mean()
        else:
            fill_value = fill_na_numeric
        df_processed[col].fillna(fill_value, inplace=True)
    
    # 类别列缺失值填充
    for col in categorical_cols:
        if fill_na_categorical == 'missing':
            fill_value = 'missing'
        elif fill_na_categorical == 'mode':
            fill_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'missing'
        else:
            fill_value = fill_na_categorical
        df_processed[col] = df_processed[col].astype('object').fillna(fill_value)
    
    # 对类别列进行编码
    encoders = {}
    for col in categorical_cols:
        if col not in skip_encoding_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le
        else:
            # 不进行编码的列保持原样（已填充missing值）
            # 此时列为object类型，可以根据需要保留或转换为category类型
            df_processed[col] = df_processed[col].astype('category')
    
    return df_processed, encoders


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

def create_time_aware_neighborhood_features(df):
    """
    创建时间感知的近邻价格特征（只使用每个房屋成交日期之前的数据）
    
    参数:
        df: 输入数据DataFrame，必须包含y_label(价格)和sold_contract_date(成交日期)列
    
    返回:
        添加了时间感知近邻价格特征的DataFrame
    """
    print("创建时间感知的近邻价格特征...")
    
    # 确保DataFrame中有价格标签和成交日期
    required_cols = ['y_label', 'sold_contract_date', 'std_address', 'locality_post']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f"警告: 缺少必要的列 {missing_cols}，无法创建时间感知近邻价格特征")
        return df
    
    # 复制DataFrame以避免修改原始数据
    df_copy = df.copy()
    
    # 确保成交日期列是日期类型
    
    #确保成交日期列是日期类型
    try:
        # 明确指定日期格式为"YYYY/M/D"
        df_copy['sold_contract_date_new'] = pd.to_datetime(df_copy['sold_contract_date'], format='%Y-%m-%d', errors='coerce')
        
        # 检查是否有无效日期并处理
        invalid_dates = df_copy['sold_contract_date'].isna().sum()
        if invalid_dates > 0:
            print(f"警告: 有 {invalid_dates} 条日期记录无法解析")
            # 填充无效日期为数据集中的中位数日期
            median_date = df_copy['sold_contract_date'].dropna().median()
            df_copy['sold_contract_date'].fillna(median_date, inplace=True)
    except Exception as e:
        print(f"转换成交日期时出错: {str(e)}")
        print("示例日期格式应为: 2022/1/29")
        return df
    
    print(f"成交日期列的数据类型: {df_copy['sold_contract_date_new'].dtype}")
    print(f"成交日期范围: {df_copy['sold_contract_date_new'].min()} 至 {df_copy['sold_contract_date_new'].max()}")
    
    # 预处理: 按邮政编码分组
    postcode_groups = {}
    for post_code in df_copy['locality_post'].unique():
        postcode_groups[post_code] = df_copy[df_copy['locality_post'] == post_code]
    
    print(f"共有 {len(postcode_groups)} 个不同的邮政编码")
    
    # 固定使用k=5作为KNN的近邻数量
    k = 5
    
    # 1. 基于地理位置的时间感知近邻价格特征
    if 'prop_x' in df.columns and 'prop_y' in df.columns:
        from sklearn.neighbors import KNeighborsRegressor
        
        # 初始化价格预测数组
        geo_pred = np.zeros(len(df_copy)) * np.nan
        
        # 为每个房屋单独计算其历史近邻价格
        for i, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc=f"计算{k}个地理近邻价格"):
            try:
                # 获取当前房屋的成交日期、地址和邮政编码
                current_date = row['sold_contract_date_new']
                current_address = row['std_address']
                current_post = row['locality_post']
                
                # 首先获取同一邮政编码区域的房屋
                same_post_data = postcode_groups.get(current_post, pd.DataFrame())
                
                # 筛选出在当前房屋成交日期之前的所有房屋数据，并排除当前房屋自身
                historical_data = same_post_data[
                    (same_post_data['sold_contract_date_new'] < current_date) & 
                    (same_post_data['std_address'] != current_address)
                ]
                #print("same_post_data- -------",len(historical_data))
                # 如果同邮编历史数据不足，尝试获取附近邮编的数据
                if len(historical_data) < k * 2:  # 获取更多数据以确保有足够的有效样本
                    # 查找所有数据中的历史数据
                    all_historical_data = df_copy[
                        (df_copy['sold_contract_date_new'] < current_date) & 
                        (df_copy['std_address'] != current_address)
                    ]
                    
                    # 如果历史数据不足k个，则跳过
                    if len(all_historical_data) < k:
                        continue
                    
                    # 使用所有历史数据
                    historical_data = all_historical_data
                
                # 准备地理坐标数据
                X_geo_hist = historical_data[['prop_x', 'prop_y']].dropna().values
                y_price_hist = historical_data.loc[historical_data[['prop_x', 'prop_y']].dropna().index, 'y_label'].values
                
                # 如果历史有效数据不足k个，则跳过
                if len(X_geo_hist) < k:
                    continue
                
                # 创建KNN回归器并拟合历史数据
                knn = KNeighborsRegressor(n_neighbors=min(k, len(X_geo_hist)), weights='distance')
                knn.fit(X_geo_hist, y_price_hist)
                
                # 预测当前房屋的价格
                current_coords = np.array([[row['prop_x'], row['prop_y']]])
                geo_pred[i] = knn.predict(current_coords)[0]
            except Exception as e:
                print(f"计算房屋 {row['std_address']} 的地理近邻价格时出错: {str(e)}")
        
        # 添加地理近邻价格特征
        df_copy[f'geo_knn_{k}_price_historical'] = geo_pred
        
        # 注释掉的价格比率特征，不再计算
        # df_copy[f'price_to_geo_knn_{k}_ratio_historical'] = df_copy['y_label'] / df_copy[f'geo_knn_{k}_price_historical'].replace(0, np.nan)
        # drop掉sold_contract_date_new列
        print(f"已添加{k}个时间感知地理近邻的价格特征，非空值比例: {df_copy[f'geo_knn_{k}_price_historical'].notna().mean():.2%}")
    
    # 2. 基于邮政编码/地区的时间感知价格统计特征
    if 'locality_post' in df.columns:
        # 为每个房屋计算其邮政编码区域的历史价格统计
        # 初始化统计数据列
        postcode_stats_cols = ['postcode_count_hist', 'postcode_mean_price_hist', 
                             'postcode_median_price_hist', 'postcode_std_price_hist']
        for col in postcode_stats_cols:
            df_copy[col] = np.nan
        
        # 为每个房屋单独计算
        for i, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc="计算邮政编码历史价格"):
            try:
                # 获取当前房屋的成交日期、地址和邮政编码
                current_date = row['sold_contract_date_new']
                current_address = row['std_address']
                current_postcode = row['locality_post']
                
                # 使用预先分组的数据快速获取同邮编的房屋
                same_post_data = postcode_groups.get(current_postcode, pd.DataFrame())
                
                # 筛选出在当前房屋成交日期之前且属于同一邮政编码的所有房屋数据
                # 并排除当前房屋自身
                historical_data = same_post_data[
                    (same_post_data['sold_contract_date_new'] < current_date) & 
                    (same_post_data['std_address'] != current_address)
                ]
                
                # 如果历史数据过少，则跳过
                if len(historical_data) < 3:
                    continue
                
                # 计算邮政编码区域的历史价格统计
                df_copy.at[i, 'postcode_count_hist'] = len(historical_data)
                df_copy.at[i, 'postcode_mean_price_hist'] = historical_data['y_label'].mean()
                df_copy.at[i, 'postcode_median_price_hist'] = historical_data['y_label'].median()
                df_copy.at[i, 'postcode_std_price_hist'] = historical_data['y_label'].std()
            except Exception as e:
                print(f"计算房屋 {row['std_address']} 的邮政编码历史价格时出错: {str(e)}")
        
        # 注释掉的价格差异特征，不再计算
        # df_copy['price_diff_from_postcode_mean_hist'] = df_copy['y_label'] - df_copy['postcode_mean_price_hist']
        # df_copy['price_ratio_to_postcode_mean_hist'] = df_copy['y_label'] / df_copy['postcode_mean_price_hist'].replace(0, np.nan)
        
        # 输出特征创建情况
        print(f"已添加基于邮政编码的时间感知价格特征，非空值比例: {df_copy['postcode_mean_price_hist'].notna().mean():.2%}")
    
    # 3. 基于郊区(suburb)的时间感知价格统计特征
    if 'locality_suburb' in df.columns:
        # 预处理: 按郊区分组
        suburb_groups = {}
        for suburb in df_copy['locality_suburb'].unique():
            suburb_groups[suburb] = df_copy[df_copy['locality_suburb'] == suburb]
        
        # 为每个房屋计算其郊区的历史价格统计
        # 初始化统计数据列
        suburb_stats_cols = ['suburb_count_hist', 'suburb_mean_price_hist', 
                           'suburb_median_price_hist', 'suburb_std_price_hist']
        for col in suburb_stats_cols:
            df_copy[col] = np.nan
        
        # 为每个房屋单独计算
        for i, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc="计算郊区历史价格"):
            try:
                # 获取当前房屋的成交日期、地址和郊区
                current_date = row['sold_contract_date_new']
                current_address = row['std_address']
                current_suburb = row['locality_suburb']
                
                # 使用预先分组的数据快速获取同郊区的房屋
                same_suburb_data = suburb_groups.get(current_suburb, pd.DataFrame())
                
                # 筛选出在当前房屋成交日期之前且属于同一郊区的所有房屋数据
                # 并排除当前房屋自身
                historical_data = same_suburb_data[
                    (same_suburb_data['sold_contract_date_new'] < current_date) & 
                    (same_suburb_data['std_address'] != current_address)
                ]
                
                # 如果历史数据过少，则跳过
                if len(historical_data) < 3:
                    continue
                
                # 计算郊区的历史价格统计
                df_copy.at[i, 'suburb_count_hist'] = len(historical_data)
                df_copy.at[i, 'suburb_mean_price_hist'] = historical_data['y_label'].mean()
                df_copy.at[i, 'suburb_median_price_hist'] = historical_data['y_label'].median()
                df_copy.at[i, 'suburb_std_price_hist'] = historical_data['y_label'].std()
            except Exception as e:
                print(f"计算房屋 {row['std_address']} 的郊区历史价格时出错: {str(e)}")
        
        # 注释掉的价格差异特征，不再计算
        # df_copy['price_diff_from_suburb_mean_hist'] = df_copy['y_label'] - df_copy['suburb_mean_price_hist']
        # df_copy['price_ratio_to_suburb_mean_hist'] = df_copy['y_label'] / df_copy['suburb_mean_price_hist'].replace(0, np.nan)
        
        # 输出特征创建情况
        print(f"已添加基于郊区的时间感知价格特征，非空值比例: {df_copy['suburb_mean_price_hist'].notna().mean():.2%}")
    
    # 4. 基于房屋特征相似度的时间感知近邻价格
    # feature_cols = ['prop_bed', 'prop_bath', 'prop_carpark', 'internal_area', 'land_size']
    # if all(col in df.columns for col in feature_cols):
    #     from sklearn.neighbors import KNeighborsRegressor
    #     from sklearn.preprocessing import StandardScaler
        
    #     # 初始化标准化器
    #     scaler = StandardScaler()
        
    #     # 初始化价格预测数组
    #     feature_pred = np.zeros(len(df_copy)) * np.nan
        
    #     # 为每个房屋单独计算其历史近邻价格
    #     for i, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc=f"计算{k}个特征近邻价格"):
    #         try:
    #             # 获取当前房屋的成交日期、地址和邮政编码
    #             current_date = row['sold_contract_date']
    #             current_address = row['std_address']
    #             current_post = row['locality_post']
                
    #             # 首先获取同一邮政编码区域的房屋
    #             same_post_data = postcode_groups.get(current_post, pd.DataFrame())
                
    #             # 筛选出在当前房屋成交日期之前的所有房屋数据，并排除当前房屋自身
    #             historical_data = same_post_data[
    #                 (same_post_data['sold_contract_date'] < current_date) & 
    #                 (same_post_data['std_address'] != current_address)
    #             ]
                
    #             # 如果同邮编历史数据不足，尝试获取所有历史数据
    #             if len(historical_data) < k * 2:  # 获取更多数据以确保有足够的有效样本
    #                 # 查找所有数据中的历史数据
    #                 all_historical_data = df_copy[
    #                     (df_copy['sold_contract_date'] < current_date) & 
    #                     (df_copy['std_address'] != current_address)
    #                 ]
                    
    #                 # 如果历史数据不足k个，则跳过
    #                 if len(all_historical_data) < k:
    #                     continue
                    
    #                 # 使用所有历史数据
    #                 historical_data = all_historical_data
                
    #             # 准备特征数据
    #             X_features_hist = historical_data[feature_cols].dropna()
    #             y_price_hist = historical_data.loc[X_features_hist.index, 'y_label'].values
                
    #             # 如果历史有效数据不足k个，则跳过
    #             if len(X_features_hist) < k:
    #                 continue
                
    #             # 标准化特征
    #             X_features_scaled = scaler.fit_transform(X_features_hist)
                
    #             # 创建KNN回归器并拟合历史数据
    #             knn = KNeighborsRegressor(n_neighbors=min(k, len(X_features_hist)), weights='distance')
    #             knn.fit(X_features_scaled, y_price_hist)
                
    #             # 预测当前房屋的价格
    #             current_features = np.array([row[feature_cols]])
    #             if np.isnan(current_features).any():
    #                 continue
    #             current_features_scaled = scaler.transform(current_features)
    #             feature_pred[i] = knn.predict(current_features_scaled)[0]
    #         except Exception as e:
    #             print(f"计算房屋 {row['std_address']} 的特征近邻价格时出错: {str(e)}")
        
    #     # 添加特征近邻价格特征
    #     df_copy[f'feature_knn_{k}_price_historical'] = feature_pred
        
        # 注释掉的价格比率特征，不再计算
        # df_copy[f'price_to_feature_knn_{k}_ratio_historical'] = df_copy['y_label'] / df_copy[f'feature_knn_{k}_price_historical'].replace(0, np.nan)
     
        print(f"已添加{k}个时间感知特征近邻的价格特征，非空值比例: {df_copy[f'feature_knn_{k}_price_historical'].notna().mean():.2%}")
    df_copy = df_copy.drop(columns=['sold_contract_date_new'])
  
    return df_copy

def create_time_features(df):
    """
    从成交日期(sold_contract_date)创建时间特征
    
    参数:
        df: 输入数据DataFrame，必须包含sold_contract_date列
    
    返回:
        添加了时间特征的DataFrame
    """
    print("从成交日期创建时间特征...")
    
    # 确保DataFrame中有成交日期列
    if 'sold_contract_date' not in df.columns:
        print("警告: 缺少sold_contract_date列，无法创建时间特征")
        return df
    
    # 复制DataFrame以避免修改原始数据
    df_copy = df.copy()
    
    # 输出几个日期样本用于调试
    print("原始日期样本:")
    for date in df_copy['sold_contract_date'].head(5):
        print(f"  {date}, 类型: {type(date)}")
    
    #确保成交日期列是日期类型
    try:
        # 明确指定日期格式为"2022-01-01"
        df_copy['sold_contract_date'] = df_copy['sold_contract_date'].astype(str)

        # 转换为 datetime 类型
        df_copy['sold_contract_date'] = pd.to_datetime(df_copy['sold_contract_date'], format='%Y-%m-%d', errors='coerce')
        
        # 检查是否有无效日期并处理
        invalid_dates = df_copy['sold_contract_date'].isna().sum()
        if invalid_dates > 0:
            print(f"警告: 有 {invalid_dates} 条日期记录无法解析")
            # 填充无效日期为数据集中的中位数日期
            median_date = df_copy['sold_contract_date'].dropna().median()
            df_copy['sold_contract_date'].fillna(median_date, inplace=True)
    except Exception as e:
        print(f"转换成交日期时出错: {str(e)}")
        print("示例日期格式应为: 2022/1/29")
        return df
    
    print(df_copy['sold_contract_date'].head(10))
    print(f"成交日期列的数据类型: {df_copy['sold_contract_date'].dtype}")
    print(f"成交日期范围: {df_copy['sold_contract_date'].min()} 至 {df_copy['sold_contract_date'].max()}")
    
    # 1. 提取基础时间特征
    df_copy['year'] = df_copy['sold_contract_date'].dt.year
    df_copy['month'] = df_copy['sold_contract_date'].dt.month
    df_copy['day'] = df_copy['sold_contract_date'].dt.day
    df_copy['dayofweek'] = df_copy['sold_contract_date'].dt.dayofweek  # 0=星期一, 6=星期日
    df_copy['quarter'] = df_copy['sold_contract_date'].dt.quarter
    
    # 2. 创建季节特征（1=春，2=夏，3=秋，4=冬）
    # 注意：南半球和北半球的季节划分不同，这里按北半球划分
    df_copy['season'] = df_copy['month'].apply(lambda x: 
                                           1 if x in [3, 4, 5] else  # 春季
                                           2 if x in [6, 7, 8] else  # 夏季
                                           3 if x in [9, 10, 11] else  # 秋季
                                           4)  # 冬季
    
    # 季节的周期性编码
    df_copy['season_sin'] = np.sin(2 * np.pi * df_copy['season'] / 4)
    df_copy['season_cos'] = np.cos(2 * np.pi * df_copy['season'] / 4)
    
    # 3. 月份和星期的周期性编码（捕捉循环特性）
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
    
    df_copy['dayofweek_sin'] = np.sin(2 * np.pi * df_copy['dayofweek'] / 7)
    df_copy['dayofweek_cos'] = np.cos(2 * np.pi * df_copy['dayofweek'] / 7)
    
    # 4. 月初/月末/年初/年末特征
    # 月初定义为每月的前10天
    df_copy['is_month_start'] = (df_copy['day'] <= 10).astype(int)
    # 月末定义为每月的最后10天
    df_copy['is_month_end'] = (df_copy['day'] >= 21).astype(int)
    # 年初定义为1-2月
    df_copy['is_year_start'] = df_copy['month'].isin([1, 2]).astype(int)
    # 年末定义为11-12月
    df_copy['is_year_end'] = df_copy['month'].isin([11, 12]).astype(int)
    
    # 5. 节假日季特征（假设12月和1月为节日季）
    df_copy['is_holiday_season'] = df_copy['month'].isin([12, 1]).astype(int)
    
    # 6. 日期差特征（距离某个参考日期的天数）
    reference_date = pd.Timestamp('2015-01-01')  # 选择一个参考日期
    df_copy['days_since_reference'] = (df_copy['sold_contract_date'] - reference_date).dt.days
    
    # 7. 年代特征
    df_copy['decade'] = (df_copy['year'] // 10) * 10
    
    # 8. 工作日/周末特征
    df_copy['is_weekend'] = (df_copy['dayofweek'] >= 5).astype(int)
    
    # 9. 季度中的月份特征
    df_copy['month_in_quarter'] = (df_copy['month'] - 1) % 3 + 1
    
    # 10. 时间趋势特征
    min_date = df_copy['sold_contract_date'].min()
    df_copy['time_trend'] = (df_copy['sold_contract_date'] - min_date).dt.days
    df_copy['time_trend_squared'] = df_copy['time_trend'] ** 2
    
    # 将sold_contract_date保存为字符串类型的日期字段，方便查看但不作为特征
    df_copy['sold_contract_date'] = df_copy['sold_contract_date'].dt.strftime('%Y-%m-%d')
    
    # 移除原始日期列，避免将日期对象传递给模型
    #df_copy = df_copy.drop(columns=['sold_contract_date'])
    
    print(f"已创建 {len(df_copy.columns) - len(df.columns) + 1} 个时间特征")  # +1 是因为删除了一列
    
    return df_copy

def gen_house_price_features(gen_deep_features=False):
    df = pd.read_csv(r"D:\code\gen\property_wize_price\resources\house_samples.csv")
    # print(encoders)
    print(df.columns)
    
    # 输出sold_contract_date列的一些样本，确认格式
    if 'sold_contract_date' in df.columns:
        print("原始sold_contract_date样本:")
        print(df['sold_contract_date'].head(10))
    
    # 创建两个DataFrame，一个用于存储常规特征，一个用于存储深度模型特征
    df_regular = df.copy()  # 常规特征
    
    # 如果key_advantages列存在，使用BERT进行向量化
    if 'key_advantages' in df.columns and gen_deep_features:
        print("正在将key_advantages转换为BERT向量特征...")
        
        # 获取唯一标识符列，用于后续合并数据
        id_columns = ['std_address','y_label']
        df_deep = df[id_columns].copy()  # 仅包含ID列的深度特征DataFrame

        
        # 填充key_advantages中的缺失值并转换为字符串
        df['key_advantages'] = df['key_advantages'].fillna('').astype(str)

        df['txt'] = df['key_advantages'] + df['std_address']
        
        # 打印key_advantages列的一些样本和数据类型，用于调试
        print(f"key_advantages列的数据类型: {type(df['key_advantages'])}")
        print(f"前5个key_advantages样本:")
        for i, text in enumerate(df['key_advantages'].head(5)):
            print(f"  {i+1}. 类型: {type(text)}, 内容: {str(text)[:50] if len(str(text)) > 50 else str(text)}")
        
        # 确保转换为字符串列表
        texts = df['txt'].tolist()
        
        print(f"转换后texts的类型: {type(texts)}")
        print(f"texts中第一个元素的类型: {type(texts[0]) if texts else 'Empty list'}")
        print(f"texts长度: {len(texts)}")
        
        # 使用BERT将文本转换为向量
        embeddings = text_to_bert_embeddings(texts)
        
        #使用PCA将BERT嵌入向量降维到10维
        print("正在对BERT向量进行PCA降维到10维...")
        pca = PCA(n_components=10)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # 输出解释方差比例
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"PCA降维后保留的信息量: {explained_variance:.2%}")
        
        # 将降维后的向量添加为新特征列到深度特征DataFrame
        for i in range(reduced_embeddings.shape[1]):
            df_deep[f'key_adv_bert_{i}'] = reduced_embeddings[:, i]
            df_regular[f'key_adv_bert_{i}'] = reduced_embeddings[:, i]
        
        # 保存深度特征到单独的CSV文件
        deep_features_path = r"D:\code\gen\property_wize_price\resources\house_samples_deep_features.csv"
        df_deep.to_csv(deep_features_path, index=False)
        print(f"深度特征已保存到: {deep_features_path}")
        
        # 从常规特征DataFrame中删除key_advantages列
        df_regular = df_regular.drop(columns=['key_advantages'])
        #df_regular = df_regular.drop(columns=['txt'])
        print(f"BERT特征降维完成，添加了{embeddings.shape[1]}个深度特征")
    else:
        print("数据集中未找到key_advantages列")

       # 创建基于时间感知的近邻价格特征
    df_regular = create_time_aware_neighborhood_features(df_regular) #[:1000]

    df_regular = create_time_features(df_regular)
    print("sold_contract_date- 1111",df_regular['sold_contract_date'].head(10))
    # 处理常规特征
    df_regular, encoders = preprocess_features(df_regular, skip_encoding_cols=['std_address'])
    print("sold_contract_date- 2222",df_regular['sold_contract_date'].head(10))


    
    # 创建新特征
    df_regular = create_new_features(df_regular)
    print("sold_contract_date- 1111",df_regular['sold_contract_date'].head(10))

    #df_regular = df_regular.drop(columns=['sold_contract_date'])
    #df_regular = df_regular.drop(columns=['land_value_internal_area_interaction'])   


    # 按照std_address去重
    df_regular = df_regular.drop_duplicates(subset=['std_address'])
    
    # 去掉不需要的列
    df_regular = df_regular.drop(columns=['full_address_x', 'prop_id_y',
                                          'land_value_2022','sec_med_July18','sec_med_July19','sec_med_July20','sec_med_July21','sec_med_July22']) #v,'prop_id' 'prop_id_x',
    #                       'prop_id_y','land_value_2022','sec_med_July18','sec_med_July19','sec_med_July20','sec_med_July21','sec_med_July22']) #v,'prop_id' 'prop_id_x',
  
    # 移除低方差特征
    low_var_threshold = 0.01
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
 
    for col in numeric_cols:
        if df[col].var() < low_var_threshold:
            df = df.drop(columns=[col])
            print(f"移除低方差特征: {col}")
    
    df_regular = encode_categorical_features(df_regular)
    # 重命名列并调整列顺序
    df_regular.rename(columns={'sold_price': 'y_label', 'prop_id_x': 'prop_id'}, inplace=True)
    df_regular.insert(0, 'prop_id', df_regular.pop('prop_id'))
    df_regular.insert(0, 'std_address', df_regular.pop('std_address'))
    df_regular.insert(0, 'y_label', df_regular.pop('y_label'))
    
    print(df_regular.columns)
    print(df_regular.shape) 
    
    # 保存常规特征到CSV文件
    df_regular.to_csv(r"D:\code\gen\property_wize_price\resources\house_samples_features_v2.csv", index=False)
    print(f"常规特征已保存到: D:\\code\\gen\\property_wize_price\\resources\\house_samples_features_v2.csv")


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


# 使用示例
if __name__ == "__main__": 
    gen_house_price_features(gen_deep_features=True) 