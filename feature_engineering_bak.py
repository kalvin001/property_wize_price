import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import stats
import argparse
from pathlib import Path
import re
import jieba  # 导入中文分词库
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings("ignore")

# 设置编码
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.6及以下版本不支持reconfigure
        pass

# 添加BERT模型处理函数
def load_bert_model():
    """
    加载预训练的BERT模型
    
    返回:
        tokenizer: BERT分词器
        model: BERT模型
    """
    print("加载BERT模型...")
    try:
        # 加载预训练的中文BERT模型
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertModel.from_pretrained('bert-base-chinese')
        print("BERT模型加载成功")
        return tokenizer, model
    except Exception as e:
        print(f"加载BERT模型失败: {str(e)}")
        print("将使用备用方法处理文本特征")
        return None, None

def get_bert_embeddings(texts, tokenizer, model, max_length=128, batch_size=32):
    """
    使用BERT模型获取文本的embedding向量
    
    参数:
        texts: 文本列表
        tokenizer: BERT分词器
        model: BERT模型
        max_length: 最大序列长度
        batch_size: 批处理大小
        
    返回:
        numpy数组形式的embedding向量
    """
    if tokenizer is None or model is None:
        return None
        
    # 将模型设置为评估模式
    model.eval()
    
    # 初始化结果列表
    all_embeddings = []
    
    # 处理文本分批
    for i in tqdm(range(0, len(texts), batch_size), desc="生成文本Embedding"):
        batch_texts = texts[i:i+batch_size]
        
        # 预处理文本
        preprocessed_texts = [preprocess_text(text) for text in batch_texts]
        
        # 对文本进行编码
        encoded_input = tokenizer(
            preprocessed_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # 将输入移动到可用设备
        if torch.cuda.is_available():
            encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
            model = model.cuda()
        
        # 使用模型获取输出，不计算梯度
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # 使用[CLS]令牌的输出(表示整个句子的语义)
        batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(batch_embeddings)
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 合并所有批次的结果
    if all_embeddings:
        return np.vstack(all_embeddings)
    return None

def process_text_with_bert(df, text_columns):
    """
    使用BERT处理文本列，将文本转换为embedding特征
    
    参数:
        df: 数据框
        text_columns: 要处理的文本列列表
        
    返回:
        添加了embedding特征的数据框
    """
    print(f"使用BERT处理文本特征: {text_columns}")
    
    # 加载BERT模型
    tokenizer, model = load_bert_model()
    
    if tokenizer is None or model is None:
        print("BERT模型加载失败，跳过文本embedding处理")
        return df
    
    # 跟踪原始文本列，稍后移除
    processed_text_columns = []
    
    # 处理每个文本列
    for col in text_columns:
        if col in df.columns:
            print(f"处理文本列: {col}")
            processed_text_columns.append(col)
            
            # 获取文本数据
            texts = df[col].fillna('').astype(str).values
            
            # 获取文本的embedding
            embeddings = get_bert_embeddings(texts, tokenizer, model)
            
            if embeddings is not None:
                # 降维以减少特征数量(可选)
                reduced_dim = min(20, embeddings.shape[1])  # 增加保留维度到20
                svd = TruncatedSVD(n_components=reduced_dim)
                reduced_embeddings = svd.fit_transform(embeddings)
                
                # 将embedding添加为新特征，并确保是float32类型
                for i in range(reduced_dim):
                    df[f'{col}_bert_{i}'] = reduced_embeddings[:, i].astype(np.float32)
                
                print(f"  - 成功为{col}创建{reduced_dim}个BERT embedding特征")
                
                # 计算解释方差
                explained_var = svd.explained_variance_ratio_.sum()
                print(f"  - 这些特征解释了原始embedding的{explained_var:.2%}方差")
            else:
                print(f"  - 无法为{col}生成embedding")
                # 创建一个虚拟特征以避免错误
                df[f'{col}_dummy'] = 0.0
    
    # 清理内存
    del tokenizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return df, processed_text_columns

def load_data(data_file):
    """加载原始数据"""
    print(f"正在加载数据: {data_file}")
    df = pd.read_csv(data_file)
    print(f"数据加载完成，共有 {df.shape[0]} 行 {df.shape[1]} 列")
    return df

# 添加文本处理函数
def preprocess_text(text):
    """
    对文本进行预处理：清洗、分词等
    """
    if isinstance(text, str):
        # 移除特殊字符，保留中文字符、英文字母和数字
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
        # 转换为小写
        text = text.lower()
        # 使用jieba进行中文分词
        words = ' '.join(jieba.cut(text))
        return words
    return ''

def vectorize_text_features(df):
    """
    对文本特征进行向量化
    
    参数:
        df: 包含文本特征的DataFrame
        
    返回:
        包含向量化特征的DataFrame和需要删除的原始文本列列表
    """
    print("对文本特征进行向量化...")
    
    # 待处理的文本特征列表
    text_features = []
    
    # 检查每一列，找出文本特征
    for col in df.columns:
        if df[col].dtype == 'object':
            # 保留所有文本特征，包括house_name
            if col != 'prop_id' and col != 'y_label':  # 排除ID和目标列
                text_features.append(col)
    
    print(f"发现需要向量化的文本特征: {text_features}")
    
    # 跟踪所有处理过的原始文本列
    all_processed_columns = []
    
    # 使用BERT处理所有重要文本特征，包括house_name和key_advantages
    key_text_features = [col for col in text_features if col == 'house_name' 
                        or 'advantage' in col.lower() 
                        or 'description' in col.lower() 
                        or 'feature' in col.lower()
                        or 'address' in col.lower()]
    
    if key_text_features:
        print(f"使用BERT处理关键文本特征: {key_text_features}")
        df, processed_bert_columns = process_text_with_bert(df, key_text_features)
        all_processed_columns.extend(processed_bert_columns)
    
    # 对其余文本特征使用TF-IDF处理
    remaining_features = [col for col in text_features if col not in key_text_features]
    
    # 处理每一个文本特征
    for feature in remaining_features:
        if feature in df.columns:
            print(f"使用TF-IDF处理文本特征: {feature}")
            all_processed_columns.append(feature)
                
            # 预处理文本
            df[f'{feature}_processed'] = df[feature].astype(str).apply(preprocess_text)
            
            try:
                # 使用TF-IDF向量化文本
                tfidf = TfidfVectorizer(
                    max_features=100,  # 限制特征数量
                    min_df=2,          # 降低最小文档频率要求
                    max_df=0.95,       # 增加最大文档频率
                    stop_words='english'  # 英文停用词
                )
                
                # 转换文本数据
                text_tfidf = tfidf.fit_transform(df[f'{feature}_processed'])
                
                # 检查特征数量，如果太少则调整SVD组件数量
                n_features = text_tfidf.shape[1]
                n_components = min(10, n_features)  # 确保组件数不超过特征数
                
                if n_components >= 2:  # 至少需要2个特征才能应用SVD
                    # 应用降维，将高维TF-IDF向量降到较低维度
                    svd = TruncatedSVD(n_components=n_components)
                    text_svd = svd.fit_transform(text_tfidf)
                    
                    # 为每个组件创建特征
                    for i in range(text_svd.shape[1]):
                        df[f'{feature}_vec_{i}'] = text_svd[:, i].astype(np.float32)
                    
                    # 获取并保存最重要的词汇及其权重
                    feature_names = tfidf.get_feature_names_out()
                    
                    # 计算每个词的平均TF-IDF得分
                    tfidf_sum = np.asarray(text_tfidf.sum(axis=0)).flatten()
                    word_importance = [(word, tfidf_sum[idx]) for idx, word in enumerate(feature_names)]
                    top_words = sorted(word_importance, key=lambda x: x[1], reverse=True)[:min(10, len(word_importance))]
                    
                    print(f"  - {feature}特征的Top10关键词: {[word for word, _ in top_words]}")
                    print(f"  - 成功为{feature}创建{n_components}个向量特征")
                else:
                    # 如果特征太少，直接编码
                    print(f"  - {feature}的特征数量太少({n_features})，使用虚拟特征替代")
                    df[f'{feature}_encoded'] = 0.0
            except Exception as e:
                print(f"  - 处理{feature}时出错: {str(e)}")
                # 为出错的特征创建虚拟特征
                df[f'{feature}_dummy'] = 0.0
            
            # 删除处理过的中间列
            if f'{feature}_processed' in df.columns:
                all_processed_columns.append(f'{feature}_processed')
    
    return df, all_processed_columns

def handle_missing_values(df):
    """
    处理缺失值
    
    Args:
        df: 数据框
        
    Returns:
        处理了缺失值的数据框
    """
    print("处理缺失值...")
    
    # 计算每列的缺失值比例
    missing_ratio = df.isna().mean()
    
    # 不删除任何列，只填充缺失值
    
    # 对于数值型列，用中位数填充缺失值
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # 对于分类型列，用众数填充缺失值
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isna().sum() > 0:  # 如果有缺失值
            mode_val = df[col].mode()[0]  # 获取众数
            df[col] = df[col].fillna(mode_val)
    
    # 对于布尔型列，用False填充缺失值
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].fillna(False)
    
    return df



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

def create_new_features(df):
    """
    创建新特征
    
    参数:
        df: 包含原始特征的DataFrame
    
    返回:
        添加了新特征的DataFrame
    """
    print("创建新特征...")
    
    # 确保必要的列存在
    required_cols = ['internal_area', 'land_size', 'land_value', 'prop_bed', 'prop_bath']
    for col in required_cols:
        if col not in df.columns:
            print(f"警告: 未找到列 {col}，无法创建某些新特征")
    
    # 确保数值列的类型正确
    numeric_cols = ['prop_bed', 'prop_bath', 'prop_carpark', 'land_value', 'land_size', 
                   'internal_area', 'balcony_area', 'garage_area', 'luxury_level', 
                   'luxury_level_score', 'luxury_level_desc_score', 'luxury_level_images_score',
                   'quality_score', 'view_quality']
    
    for col in numeric_cols:
        if col in df.columns:
            # 如果列存在，尝试转换为数值类型
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                print(f"警告: 无法将列 {col} 转换为数值类型")
    
    # 创建新的特征
    
    # 计算居住面积相关特征（不使用房价）
    if 'internal_area' in df.columns and 'land_size' in df.columns:
        # 计算土地利用率
        mask = (df['land_size'] > 0)
        df.loc[mask, 'land_utilization'] = df.loc[mask, 'internal_area'] / df.loc[mask, 'land_size']
    
    # 计算总房间数
    if 'prop_bed' in df.columns and 'prop_bath' in df.columns:
        df['total_rooms'] = df['prop_bed'] + df['prop_bath']
        
        # 计算浴室/卧室比例
        mask = (df['prop_bed'] > 0)
        df.loc[mask, 'bath_bed_ratio'] = df.loc[mask, 'prop_bath'] / df.loc[mask, 'prop_bed']
    
    # 计算卧室大小类别
    if 'internal_area' in df.columns and 'prop_bed' in df.columns:
        mask = (df['prop_bed'] > 0)
        df.loc[mask, 'bedroom_size_category'] = df.loc[mask, 'internal_area'] / df.loc[mask, 'prop_bed']
        # 将卧室大小分为不同类别
        df['bedroom_size_category'] = pd.cut(
            df['bedroom_size_category'], 
            bins=[0, 15, 25, 35, float('inf')],
            labels=['小', '中', '大', '超大']
        )
        # 将分类变量转换为数值编码
        le = LabelEncoder()
        df['bedroom_size_category'] = le.fit_transform(df['bedroom_size_category'].astype(str))
    
    # 创建基于经纬度的特征
    if 'prop_x' in df.columns and 'prop_y' in df.columns:
        # 确保是数值类型
        df['prop_x'] = pd.to_numeric(df['prop_x'], errors='coerce')
        df['prop_y'] = pd.to_numeric(df['prop_y'], errors='coerce')
        
        # 计算与城市中心的距离（假设中心点）
        city_center_x = df['prop_x'].mean()
        city_center_y = df['prop_y'].mean()
        
        df['dist_to_center'] = np.sqrt(
            (df['prop_x'] - city_center_x)**2 + 
            (df['prop_y'] - city_center_y)**2
        )
    
    # 创建基于土地价值的特征（不使用房价）
    if 'land_value' in df.columns and 'land_size' in df.columns:
        mask = (df['land_size'] > 0)
        df.loc[mask, 'land_value_per_sqm'] = df.loc[mask, 'land_value'] / df.loc[mask, 'land_size']
    
    # 计算卧室密度
    if 'prop_bed' in df.columns and 'internal_area' in df.columns:
        mask = (df['internal_area'] > 0)
        df.loc[mask, 'bedroom_density'] = df.loc[mask, 'prop_bed'] / df.loc[mask, 'internal_area'] * 100  # 每100平方米的卧室数
    
    # 计算浴室密度
    if 'prop_bath' in df.columns and 'internal_area' in df.columns:
        mask = (df['internal_area'] > 0)
        df.loc[mask, 'bathroom_density'] = df.loc[mask, 'prop_bath'] / df.loc[mask, 'internal_area'] * 100  # 每100平方米的浴室数
    
    # 如果有车位信息，计算车位相关特征
    if 'prop_carpark' in df.columns and 'internal_area' in df.columns:
        mask = (df['internal_area'] > 0)
        df.loc[mask, 'carpark_density'] = df.loc[mask, 'prop_carpark'] / df.loc[mask, 'internal_area'] * 100  # 每100平方米的车位数
    
    # 创建奢华指数综合特征
    luxury_features = ['luxury_level', 'luxury_level_score', 'luxury_level_desc_score', 'luxury_level_images_score']
    luxury_features = [f for f in luxury_features if f in df.columns]
    
    if len(luxury_features) > 0:
        # 确保所有奢华特征是数值类型
        for col in luxury_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # 创建综合指标
        df['luxury_index'] = df[luxury_features].mean(axis=1)
    
    # 创建房产质量综合特征
    quality_features = ['quality_score', 'view_quality']
    quality_features = [f for f in quality_features if f in df.columns]
    
    if len(quality_features) > 0:
        # 确保所有质量特征是数值类型
        for col in quality_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # 创建综合指标
        df['property_quality_index'] = df[quality_features].mean(axis=1)
    
    # 创建是否有游泳池的特征
    if 'indoor_swimming_pool_cnt' in df.columns and 'outdoor_swimming_pool_cnt' in df.columns:
        df['indoor_swimming_pool_cnt'] = pd.to_numeric(df['indoor_swimming_pool_cnt'], errors='coerce').fillna(0)
        df['outdoor_swimming_pool_cnt'] = pd.to_numeric(df['outdoor_swimming_pool_cnt'], errors='coerce').fillna(0)
        df['has_pool'] = ((df['indoor_swimming_pool_cnt'] + df['outdoor_swimming_pool_cnt']) > 0).astype(int)
    
    # 不删除任何特征，即使是NaN值过多或方差过小的特征
    
    # 处理缺失值和无穷值
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    for col in numeric_cols:
        if col != 'y_label' and col != 'prop_id':
            # 将无穷值替换为NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # 用中位数填充NaN
            median_val = df[col].median()
            if pd.isna(median_val):  # 如果中位数是NaN，用0填充
                median_val = 0
            df[col] = df[col].fillna(median_val)
    
    return df

def handle_outliers(df, target_col='y_label', numeric_cols=None):
    """处理异常值"""
    print("处理异常值...")
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    
    # 排除目标变量
    if target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)
    
    # 使用Z-score方法识别并修复异常值
    for col in numeric_cols:
        if df[col].nunique() > 1:  # 确保列有足够的唯一值
            try:
                z_scores = stats.zscore(df[col], nan_policy='omit')
                abs_z_scores = np.abs(z_scores)
                filtered_entries = (abs_z_scores < 3)  # 3个标准差以外视为异常值
                
                # 替换异常值为列的中位数
                median_value = df.loc[filtered_entries, col].median()
                if pd.isna(median_value):  # 如果中位数是NaN，用0填充
                    median_value = 0
                df.loc[~filtered_entries, col] = median_value
            except:
                print(f"  - 处理{col}的异常值时出错，跳过")
    
    return df

def encode_categorical_features(df):
    """
    将分类特征编码为数值。
    
    参数:
        df: 包含分类特征的DataFrame
    
    返回:
        处理后的DataFrame，保留所有文本特征
    """
    print("编码分类特征...")
    
    # 找到所有需要编码的分类特征列
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    
    print(f"发现需要编码的分类特征: {categorical_features.tolist()}")
    
    if len(categorical_features) > 0:
        # 使用LabelEncoder处理分类变量
        from sklearn.preprocessing import LabelEncoder
        
        for col in categorical_features:
            if col != 'y_label' and col != 'prop_id':  # 排除目标列和ID列
                print(f"编码特征: {col}")
                
                # 检查是否为分类类型，如果是，先转换为对象类型
                if pd.api.types.is_categorical_dtype(df[col]):
                    df[col] = df[col].astype('object')
                
                # 填充缺失值
                if df[col].isna().any():
                    df[col] = df[col].fillna('missing')
                
                # 尝试进行编码
                try:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    print(f"  - 成功对{col}进行编码")
                except Exception as e:
                    print(f"  - 编码特征 {col} 时出错: {str(e)}")
                    # 如果编码失败，创建数值虚拟特征并标记原始列为待删除
                    df[f'{col}_encoded_dummy'] = 0.0
                    print(f"  - 为{col}创建虚拟特征")
    
    return df

def ensure_numeric_features(df):
    """
    确保所有特征都是数值类型
    
    参数:
        df: 数据框
        
    返回:
        处理后的数据框，所有特征都是数值类型
    """
    print("确保所有特征都是数值类型...")
    
    # 标识需要保留的列
    keep_cols = ['prop_id', 'y_label']
    
    # 检查每一列
    for col in df.columns:
        if col not in keep_cols:
            # 如果列不是数值类型，尝试转换
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    # 尝试转换为数值
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"  - 成功将{col}转换为数值类型")
                except:
                    print(f"  - 无法将{col}转换为数值类型，创建虚拟特征")
                    # 创建虚拟特征
                    df[f'{col}_numeric'] = 0.0
                    # 删除原始非数值列
                    df = df.drop(columns=[col])
    
    # 确保所有浮点数列的类型一致
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        if col not in keep_cols:
            df[col] = df[col].astype(np.float32)
    
    # 最后检查一遍，确保没有非数值列
    non_numeric_cols = [col for col in df.columns 
                      if not pd.api.types.is_numeric_dtype(df[col]) 
                      and col != 'prop_id' and col != 'y_label']
    
    if non_numeric_cols:
        print(f"警告: 仍有非数值列无法处理，将删除这些列: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)
    
    return df

def cleanup_dataframe(df, columns_to_remove):
    """
    清理数据框，移除原始文本列
    
    参数:
        df: 数据框
        columns_to_remove: 要移除的列列表
    
    返回:
        清理后的数据框
    """
    print("清理数据框...")
    
    # 移除原始文本列
    if columns_to_remove:
        print(f"移除原始文本列: {columns_to_remove}")
        columns_to_remove = [col for col in columns_to_remove if col in df.columns]
        df = df.drop(columns=columns_to_remove)
    
    # 处理所有BERT特征，确保它们是数值类型
    bert_cols = [col for col in df.columns if '_bert_' in col]
    for col in bert_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
    
    # 处理processed后缀的特征
    processed_cols = [col for col in df.columns if '_processed' in col]
    for col in processed_cols:
        try:
            # 尝试对这些列进行编码
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"  - 成功编码{col}")
        except:
            # 如果编码失败，创建数值虚拟特征并删除原始列
            df[f'{col}_dummy'] = 0.0
            df = df.drop(columns=[col])
            print(f"  - 无法编码{col}，已创建虚拟特征并删除原始列")
    
    # 确保所有其他非数值列被转换或删除
    df = ensure_numeric_features(df)
    
    return df

def engineer_features(data_file, output_file, skip_outlier=False):
    """
    执行特征工程
    
    Args:
        data_file: 输入数据文件
        output_file: 输出数据文件
        skip_outlier: 是否跳过异常值处理
    """
    # 读取数据
    print(f"读取数据: {data_file}")
    try:
        df = pd.read_csv(data_file, encoding='utf-8')
    except UnicodeDecodeError:
        print("尝试使用不同编码...")
        df = pd.read_csv(data_file, encoding='latin1')  # 尝试其他编码
    
    print(f"数据加载完成，共有 {df.shape[0]} 行 {df.shape[1]} 列")
    
    # 检查是否已有目标变量
    if 'y_label' not in df.columns:
        print("警告: 数据中没有找到目标变量'y_label'")
    
    # 删除只有少数特定的不需要的列
    print("移除不需要的列...")
    cols_to_drop = ['prop_id_y', 'unreliable_features']  # 只删除这两列
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"移除列: {col}")
    
    # 处理缺失值，但不删除任何列
    df = handle_missing_values(df)
    
    # 创建新特征
    df = create_new_features(df)
    
    # 处理异常值
    if not skip_outlier:
        df = handle_outliers(df)
    else:
        print("跳过异常值处理")
    
    # 对文本特征进行向量化 - 必须在编码之前处理
    df, processed_text_columns = vectorize_text_features(df)
    
    # 对分类特征进行编码，但保留文本特征
    #df = encode_categorical_features(df)
    
    # 清理数据框，移除原始文本列
    #df = cleanup_dataframe(df, processed_text_columns)

    df,_ = preprocess_features(df)
    
    # 最终检查数值列的有效性
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    for col in numeric_cols:
        if col != 'y_label' and col != 'prop_id':
            # 确保没有NaN或无穷值
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
    
    # 保存结果
    print(f"保存结果到: {output_file}")
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_file, index=False)
    print(f"特征工程完成! 处理后的数据有 {df.shape[0]} 行 {df.shape[1]} 列")
    
    # 打印最终列类型信息
    print("\n最终特征类型分布:")
    print(df.dtypes.value_counts())
    
    # 确认没有非数值型列(除了prop_id和y_label)
    non_numeric_cols = [col for col in df.columns 
                      if not pd.api.types.is_numeric_dtype(df[col])
                      and col != 'prop_id' and col != 'y_label']
    if non_numeric_cols:
        print(f"\n警告: 仍有非数值列: {non_numeric_cols}")
    else:
        print("\n所有特征已成功转换为数值类型✓")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='对房价数据进行特征工程')
    parser.add_argument('--data_file', type=str, default='resources/house_samples_raw.csv',
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