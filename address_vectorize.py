import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import jieba
import re
import argparse
from pathlib import Path

# 设置编码
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.6及以下版本不支持reconfigure
        pass

def preprocess_address(address):
    """对地址文本进行预处理"""
    if isinstance(address, str):
        # 移除特殊字符，保留中文字符、英文字母和数字
        address = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', address)
        # 使用jieba进行中文分词
        words = ' '.join(jieba.cut(address))
        return words
    return ''

def vectorize_address(input_file, output_file, n_components=10):
    """
    将地址字段进行向量化处理
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        n_components: 降维后的特征数量
    """
    print(f"从 {input_file} 读取数据...")
    df = pd.read_csv(input_file)
    
    # 确保std_address列存在
    if 'std_address' not in df.columns:
        print("错误: 数据集中没有'std_address'列")
        return
    
    print(f"处理 {len(df)} 条地址...")
    
    # 预处理地址
    df['address_processed'] = df['std_address'].apply(preprocess_address)
    
    # 使用TF-IDF向量化地址
    print("使用TF-IDF向量化地址...")
    tfidf = TfidfVectorizer(
        max_features=100,  # 最多使用100个特征
        min_df=5,          # 最小文档频率
        max_df=0.9         # 最大文档频率
    )
    
    address_tfidf = tfidf.fit_transform(df['address_processed'])
    print(f"TF-IDF向量化后的特征数: {address_tfidf.shape[1]}")
    
    # 降维
    print(f"使用SVD降维到 {n_components} 个特征...")
    svd = TruncatedSVD(n_components=n_components)
    address_svd = svd.fit_transform(address_tfidf)
    
    # 添加降维后的特征到数据框
    for i in range(n_components):
        df[f'address_vec_{i}'] = address_svd[:, i]
    
    # 输出最重要的地址特征词
    feature_names = tfidf.get_feature_names_out()
    tfidf_sum = np.asarray(address_tfidf.sum(axis=0)).flatten()
    word_importance = [(word, tfidf_sum[idx]) for idx, word in enumerate(feature_names)]
    top_words = sorted(word_importance, key=lambda x: x[1], reverse=True)[:20]
    
    print("\n地址中最重要的20个特征词:")
    for word, score in top_words:
        print(f"  - {word}: {score:.4f}")
    
    # 计算特征解释方差
    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"\n{n_components}个特征解释了原始向量空间的 {explained_variance:.2%} 方差")
    
    # 删除处理中间列
    df = df.drop(columns=['address_processed'])
    
    # 保存结果
    print(f"保存结果到 {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"处理完成! 添加了 {n_components} 个地址向量特征")
    return df

def main():
    parser = argparse.ArgumentParser(description='将地址字段向量化')
    parser.add_argument('--input_file', type=str, default='resources/house_samples_engineered_noleak.csv',
                       help='输入文件路径')
    parser.add_argument('--output_file', type=str, default='resources/house_samples_engineered_with_address.csv',
                       help='输出文件路径')
    parser.add_argument('--n_components', type=int, default=10,
                       help='降维后的特征数量')
    args = parser.parse_args()
    
    vectorize_address(args.input_file, args.output_file, args.n_components)

if __name__ == "__main__":
    main() 