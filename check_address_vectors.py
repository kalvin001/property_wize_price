import pandas as pd
import numpy as np

# 读取带有地址向量特征的数据集
df = pd.read_csv('resources/house_samples_engineered_with_address.csv')

# 找出地址向量特征列
address_cols = [col for col in df.columns if 'address_vec' in col]
print(f'生成的地址向量特征数: {len(address_cols)}')
print(f'地址向量特征列: {address_cols}')

if len(address_cols) > 0:
    # 随机选择几个样本进行展示
    sample_indices = np.random.choice(len(df), 3, replace=False)
    for i, idx in enumerate(sample_indices):
        vector_vals = df.iloc[idx][address_cols].values
        address = df.iloc[idx]['std_address']
        print(f'\n样本 {i+1}:')
        print(f'地址: {address}')
        print(f'向量: {vector_vals}')
    
    # 检查地址向量特征的方差
    print('\n地址向量特征方差:')
    for col in address_cols:
        variance = df[col].var()
        print(f'{col}: {variance:.6f}')
    
    # 保存向量特征列名到文件
    with open('address_vector_columns.txt', 'w') as f:
        f.write('\n'.join(address_cols))
    print(f'\n地址向量特征列名已保存到 address_vector_columns.txt')
else:
    print('没有找到地址向量特征!') 