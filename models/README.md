# 房产相似度模型

## 模型说明

`PropertySimilarityModel` 是一个基于KNN算法的自定义房产相似度计算模型，专门设计用于房地产估价系统。该模型通过自定义的相似度计算逻辑，筛选和排序相似房产。

## 相似房产筛选条件

模型使用以下条件筛选相似房产：

1. 在同一个council（地方政府区域）
2. 属于同一个zoning（分区）
3. 卧室数差距严格小于2
4. 都是House类别
5. 直线距离不超过1公里
6. 成交时间距现在不超过一年
7. 成交价在目标房源估价的70%-150%范围内

## 相似度得分计算

相似度得分计算公式：

```
得分 = 距离*10 + 日期差/10 + 卧室数差*5 + 卫生间差*2 + min(车位差*0.5, 1) + 
      (两者地价差的绝对值/相似成交的成交价)*10 + 房屋特征未标注分 + 
      abs(占地面积差)/目标房源占地面积*10
```

其中：
- 距离 = 坐标直线距离公里数 + 不在同一条街道得分0.2 + 不在同一suburb得分0.2 + 不在同一sector得分0.2
- 所有差异均取绝对值
- 得分越小表示相似度越高
- 目标房源自身的相似度得分设为25

## 使用方法

### 单个房产相似度计算

```python
from models.property_similarity_model import PropertySimilarityModel

# 初始化模型
model = PropertySimilarityModel()

# 为单个目标房产寻找相似房产
similar_properties = model.predict_single(
    target_property=target_property,  # 目标房产特征（pd.Series）
    properties=all_properties,        # 候选房产数据集（pd.DataFrame）
    target_price=estimated_price      # 目标房产估价
)

# 获取相似度最高的前N个房产
top_n = similar_properties.head(5)
```

### 批量房产相似度计算

```python
# 为多个目标房产寻找相似房产
similar_properties_list = model.predict(
    X=target_properties,           # 目标房产特征集（pd.DataFrame）
    target_prices=estimated_prices # 目标房产估价列表
)

# 结果是一个列表，每个元素是对应目标房产的相似房产数据框
```

## 参数设置

可以在初始化时设置以下参数：

```python
model = PropertySimilarityModel(
    n_neighbors=10,        # 默认获取的相似房产数量
    max_distance=1.0,      # 最大距离（公里）
    max_bed_diff=2,        # 最大卧室差异
    max_days=365,          # 最大天数差异
    min_price_ratio=0.7,   # 最小价格比例
    max_price_ratio=1.5,   # 最大价格比例
    self_similarity_score=25  # 目标房源自身的相似度得分
)
``` 