# 模型模块初始化文件
# 导出所有模型接口和实现

from models.model_interface import ModelInterface  
from models.model_factory import ModelFactory
from models.xgboost_model import XGBoostModel
from models.linear_model import LinearModel
from models.knn_model import KNNModel
from models.geographic_knn_model import GeographicKNNModel
from models.weighted_knn_model import WeightedKNNModel
from models.property_similarity_knn_model import PropertySimilarityKNNModel

# 导入新创建的PropertySimilarityModel
try:
    from models.property_similarity_model import PropertySimilarityModel
except ImportError:
    # 处理BaseModel可能不存在的情况
    PropertySimilarityModel = None

# 导入新添加的模型
try:
    from models.lightgbm_model import LightGBMModel
except ImportError:
    LightGBMModel = None
    
try:
    from models.catboost_model import CatBoostModel
except ImportError:
    CatBoostModel = None
    
try:
    from models.randomforest_model import RandomForestModel
except ImportError:
    RandomForestModel = None

# 导入PyTorch深度神经网络模型
try:
    from models.torch_nn_model import TorchNNModel
except ImportError:
    TorchNNModel = None

# 导入高级PyTorch深度神经网络模型
try:
    from models.torch_advanced_nn_model import TorchAdvancedNNModel
except ImportError:
    TorchAdvancedNNModel = None

__all__ = [
    'ModelInterface',
    'ModelFactory',
    'XGBoostModel', 
    'LinearModel',
    'KNNModel',
    'GeographicKNNModel',
    'WeightedKNNModel',
    'PropertySimilarityKNNModel',
    'PropertySimilarityModel',  # 添加新模型到导出列表
    'LightGBMModel',
    'CatBoostModel',
    'RandomForestModel',
    'TorchNNModel',  # 添加PyTorch深度神经网络模型
    'TorchAdvancedNNModel'  # 添加高级PyTorch深度神经网络模型
]

"""
房产估价模型模块
"""

import sys

# 设置编码
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.6及以下版本不支持reconfigure
        pass

# 移除该处ModelFactory类，使用从model_factory.py导入的类 