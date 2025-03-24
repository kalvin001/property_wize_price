import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Type
import os
import glob
import joblib
import json

from models.model_interface import ModelInterface
from models.xgboost_model import XGBoostModel
from models.linear_model import LinearModel 
from models.knn_model import KNNModel
from models.geographic_knn_model import GeographicKNNModel
from models.weighted_knn_model import WeightedKNNModel
from models.property_similarity_knn_model import PropertySimilarityKNNModel
from models.property_similarity_model import PropertySimilarityModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from models.randomforest_model import RandomForestModel
from models.torch_nn_model import TorchNNModel
from models.torch_advanced_nn_model import TorchAdvancedNNModel

class ModelFactory:
    """
    模型工厂类，用于创建和加载不同类型的模型
    """
    
    # 支持的模型类型
    MODEL_TYPES = {
        "xgboost": XGBoostModel,
        "linear": LinearModel,
        "ridge": lambda **kwargs: LinearModel(model_type="ridge", **kwargs),
        "lasso": lambda **kwargs: LinearModel(model_type="lasso", **kwargs),
        "knn": KNNModel,
        "geographic_knn": GeographicKNNModel,
        "weighted_knn": WeightedKNNModel,
        "property_similarity_knn": PropertySimilarityKNNModel,
        "property_similarity": PropertySimilarityModel,
        "lightgbm": LightGBMModel,
        "catboost": CatBoostModel,
        "randomforest": RandomForestModel,
        "torch_nn": TorchNNModel,
        "torch_advanced_nn": TorchAdvancedNNModel
        # "elasticnet": lambda **kwargs: LinearModel(model_type="elasticnet", **kwargs)
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> ModelInterface:
        """
        创建指定类型的模型
        
        Args:
            model_type: 模型类型，支持'xgboost', 'linear', 'ridge', 'lasso', 'elasticnet', 'knn', 'torch_nn', 'torch_advanced_nn'等
            **kwargs: 模型参数
            
        Returns:
            模型实例
        """
        model_type = model_type.lower()
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"不支持的模型类型: {model_type}，支持的类型: {list(cls.MODEL_TYPES.keys())}")
        
        model_class = cls.MODEL_TYPES[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def load_model(cls, path: str) -> ModelInterface:
        """
        加载指定路径的模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        try:
            # 首先尝试确定模型类型
            model_type = "xgboost"  # 默认类型
            
            # 尝试从文件名推断模型类型
            file_name = os.path.basename(path)
            if "linear" in file_name:
                model_type = "linear"
            elif "ridge" in file_name:
                model_type = "ridge"
            elif "lasso" in file_name:
                model_type = "lasso"
            elif "elastic" in file_name:
                model_type = "elasticnet"
            elif "geographic_knn" in file_name:
                model_type = "geographic_knn"
            elif "weighted_knn" in file_name:
                model_type = "weighted_knn"
            elif "property_similarity_knn" in file_name:
                model_type = "property_similarity_knn"
            elif "property_similarity" in file_name:
                model_type = "property_similarity"
            elif "knn" in file_name:
                model_type = "knn"
            
            # 尝试从元数据文件获取更准确的模型类型
            meta_file = os.path.splitext(path)[0] + "_meta.json"
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if "model_type" in metadata:
                        meta_type = metadata["model_type"].lower()
                        if "linear" in meta_type:
                            if "ridge" in meta_type:
                                model_type = "ridge"
                            elif "lasso" in meta_type:
                                model_type = "lasso"
                            elif "elasticnet" in meta_type:
                                model_type = "elasticnet"
                            else:
                                model_type = "linear"
                        elif "xgboost" in meta_type:
                            model_type = "xgboost"
                        elif "geographicknn" in meta_type:
                            model_type = "geographic_knn"
                        elif "weightedknn" in meta_type:
                            model_type = "weighted_knn"
                        elif "propertysimilarityknn" in meta_type:
                            model_type = "property_similarity_knn"
                        elif "propertysimilarity" in meta_type:
                            model_type = "property_similarity"
                        elif "knn" in meta_type:
                            model_type = "knn"
                except Exception as e:
                    print(f"读取元数据文件失败: {e}")
            
            # 加载模型
            # 尝试使用合适的模型类来加载
            model_class = cls.MODEL_TYPES.get(model_type)
            if model_class:
                try:
                    model = model_class.load(path)
                    # 确保model_path被正确设置
                    model.model_path = path
                    return model
                except Exception as e:
                    print(f"使用{model_type}模型类加载失败: {e}")
            
            # 如果使用特定模型类加载失败，尝试通用方法
            data = joblib.load(path)
            
            # 处理不同的数据格式
            if isinstance(data, dict) and "model" in data:
                # 创建合适的模型
                model_obj = data["model"]
                model_type_str = str(type(model_obj)).lower()
                
                if "xgboost" in model_type_str:
                    model = cls.MODEL_TYPES["xgboost"]()
                elif hasattr(model_obj, "coef_"):  # 线性模型
                    if hasattr(model_obj, "alpha") and hasattr(model_obj, "l1_ratio"):
                        model = cls.MODEL_TYPES["elasticnet"]()
                    elif hasattr(model_obj, "alpha") and not hasattr(model_obj, "l1_ratio"):
                        if hasattr(model_obj, "fit_intercept") and model_obj.fit_intercept:
                            model = cls.MODEL_TYPES["ridge"]()
                        else:
                            model = cls.MODEL_TYPES["lasso"]()
                    else:
                        model = cls.MODEL_TYPES["linear"]()
                elif "knn" in model_type_str or "neighbor" in model_type_str:
                    # 尝试判断具体的KNN模型类型
                    metadata = data.get("metadata", {})
                    model_type_meta = metadata.get("model_type", "").lower()
                    
                    if "geographic" in model_type_meta:
                        model = cls.MODEL_TYPES["geographic_knn"]()
                    elif "weighted" in model_type_meta:
                        model = cls.MODEL_TYPES["weighted_knn"]()
                    elif "propertysimilarityknn" in model_type_meta:
                        model = cls.MODEL_TYPES["property_similarity_knn"]()
                    elif "propertysimilarity" in model_type_meta:
                        model = cls.MODEL_TYPES["property_similarity"]()
                    else:
                        model = cls.MODEL_TYPES["knn"]()
                else:
                    # 默认使用XGBoost
                    model = cls.MODEL_TYPES["xgboost"]()
                
                # 设置模型属性
                model.model = data["model"]
                if "metadata" in data:
                    model._metadata = data["metadata"]
                if "feature_names" in data:
                    model._feature_names = data["feature_names"]
                
                # 确保model_path被正确设置
                model.model_path = path
                
                return model
            else:
                # 简单加载 - 兼容旧版本
                model = cls.MODEL_TYPES["xgboost"]()
                model.model = data
                model.model_path = path
                return model
                
        except Exception as e:
            raise ValueError(f"加载模型失败: {str(e)}")
    
    @classmethod
    def list_models(cls, model_dir: str = "model") -> List[Dict[str, Any]]:
        """
        列出指定目录下的所有模型
        
        Args:
            model_dir: 模型目录
            
        Returns:
            模型信息列表
        """
        # 支持的模型文件扩展名
        model_exts = [".joblib", ".pkl"]
        
        # 查找所有模型文件
        model_files = []
        for ext in model_exts:
            model_files.extend(glob.glob(os.path.join(model_dir, f"*{ext}")))
        
        # 提取模型信息
        model_info = []
        for model_file in model_files:
            try:
                # 尝试读取模型元数据
                meta_file = os.path.splitext(model_file)[0] + "_meta.json"
                if os.path.exists(meta_file):
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                        
                    model_name = os.path.basename(model_file).split(".")[0]
                    model_info.append({
                        "name": model_name,
                        "path": model_file,
                        "type": metadata.get("model_type", "未知"),
                        "metrics": metadata.get("metrics", {}),
                        "metadata": metadata
                    })
                else:
                    # 如果没有元数据文件，尝试加载模型获取信息
                    model = cls.load_model(model_file)
                    model_name = os.path.basename(model_file).split(".")[0]
                    model_info.append({
                        "name": model_name,
                        "path": model_file,
                        "type": model.metadata.get("model_type", "未知"),
                        "metrics": model.metadata.get("metrics", {}),
                        "metadata": model.metadata
                    })
            except Exception as e:
                print(f"加载模型 {model_file} 失败: {e}")
        
        return model_info 
    

if __name__ == "__main__":
    print(ModelFactory.list_models())
