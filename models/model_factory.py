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
        "elasticnet": lambda **kwargs: LinearModel(model_type="elasticnet", **kwargs)
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> ModelInterface:
        """
        创建指定类型的模型
        
        Args:
            model_type: 模型类型，支持'xgboost', 'linear', 'ridge', 'lasso', 'elasticnet'
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
        从文件中加载模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # 加载模型数据
        model_data = joblib.load(path)
        
        if isinstance(model_data, dict) and "metadata" in model_data:
            # 新格式：包含模型和元数据的字典
            metadata = model_data["metadata"]
            model_type = metadata.get("model_type", "").lower()
            
            # 从model_type中提取基本类型
            if model_type.startswith("linear-"):
                base_type = "linear"
                sub_type = model_type.split("-")[1]
                # 创建对应类型的模型
                if sub_type in ["ridge", "lasso", "elasticnet"]:
                    model = LinearModel(model_type=sub_type)
                else:
                    model = LinearModel()
            elif model_type.lower() == "xgboost":
                model = XGBoostModel()
            else:
                # 尝试使用默认的XGBoost模型
                print(f"未知的模型类型: {model_type}，尝试使用XGBoost模型加载")
                model = XGBoostModel()
            
            # 设置模型属性
            model.model = model_data["model"]
            model.metadata = metadata
            model.feature_names = model_data.get("feature_names")
            
            return model
        else:
            # 旧格式：直接是模型对象
            print("警告: 使用旧格式加载模型，无法恢复完整元数据")
            # 尝试判断模型类型
            if hasattr(model_data, 'feature_importances_'):
                # 可能是XGBoost模型
                model = XGBoostModel()
                model.model = model_data
                return model
            elif hasattr(model_data, 'coef_'):
                # 可能是线性模型
                model = LinearModel()
                model.model = model_data
                return model
            else:
                raise ValueError("无法确定模型类型")
    
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