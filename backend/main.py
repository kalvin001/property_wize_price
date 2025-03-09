from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional, Callable, Type, TypeVar
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from pydantic_core import core_schema
from pydantic.json import pydantic_encoder
import inspect

# 添加自定义JSON编码器处理NumPy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 自定义Pydantic序列化函数
def numpy_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return pydantic_encoder(obj)  # 默认的Pydantic序列化

# 更全面的NumPy类型转换函数
def convert_numpy_types(obj):
    """递归转换所有NumPy类型为Python原生类型"""
    if obj is None:
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif pd.isna(obj):  # 处理pandas的NaN, NaT等特殊值
        return None
    elif hasattr(obj, '__dict__'):
        # 处理自定义对象
        return {k: convert_numpy_types(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_') and not inspect.ismethod(v)}
    return obj

# 创建一个自定义的BaseModel，自动处理NumPy数据类型
class NumpyBaseModel(BaseModel):
    """扩展的基础模型，自动处理NumPy数据类型"""
    
    @classmethod
    def _convert_numpy_types(cls, v):
        return convert_numpy_types(v)
    
    @field_validator('*')
    @classmethod
    def validate_numpy_types(cls, v):
        return cls._convert_numpy_types(v)
    
    model_config = {
        'arbitrary_types_allowed': True,
        'json_encoders': {
            np.integer: lambda v: int(v),
            np.floating: lambda v: float(v),
            np.ndarray: lambda v: v.tolist()
        }
    }

    # 重写model_dump方法以确保NumPy类型被正确处理
    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        return convert_numpy_types(data)

    # 重写dict方法以确保NumPy类型被正确处理
    def dict(self, **kwargs):
        if hasattr(super(), 'dict'):
            data = super().dict(**kwargs)
            return convert_numpy_types(data)
        else:  # 兼容Pydantic v2
            data = self.model_dump(**kwargs)
            return convert_numpy_types(data)

app = FastAPI(title="房产估价API")

# 配置应用JSON序列化
import fastapi.encoders
# 保存原始的jsonable_encoder
original_jsonable_encoder = fastapi.encoders.jsonable_encoder

# 创建一个包装函数，处理NumPy类型
def numpy_jsonable_encoder(obj, *args, **kwargs):
    """确保所有NumPy类型都被转换为Python原生类型"""
    # 预处理对象，转换NumPy类型
    obj = convert_numpy_types(obj)
    
    custom_encoder = kwargs.get('custom_encoder', {})
    # 添加NumPy类型处理
    numpy_encoders = {
        np.integer: lambda v: int(v),
        np.floating: lambda v: float(v),
        np.ndarray: lambda v: v.tolist()
    }
    # 合并编码器
    custom_encoder.update(numpy_encoders)
    kwargs['custom_encoder'] = custom_encoder
    # 调用原始函数
    return original_jsonable_encoder(obj, *args, **kwargs)

# 替换编码器
fastapi.encoders.jsonable_encoder = numpy_jsonable_encoder

# 直接覆盖Pydantic序列化函数
import pydantic.json
original_pydantic_encoder = pydantic.json.pydantic_encoder

def patched_pydantic_encoder(obj, **kwargs):
    """确保所有NumPy类型都被转换为Python原生类型"""
    # 预处理对象，转换NumPy类型
    obj = convert_numpy_types(obj)
    # 调用原始编码器
    return original_pydantic_encoder(obj, **kwargs)

# 替换Pydantic编码器
pydantic.json.pydantic_encoder = patched_pydantic_encoder

# 直接修补fastapi.routing.serialize_response函数
import fastapi.routing
original_serialize_response = fastapi.routing.serialize_response

async def patched_serialize_response(
    field, response_content, include=None, exclude=None, by_alias=True, exclude_unset=False, **kwargs
):
    """确保所有FastAPI响应在序列化前都经过NumPy类型转换"""
    # 预处理响应内容，转换NumPy类型
    response_content = convert_numpy_types(response_content)
    # 调用原始函数
    return await original_serialize_response(
        field, response_content, include, exclude, by_alias, exclude_unset, **kwargs
    )

# 替换序列化函数
fastapi.routing.serialize_response = patched_serialize_response

# 修补pandas.Series.__getitem__方法，确保返回Python原生类型
original_series_getitem = pd.Series.__getitem__

def patched_series_getitem(self, key):
    value = original_series_getitem(self, key)
    # 转换NumPy类型为Python原生类型
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    return value

# 应用补丁
pd.Series.__getitem__ = patched_series_getitem

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应当设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加自定义响应类，使用NumpyEncoder处理NumPy类型
class NumpyJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=True,
            indent=None,
            separators=(",", ":"),
            cls=NumpyEncoder,
        ).encode("utf-8")

# 设置默认响应类
app.json_response_class = NumpyJSONResponse

# 数据模型
class PropertyFeatures(NumpyBaseModel):
    features: Dict[str, Any]

class PredictionResult(NumpyBaseModel):
    predicted_price: float
    feature_importance: List[Dict[str, Any]]

# 全局变量
MODEL_PATH = Path("../model/xgb_model.joblib")  # 更新为XGBoost模型路径
# 备选路径，以防主路径不工作
ALTERNATE_MODEL_PATHS = [
    Path("model/xgb_model.joblib"),
    Path("./model/xgb_model.joblib"),
    Path("../../model/xgb_model.joblib"),
    Path(os.path.abspath("../model/xgb_model.joblib"))
]
FEATURE_COLS = []
MODEL = None

# 全局变量，用于存储房产数据
PROPERTIES_DF = None

# 重命名现有的PredictionResult类，以便不冲突
class ModelPredictionResult(NumpyBaseModel):
    predicted_price: float
    feature_importance: List[Dict[str, Any]]

# 添加新的房产相关模型
class Property(NumpyBaseModel):
    prop_id: str
    address: str
    predicted_price: float = 0
    features: Dict[str, Any]

class PropertyDetail(Property):
    feature_importance: List[Dict[str, Any]] = []
    comparable_properties: List[Dict[str, Any]] = []
    price_trends: List[Dict[str, Any]] = []  # 历史价格趋势
    price_range: Dict[str, float] = {}  # 价格预测区间
    neighborhood_stats: Dict[str, Any] = {}  # 周边区域统计
    confidence_interval: Dict[str, float] = {}  # 置信区间
    ai_explanation: Dict[str, Any] = {}  # 更详细的模型解释（重命名避免冲突）
    
    model_config = {
        'protected_namespaces': ()  # 禁用保护命名空间检查
    }

class PropertyListResponse(NumpyBaseModel):
    total: int
    page: int
    page_size: int
    properties: List[Property]

@app.on_event("startup")
async def startup_event():
    global MODEL, FEATURE_COLS, PROPERTIES_DF
    
    try:
        # 加载模型和特征列
        model_path = Path("../model/xgb_model.joblib")
        feature_cols_path = Path("../model/feature_cols.joblib")
        
        if model_path.exists() and feature_cols_path.exists():
            MODEL = joblib.load(model_path)
            FEATURE_COLS = joblib.load(feature_cols_path)
            print(f"模型已加载，特征数量: {len(FEATURE_COLS)}")
            
            # 设置XGBoost的参数支持分类特征
            if hasattr(MODEL, '_Booster'):
                MODEL._Booster.set_param('enable_categorical', 'true')
                print("已启用XGBoost分类特征支持")
        else:
            print("模型文件不存在，无法加载模型")
            
        # 尝试加载房产数据
        try:
            csv_path = Path("../resources/house_samples_features.csv")
            if csv_path.exists():
                PROPERTIES_DF = pd.read_csv(csv_path)
                print(f"房产数据已加载，共 {len(PROPERTIES_DF)} 条记录")
            else:
                print("房产数据文件不存在")
        except Exception as e:
            print(f"加载房产数据出错: {e}")
            
    except Exception as e:
        print(f"启动时出错: {e}")

@app.get("/")
async def root():
    return {"message": "房产估价API已启动"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}

@app.get("/api/model/info")
async def model_info():
    """获取模型信息"""
    if MODEL is None:
        raise HTTPException(status_code=404, detail="模型尚未加载")
    
    try:
        # 尝试从公共数据目录加载模型指标
        metrics_path = Path("../frontend/public/data/model_metrics.json")
        data_info_path = Path("../frontend/public/data/data_info.json")
        
        model_metrics = {}
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                model_metrics = json.load(f)
                
            # 注意：现在所有指标都已在analyze_data.py中计算，不需要在这里重新计算
            # 如果后续有新指标需要添加，可以在这里计算
            
            # 如果没有特征重要性数据，但模型有feature_importances_属性，则添加
            if "feature_importance" not in model_metrics and MODEL is not None and hasattr(MODEL, 'feature_importances_'):
                feature_importances = MODEL.feature_importances_
                sorted_idx = np.argsort(feature_importances)[::-1]
                top_features = []
                
                for i in sorted_idx[:20]:  # 取前20个最重要的特征
                    if i < len(FEATURE_COLS):
                        top_features.append({
                            "feature": FEATURE_COLS[i],
                            "importance": float(feature_importances[i])
                        })
                        
                model_metrics["feature_importance"] = top_features
        
        data_info = {}
        if data_info_path.exists():
            with open(data_info_path, 'r') as f:
                data_info = json.load(f)
        
        return {
            "model_type": "XGBRegressor",  # 更新为XGBoost
            "features_count": len(FEATURE_COLS),
            "feature_names": FEATURE_COLS[:10] + ["..."] if len(FEATURE_COLS) > 10 else FEATURE_COLS,
            "metrics": model_metrics,
            "data_info": data_info
        }
    except Exception as e:
        return {
            "model_type": "XGBRegressor",  # 更新为XGBoost
            "features_count": len(FEATURE_COLS),
            "error": str(e)
        }

@app.get("/api/properties/sample")
async def get_sample_properties():
    """获取样本房产及其分析"""
    sample_path = Path("../frontend/public/data/sample_properties.json")
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="样本房产数据不存在")
    
    with open(sample_path, 'r') as f:
        sample_properties = json.load(f)
    
    return sample_properties

@app.post("/api/predict")
async def predict_price(property_data: PropertyFeatures):
    """预测房价"""
    if MODEL is None:
        raise HTTPException(status_code=404, detail="模型尚未加载")
    
    try:
        # 将输入特征转换为DataFrame
        features_df = pd.DataFrame([property_data.features])
        
        # 确保所有特征列都存在
        for col in FEATURE_COLS:
            if col not in features_df.columns:
                features_df[col] = 0  # 如果特征不存在，用0填充
        
        # 只保留模型使用的特征，并按正确顺序排列
        features_df = features_df[FEATURE_COLS]
        
        # 确保数据类型正确
        for col in features_df.columns:
            if features_df[col].dtype == 'object':
                # 尝试转换为数值型
                try:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                except:
                    # 如果无法转换为数值，将其转换为分类型
                    features_df[col] = features_df[col].astype('category')
            # 处理日期类型
            elif pd.api.types.is_datetime64_any_dtype(features_df[col]):
                print(f"属性列表预测时检测到日期类型列: {col}，将转换为数值")
                # 将日期转换为时间戳（从1970-01-01起的天数）
                features_df[col] = (features_df[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1 day")
        
        # 再次检查所有列的类型
        cols_to_drop = []
        for col in features_df.columns:
            if not (pd.api.types.is_numeric_dtype(features_df[col]) or 
                    pd.api.types.is_bool_dtype(features_df[col]) or 
                    pd.api.types.is_categorical_dtype(features_df[col])):
                print(f"警告: 属性列表预测前列 {col} 类型 {features_df[col].dtype} 不被模型支持，将移除")
                cols_to_drop.append(col)
        
        # 移除不支持的列
        if cols_to_drop:
            features_df = features_df.drop(columns=cols_to_drop)
        
        # 使用模型预测
        predicted_price = float(MODEL.predict(features_df)[0])
        
        # 计算特征重要性
        feature_importance = []
        if hasattr(MODEL, 'feature_importances_'):
            importances = MODEL.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = min(10, len(FEATURE_COLS))  # 最多返回前10个重要特征
            
            for i in range(top_n):
                idx = indices[i]
                if idx < len(FEATURE_COLS):
                    feature = FEATURE_COLS[idx]
                    value = features_df[feature].values[0]
                    importance = float(importances[idx])
                    feature_importance.append({
                        "feature": feature,
                        "value": float(value) if isinstance(value, (np.integer, np.floating)) else value if isinstance(value, (int, float, str, bool)) else str(value),
                        "importance": importance,
                        "effect": "positive" if importance > 0 else "negative"
                    })
        
        return ModelPredictionResult(
            predicted_price=predicted_price,
            feature_importance=feature_importance
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测过程中出错: {str(e)}")

# 添加获取房产列表的API接口
@app.get("/api/properties", response_model=PropertyListResponse)
async def get_properties(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    property_type: Optional[str] = None,
    query: Optional[str] = None
):
    """获取房产列表，支持分页和筛选"""
    if PROPERTIES_DF is None:
        raise HTTPException(status_code=404, detail="房产数据尚未加载")
    
    try:
        # 创建过滤后的数据副本
        filtered_df = PROPERTIES_DF.copy()
        
        # 应用过滤条件
        if property_type and 'prop_type' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['prop_type'] == property_type]
            
        if query:
            # 搜索地址或ID
            query_lower = query.lower()
            address_mask = filtered_df['std_address'].str.lower().str.contains(query_lower, na=False)
            id_mask = filtered_df['prop_id'].astype(str).str.contains(query_lower, na=False)
            filtered_df = filtered_df[address_mask | id_mask]
        
        # 计算总数
        total = len(filtered_df)
        
        # 应用分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged_df = filtered_df.iloc[start_idx:end_idx]
        
        # 构造响应
        properties = []
        for _, row in paged_df.iterrows():
            # 实际的y_label价格
            actual_price = float(row.get('y_label', 0))
            
            # 计算房价预测
            pred_price = 0
            if MODEL is not None and set(FEATURE_COLS).issubset(filtered_df.columns):
                try:
                    features = row[FEATURE_COLS]
                    # 使用模型预测前，确保所有特征数据类型正确
                    features_df = features.to_frame().T
                    
                    # 转换数据类型 - 将object类型转为数值型
                    for col in features_df.columns:
                        if features_df[col].dtype == 'object':
                            # 尝试转换为数值型
                            try:
                                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                            except:
                                # 如果无法转换为数值，将其转换为分类型
                                features_df[col] = features_df[col].astype('category')
                        # 处理日期类型
                        elif pd.api.types.is_datetime64_any_dtype(features_df[col]):
                            print(f"属性列表预测时检测到日期类型列: {col}，将转换为数值")
                            # 将日期转换为时间戳（从1970-01-01起的天数）
                            features_df[col] = (features_df[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1 day")
                    
                    # 再次检查所有列的类型
                    cols_to_drop = []
                    for col in features_df.columns:
                        if not (pd.api.types.is_numeric_dtype(features_df[col]) or 
                                pd.api.types.is_bool_dtype(features_df[col]) or 
                                pd.api.types.is_categorical_dtype(features_df[col])):
                            print(f"警告: 属性列表预测前列 {col} 类型 {features_df[col].dtype} 不被模型支持，将移除")
                            cols_to_drop.append(col)
                    
                    # 移除不支持的列
                    if cols_to_drop:
                        features_df = features_df.drop(columns=cols_to_drop)
                    
                    # 使用模型预测
                    pred_price = float(MODEL.predict(features_df)[0])
                     
                except Exception as e:
                    print(f"预测异常: {e}") 
            else:  
                print(f"模型不可用，使用随机预测: {pred_price}")
            
            # 创建属性字典
            features = {}
            for col in row.index:
                if col not in ['prop_id', 'std_address']:
                    val = row[col]
                    features[col] = float(val) if isinstance(val, (int, float)) and not np.isnan(val) else (
                        str(val) if not pd.isna(val) else None
                    )
            
            properties.append(Property(
                prop_id=str(row['prop_id']),
                address=row['std_address'],
                predicted_price=pred_price,
                features=features
            ))
        
        response = PropertyListResponse(
            total=total,
            page=page,
            page_size=page_size,
            properties=convert_numpy_types(properties)
        )
        
        # 转换为字典并确保所有NumPy类型都被转换为Python原生类型
        return convert_numpy_types(response.model_dump())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取房产列表出错: {str(e)}")

# 添加获取单个房产详情的API接口
@app.get("/api/properties/{prop_id}", response_model=PropertyDetail)
async def get_property_detail(prop_id: str):
    """获取单个房产的详情"""
    if PROPERTIES_DF is None:
        raise HTTPException(status_code=404, detail="房产数据尚未加载")
    
    try:
        # 查找指定ID的房产
        prop_df = PROPERTIES_DF[PROPERTIES_DF['prop_id'].astype(str) == prop_id]
        
        if len(prop_df) == 0:
            raise HTTPException(status_code=404, detail=f"找不到ID为 {prop_id} 的房产")
        
        # 获取第一个匹配的房产
        row = prop_df.iloc[0]
        
        # 使用price_analysis模块计算预测价格和特征重要性
        from price_analysis import (
            predict_property_price, 
            find_comparable_properties,
            generate_price_trends,
            calculate_price_range,
            get_neighborhood_stats,
            calculate_confidence_interval,
            get_model_explanation
        )
        
        # 导入备用数据生成函数
        from price_utils import generate_rule_based_importance, generate_dummy_comparables
        
        # 预测价格和计算特征重要性
        try:
            pred_price, feature_importance = predict_property_price(
                row=row, 
                model=MODEL, 
                feature_cols=FEATURE_COLS, 
                properties_df=PROPERTIES_DF
            )
        except Exception as e:
            print(f"预测价格和计算特征重要性时出错: {e}")
            # 使用实际价格或默认值
            if 'y_label' in row and pd.notna(row['y_label']):
                try:
                    pred_price = float(row['y_label'])
                except:
                    pred_price = 750000
            else:
                pred_price = 750000
                
            # 使用规则生成特征重要性
            feature_importance = generate_rule_based_importance(
                row=row, 
                feature_cols=FEATURE_COLS if FEATURE_COLS else [], 
                pred_price=pred_price
            )
        
        # 查找可比房产
        try:
            comparable_properties = find_comparable_properties(
                row=row,
                prop_id=prop_id,
                properties_df=PROPERTIES_DF
            )
            
            # 确保我们至少有一些可比房产
            if not comparable_properties:
                comparable_properties = generate_dummy_comparables(row, prop_id)
        except Exception as e:
            print(f"查找可比房产时出错: {e}")
            # 生成虚拟可比房产
            comparable_properties = generate_dummy_comparables(row, prop_id)
        
        # 创建属性字典
        features = {}
        for col in row.index:
            if col not in ['prop_id', 'std_address']:
                val = row[col]
                features[col] = float(val) if isinstance(val, (int, float)) and not np.isnan(val) else (
                    str(val) if not pd.isna(val) else None
                )
        
        # 房产面积
        prop_area = float(row.get('prop_area', 100))
        
        try:
            property_detail = PropertyDetail(
                prop_id=str(row['prop_id']),
                address=row['std_address'],
                predicted_price=pred_price,
                features=features,
                feature_importance=feature_importance,
                comparable_properties=comparable_properties,
                # 添加历史价格趋势数据
                price_trends=generate_price_trends(pred_price),
                # 添加价格预测区间
                price_range=calculate_price_range(pred_price),
                # 添加周边区域统计
                neighborhood_stats=get_neighborhood_stats(pred_price, prop_area, row, PROPERTIES_DF),
                # 添加置信区间
                confidence_interval=calculate_confidence_interval(pred_price),
                # 添加更详细的模型解释
                ai_explanation=get_model_explanation(pred_price, feature_importance, FEATURE_COLS)
            )
            
            # 检查并确保feature_importance不为空
            if not property_detail.feature_importance:
                property_detail.feature_importance = generate_rule_based_importance(
                    row=row, 
                    feature_cols=FEATURE_COLS if FEATURE_COLS else [],
                    pred_price=pred_price
                )
            
            # 检查并确保comparable_properties不为空
            if not property_detail.comparable_properties:
                property_detail.comparable_properties = generate_dummy_comparables(row, prop_id)
                
            # 转换为字典并确保所有NumPy类型都被转换为Python原生类型
            return convert_numpy_types(property_detail.model_dump())
            
        except Exception as e:
            print(f"创建PropertyDetail对象时出错: {e}")
            # 创建最基本的PropertyDetail对象
            property_detail = PropertyDetail(
                prop_id=str(row['prop_id']),
                address=row['std_address'],
                predicted_price=pred_price,
                features=features,
                feature_importance=generate_rule_based_importance(row, FEATURE_COLS if FEATURE_COLS else [], pred_price),
                comparable_properties=generate_dummy_comparables(row, prop_id),
                price_trends=generate_price_trends(pred_price),
                price_range=calculate_price_range(pred_price),
                neighborhood_stats=get_neighborhood_stats(pred_price, prop_area),
                confidence_interval=calculate_confidence_interval(pred_price),
                ai_explanation=get_model_explanation(pred_price, [], FEATURE_COLS if FEATURE_COLS else [])
            )
            
            # 转换为字典并确保所有NumPy类型都被转换为Python原生类型
            return convert_numpy_types(property_detail.model_dump())
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取房产详情出错: {str(e)}")

# Vercel部署支持
from mangum import Mangum
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True) 
##
#python main.py