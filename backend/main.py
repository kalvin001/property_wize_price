from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel, field_validator, ConfigDict
from typing import List, Dict, Any, Optional, Callable, Type, TypeVar
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from pydantic_core import core_schema
from pydantic.json import pydantic_encoder
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import platform
import sys
from contextlib import asynccontextmanager
import glob
import math

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    # 导入模型接口和工厂
    from models import ModelFactory, ModelInterface
    MODELS_AVAILABLE = True
    print("成功导入模型模块")
except ImportError:
    MODELS_AVAILABLE = False
    print("未找到模型模块，将使用传统方式加载模型")

# 导入项目管理API
try:
    from .project_api import init_project_api
    PROJECT_API_AVAILABLE = True
    print("成功导入项目管理模块")
except ImportError:
    PROJECT_API_AVAILABLE = False
    print("未找到项目管理模块，将不会启用项目管理功能")

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

# 创建一个自定义的BaseModel，自动处理NumPy数据类型
class NumpyBaseModel(BaseModel):
    """扩展的基础模型，自动处理NumPy数据类型"""
    
    @classmethod
    def _convert_numpy_types(cls, v):
        if isinstance(v, np.integer):
            return int(v)
        elif isinstance(v, np.floating):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, dict):
            return {k: cls._convert_numpy_types(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [cls._convert_numpy_types(item) for item in v]
        return v
    
    @field_validator('*')
    @classmethod
    def validate_numpy_types(cls, v):
        return cls._convert_numpy_types(v)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.integer: lambda v: int(v),
            np.int64: lambda v: int(v),     # 针对 np.int64 的序列化
            np.floating: lambda v: float(v),
            np.float64: lambda v: float(v),   # 针对 np.float64 的序列化
            np.ndarray: lambda v: v.tolist()
        }
    )

# 数据模型
class PropertyFeatures(NumpyBaseModel):
    features: Dict[str, Any]

class PredictionResult(NumpyBaseModel):
    predicted_price: float
    feature_importance: List[Dict[str, Any]]

# 全局变量
# 将相对路径改为使用绝对路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'model')
MODEL_PATH = Path(os.path.join(model_dir, "xgb_model.joblib"))
print(f"模型路径设置为: {MODEL_PATH}")

# 直接尝试加载已知存在的模型文件
try:
    print(f"直接尝试加载模型: {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)
    print(f"成功直接加载模型: {type(MODEL)}")
except Exception as e:
    print(f"直接加载模型失败，错误: {str(e)}")
    MODEL = None

# 备选路径，以防主路径不工作
ALTERNATE_MODEL_PATHS = [
    Path(os.path.join(model_dir, "xgb_model.joblib")),
    Path(os.path.join(model_dir, "xgboost_model.joblib")),
    Path(os.path.join(os.getcwd(), "model", "xgb_model.joblib")),
    Path(os.path.join(os.path.dirname(os.getcwd()), "model", "xgb_model.joblib"))
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

# 模型训练和管理相关的数据模型
class ModelTrainingRequest(NumpyBaseModel):
    model_type: str
    params: Dict[str, Any] = {}
    test_size: float = 0.2
    random_state: int = 42

class ModelTrainingResponse(NumpyBaseModel):
    success: bool
    message: str
    metrics: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None
    
class ModelListResponse(NumpyBaseModel):
    models: List[Dict[str, Any]]

# 使用新的lifespan替代弃用的on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用的生命周期管理"""
    global TRAINED_MODEL, SAMPLE_DATA, COLUMN_DESCRIPTIONS, RAW_DATA, MODEL, FEATURE_COLS, PROPERTIES_DF
    
    print("=== 开始应用生命周期管理 ===")
    
    # 将相对路径转换为绝对路径，确保跨平台一致性
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'model')
    
    # 尝试修复模型文件权限问题
    source_model_path = os.path.join(base_dir, 'model', 'xgb_model.joblib')
    backup_model_path = os.path.join(base_dir, 'model', 'xgb_model_backup.joblib')
    
    if os.path.exists(source_model_path):
        try:
            print(f"尝试创建模型文件备份，从 {source_model_path} 到 {backup_model_path}")
            import shutil
            shutil.copy2(source_model_path, backup_model_path)
            print("成功创建模型备份")
            
            # 尝试直接加载备份
            try:
                print(f"尝试加载模型备份: {backup_model_path}")
                model_data = joblib.load(backup_model_path)
                if isinstance(model_data, dict) and "model" in model_data:
                    MODEL = model_data["model"]
                    print(f"从备份的字典结构加载模型成功: {type(MODEL)}")
                else:
                    MODEL = model_data
                    print(f"从备份直接加载模型成功: {type(MODEL)}")
                
                TRAINED_MODEL = MODEL
            except Exception as e:
                print(f"加载模型备份失败: {str(e)}")
        except Exception as e:
            print(f"创建模型备份失败: {str(e)}")
    
    # 尝试使用pickle直接加载
    try:
        import pickle
        print("尝试使用pickle加载模型...")
        with open(source_model_path, 'rb') as f:
            pickle_model = pickle.load(f)
            print(f"使用pickle成功加载模型: {type(pickle_model)}")
            if MODEL is None:
                MODEL = pickle_model
                TRAINED_MODEL = MODEL
    except Exception as e:
        print(f"使用pickle加载失败: {str(e)}")
    
    # 注册项目管理API
    if PROJECT_API_AVAILABLE:
        init_project_api(app)
        print("已注册项目管理API路由")
    
    # 确保模型目录存在
    os.makedirs(model_dir, exist_ok=True)
    print(f"模型目录路径: {model_dir}")
    
    # 智能查找XGBoost模型文件
    xgb_model_file = None
    possible_names = [
        "xgb_model.joblib", 
        "XGBRegressor_model.joblib", 
        "xgboost_model.joblib",
        "model.joblib"
    ]
    
    # 首先尝试预定义的可能名称
    for name in possible_names:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            xgb_model_file = path
            print(f"找到模型文件: {path}")
            break
    
    # 如果没有找到，搜索目录中所有joblib文件
    if not xgb_model_file and os.path.exists(model_dir):
        print("未找到预期名称的模型文件，搜索目录中的所有joblib文件...")
        for file in os.listdir(model_dir):
            if file.endswith(".joblib") and ("xgb" in file.lower() or "boost" in file.lower()):
                xgb_model_file = os.path.join(model_dir, file)
                print(f"找到可能匹配的模型文件: {xgb_model_file}")
                break
    
    # 优先使用ModelFactory获取模型
    if MODELS_AVAILABLE:
        try:
            # 检查ModelFactory是否有get_active_model方法
            if hasattr(ModelFactory, 'get_active_model'):
                # 尝试从模型工厂获取活跃模型
                TRAINED_MODEL = ModelFactory.get_active_model()
                if TRAINED_MODEL:
                    print(f"已从模型工厂加载活跃模型: {TRAINED_MODEL.model_type}")
                    MODEL = TRAINED_MODEL
                else:
                    # 回退到传统加载方式
                    if xgb_model_file and os.path.exists(xgb_model_file):
                        try:
                            print(f"尝试加载模型文件: {xgb_model_file}")
                            model_data = joblib.load(xgb_model_file)
                            
                            # 处理模型数据可能的不同格式
                            if isinstance(model_data, dict) and "model" in model_data:
                                MODEL = model_data["model"]
                                print(f"从字典结构加载模型成功")
                                if "feature_names" in model_data:
                                    FEATURE_COLS = model_data["feature_names"]
                                    print(f"从模型数据加载特征列，共{len(FEATURE_COLS)}列")
                            else:
                                MODEL = model_data
                                print("直接加载模型成功")
                            
                            TRAINED_MODEL = MODEL
                            print(f"已从{xgb_model_file}加载模型")
                        except Exception as load_error:
                            print(f"加载模型文件出错: {str(load_error)}")
                    else:
                        print(f"警告: 找不到模型文件")
            else:
                # 如果没有get_active_model方法，尝试使用list_models方法
                print("ModelFactory没有get_active_model方法，尝试获取最新模型")
                try:
                    # 获取所有模型并选择最新的一个
                    models = ModelFactory.list_models(model_dir)
                    if models and len(models) > 0:
                        # 按创建时间排序
                        sorted_models = sorted(models, key=lambda x: os.path.getmtime(x["path"]) if "path" in x else 0, reverse=True)
                        if sorted_models:
                            latest_model_path = sorted_models[0]["path"]
                            MODEL = ModelFactory.load_model(latest_model_path)
                            TRAINED_MODEL = MODEL
                            print(f"已加载最新模型: {latest_model_path}")
                        else:
                            print("没有找到可用模型")
                    else:
                        # 回退到传统加载方式
                        if xgb_model_file and os.path.exists(xgb_model_file):
                            try:
                                print(f"尝试加载模型文件: {xgb_model_file}")
                                model_data = joblib.load(xgb_model_file)
                                
                                # 处理模型数据可能的不同格式
                                if isinstance(model_data, dict) and "model" in model_data:
                                    MODEL = model_data["model"]
                                    print(f"从字典结构加载模型成功")
                                    if "feature_names" in model_data:
                                        FEATURE_COLS = model_data["feature_names"]
                                        print(f"从模型数据加载特征列，共{len(FEATURE_COLS)}列")
                                else:
                                    MODEL = model_data
                                    print("直接加载模型成功")
                                
                                TRAINED_MODEL = MODEL
                                print(f"已从{xgb_model_file}加载模型")
                            except Exception as load_error:
                                print(f"加载模型文件出错: {str(load_error)}")
                        else:
                            print(f"警告: 找不到模型文件")
                except Exception as list_error:
                    print(f"尝试获取模型列表失败: {str(list_error)}")
                    # 回退到传统加载方式
                    if xgb_model_file and os.path.exists(xgb_model_file):
                        try:
                            print(f"尝试加载模型文件: {xgb_model_file}")
                            model_data = joblib.load(xgb_model_file)
                            
                            # 处理模型数据可能的不同格式
                            if isinstance(model_data, dict) and "model" in model_data:
                                MODEL = model_data["model"]
                                print(f"从字典结构加载模型成功")
                                if "feature_names" in model_data:
                                    FEATURE_COLS = model_data["feature_names"]
                                    print(f"从模型数据加载特征列，共{len(FEATURE_COLS)}列")
                            else:
                                MODEL = model_data
                                print("直接加载模型成功")
                            
                            TRAINED_MODEL = MODEL
                            print(f"已从{xgb_model_file}加载模型")
                        except Exception as load_error:
                            print(f"加载模型文件出错: {str(load_error)}")
                    else:
                        print(f"警告: 找不到模型文件")
        except Exception as e:
            print(f"加载模型出错: {str(e)}")
            # 回退到传统加载方式
            if xgb_model_file and os.path.exists(xgb_model_file):
                try:
                    print(f"尝试加载模型文件: {xgb_model_file}")
                    model_data = joblib.load(xgb_model_file)
                    
                    # 处理模型数据可能的不同格式
                    if isinstance(model_data, dict) and "model" in model_data:
                        MODEL = model_data["model"]
                        print(f"从字典结构加载模型成功")
                        if "feature_names" in model_data:
                            FEATURE_COLS = model_data["feature_names"]
                            print(f"从模型数据加载特征列，共{len(FEATURE_COLS)}列")
                    else:
                        MODEL = model_data
                        print("直接加载模型成功")
                    
                    TRAINED_MODEL = MODEL
                    print(f"已从{xgb_model_file}加载模型")
                except Exception as load_error:
                    print(f"加载模型文件出错: {str(load_error)}")
            else:
                print(f"警告: 找不到模型文件")
    else:
        # 如果模型模块不可用，尝试直接加载模型文件
        if xgb_model_file and os.path.exists(xgb_model_file):
            try:
                print(f"模型模块不可用，直接尝试加载模型文件: {xgb_model_file}")
                model_data = joblib.load(xgb_model_file)
                
                # 处理模型数据可能的不同格式
                if isinstance(model_data, dict) and "model" in model_data:
                    MODEL = model_data["model"]
                    print(f"从字典结构加载模型成功")
                    if "feature_names" in model_data:
                        FEATURE_COLS = model_data["feature_names"]
                        print(f"从模型数据加载特征列，共{len(FEATURE_COLS)}列")
                else:
                    MODEL = model_data
                    print("直接加载模型成功")
                
                TRAINED_MODEL = MODEL
                print(f"已从{xgb_model_file}加载模型")
            except Exception as load_error:
                print(f"加载模型文件出错: {str(load_error)}")
        else:
            print(f"警告: 找不到模型文件")
    
    # 尝试加载特征列
    try:
        feature_cols_path = os.path.join(model_dir, 'feature_cols.joblib')
        if os.path.exists(feature_cols_path) and not FEATURE_COLS:
            FEATURE_COLS = joblib.load(feature_cols_path)
            print(f"已从{feature_cols_path}加载特征列，共{len(FEATURE_COLS)}列")
        else:
            if not FEATURE_COLS:
                print(f"警告: 未找到特征列文件 {feature_cols_path}")
                FEATURE_COLS = []
    except Exception as e:
        print(f"加载特征列出错: {str(e)}")
        FEATURE_COLS = []
    
    # 加载房产数据
    try:
        # 尝试加载属性数据
        properties_paths = [
            os.path.join('../resources', 'house_samples_features.csv'),
            os.path.join('resources', 'house_samples_features.csv'),
            os.path.join('../data', 'house_samples_features.csv'),
            os.path.join('data', 'house_samples_features.csv'),
            os.path.join('data', 'properties.csv'),
            os.path.join('../data', 'properties.csv'),
            os.path.join('data', 'processed_data.csv'),
            os.path.join('../data', 'processed_data.csv')
        ]
        
        for path in properties_paths:
            if os.path.exists(path):
                print(f"尝试从{path}加载房产数据...")
                PROPERTIES_DF = pd.read_csv(path)
                print(f"成功加载房产数据，共{len(PROPERTIES_DF)}条记录")
                
                # 确保房产数据包含必要的列
                required_cols = ['prop_id', 'std_address']
                if 'prop_id' not in PROPERTIES_DF.columns:
                    # 尝试创建一个prop_id列
                    if 'id' in PROPERTIES_DF.columns:
                        PROPERTIES_DF['prop_id'] = PROPERTIES_DF['id']
                    else:
                        PROPERTIES_DF['prop_id'] = [f"PROP_{i}" for i in range(len(PROPERTIES_DF))]
                    print("已添加prop_id列")
                
                if 'std_address' not in PROPERTIES_DF.columns:
                    # 尝试创建一个std_address列
                    if 'address' in PROPERTIES_DF.columns:
                        PROPERTIES_DF['std_address'] = PROPERTIES_DF['address']
                    else:
                        PROPERTIES_DF['std_address'] = [f"地址_{i}" for i in range(len(PROPERTIES_DF))]
                    print("已添加std_address列")
                
                break
        else:
            # 如果没有找到任何数据文件，创建一个空的DataFrame
            print("警告: 未找到房产数据文件，将创建一个示例数据集")
            PROPERTIES_DF = pd.DataFrame({
                'prop_id': [f"SAMPLE_{i}" for i in range(10)],
                'std_address': [f"示例地址_{i}" for i in range(10)],
                'price': np.random.randint(500000, 2000000, 10),
                'bedrooms': np.random.randint(1, 6, 10),
                'bathrooms': np.random.randint(1, 4, 10),
                'area': np.random.randint(80, 300, 10),
                'year_built': np.random.randint(1980, 2020, 10)
            })
    except Exception as e:
        print(f"加载房产数据出错: {str(e)}")
        # 创建一个空的DataFrame
        PROPERTIES_DF = pd.DataFrame({
            'prop_id': [f"SAMPLE_{i}" for i in range(10)],
            'std_address': [f"示例地址_{i}" for i in range(10)],
            'price': np.random.randint(500000, 2000000, 10),
            'bedrooms': np.random.randint(1, 6, 10),
            'bathrooms': np.random.randint(1, 4, 10),
            'area': np.random.randint(80, 300, 10),
            'year_built': np.random.randint(1980, 2020, 10)
        })
        print("已创建示例房产数据")
    
    # 加载数据和列描述
    try:
        data_path = os.path.join('data', 'processed_data.csv')
        if os.path.exists(data_path):
            RAW_DATA = pd.read_csv(data_path)
            print(f"已加载数据: {data_path}, 共{len(RAW_DATA)}条记录")
            
            # 生成示例数据
            SAMPLE_DATA = RAW_DATA.sample(min(5, len(RAW_DATA))).to_dict('records')
            
            # 获取列描述
            column_desc_path = os.path.join('data', 'column_descriptions.json')
            if os.path.exists(column_desc_path):
                with open(column_desc_path, 'r') as f:
                    COLUMN_DESCRIPTIONS = json.load(f)
                print(f"已加载列描述: {column_desc_path}")
    except Exception as e:
        print(f"加载数据出错: {str(e)}")
    
    yield  # 这里是应用运行的地方
    
    # 应用关闭时的清理代码
    print("应用正在关闭，执行清理操作...")

# 创建FastAPI应用，使用lifespan
app = FastAPI(title="房产估价API", lifespan=lifespan)

# 配置应用JSON序列化
import fastapi.encoders
# 保存原始的jsonable_encoder
original_jsonable_encoder = fastapi.encoders.jsonable_encoder

# 创建一个包装函数，处理NumPy类型
def numpy_jsonable_encoder(obj, *args, **kwargs):
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

# 添加辅助函数，用于在API返回前转换NumPy类型
def convert_numpy_types(obj):
    """递归转换所有NumPy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # 处理非法JSON浮点值（NaN、Infinity等）
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, float):
        # 处理Python原生的非法JSON浮点值
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        # 处理自定义对象
        return {k: convert_numpy_types(v) for k, v in obj.__dict__.items()}
    return obj

# 配置CORS中间件（移到启动事件之前）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "房产估价API已启动"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}

@app.get("/api/model/info")
async def model_info():
    """获取当前激活模型的信息和诊断"""
    try:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
        
        # 检查是否有模型被加载
        if MODEL is None:
            return {
                "model_loaded": False,
                "model_type": "XGBRegressor",
                "model_directory": model_dir,
                "model_files": [os.path.basename(f) for f in glob.glob(os.path.join(model_dir, "*.joblib"))],
                "feature_count": len(FEATURE_COLS) if FEATURE_COLS is not None else 0,
                "feature_names": FEATURE_COLS
            }
            
        # 获取当前激活的模型信息
        model_type = type(MODEL).__name__ if not hasattr(MODEL, "model_type") else MODEL.model_type
        model_path = getattr(MODEL, "model_path", "未知")
        
        # 从模型路径中提取模型名称
        if model_path and model_path != "未知":
            model_name = os.path.basename(model_path).split(".")[0]
        else:
            # 尝试从模型类型推断模型名称
            if "xgboost" in model_type.lower():
                model_name = "xgboost"
            elif "ridge" in model_type.lower():
                model_name = "ridge"
            elif "lasso" in model_type.lower():
                model_name = "lasso"
            elif "linear" in model_type.lower():
                model_name = "linear"
            elif "knn" in model_type.lower():
                model_name = "knn"
            else:
                model_name = "xgboost"  # 默认
        
        # 构造模型指标文件路径
        metrics_path = os.path.join(model_dir, f"{model_name}_metrics.json")
        importance_path = os.path.join(model_dir, f"{model_name}_feature_importance.csv")
        
        # 加载模型指标
        metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                print(f"已从{metrics_path}加载模型指标")
            except Exception as e:
                print(f"加载模型指标文件失败: {str(e)}")
        
        # 如果metrics中没有feature_importance，尝试从特征重要性文件中加载
        if "feature_importance" not in metrics and os.path.exists(importance_path):
            try:
                importance_df = pd.read_csv(importance_path)
                metrics["feature_importance"] = [
                    {"feature": row["feature"], "importance": float(row["importance"])}
                    for _, row in importance_df.head(20).iterrows()
                ]
            except Exception as e:
                print(f"加载特征重要性文件失败: {str(e)}")
        
        # 如果仍然没有feature_importance，尝试从模型对象中获取
        if "feature_importance" not in metrics and hasattr(MODEL, "get_feature_importance"):
            try:
                importance_df = MODEL.get_feature_importance()
                metrics["feature_importance"] = [
                    {"feature": row["feature"], "importance": float(row["importance"])}
                    for _, row in importance_df.head(20).iterrows()
                ]
            except Exception as e:
                print(f"从模型对象获取特征重要性失败: {str(e)}")
        
        # 加载数据信息
        data_info = {}
        data_info_path = os.path.join(model_dir, "data_info.json")
        if os.path.exists(data_info_path):
            try:
                with open(data_info_path, "r") as f:
                    data_info = json.load(f)
            except Exception as e:
                print(f"加载数据集信息文件失败: {str(e)}")
        
        # 获取模型参数
        model_params = {}
        if hasattr(MODEL, "get_params"):
            try:
                model_params = MODEL.get_params()
            except Exception as e:
                print(f"获取模型参数失败: {str(e)}")
        
        # 构建响应数据
        response_data = {
            "model_loaded": True,
            "model_name": model_name,
            "model_type": model_type,
            "model_path": model_path,
            "model_directory": model_dir,
            "model_files": [os.path.basename(f) for f in glob.glob(os.path.join(model_dir, "*.joblib"))],
            "feature_count": len(FEATURE_COLS) if FEATURE_COLS is not None else 0,
            "feature_names": FEATURE_COLS,
            "metrics": metrics,
            "data_info": data_info,
            "parameters": model_params
        }
        
        # 确保返回前处理所有可能的非法JSON值
        return convert_numpy_types(response_data)
    except Exception as e:
        error_response = {
            "error": str(e),
            "model_loaded": MODEL is not None,
            "model_type": "未知",
            "model_directory": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"),
            "feature_count": len(FEATURE_COLS) if FEATURE_COLS is not None else 0
        }
        return convert_numpy_types(error_response)

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
        predicted_price = 0.0
        feature_importance = []
        
        # 检查是否为新模型框架的模型
        if MODELS_AVAILABLE and isinstance(MODEL, ModelInterface):
            # 使用新模型框架的预测方法
            predicted_price = float(MODEL.predict(features_df)[0])
            
            # 获取特征重要性
            try:
                importance_df = MODEL.get_feature_importance()
                top_n = min(10, len(importance_df))
                feature_importance = [
                    {
                        "feature": str(importance_df.iloc[i]["feature"]),
                        "importance": float(importance_df.iloc[i]["importance"]),
                        "value": float(features_df[importance_df.iloc[i]["feature"]].iloc[0])
                        if importance_df.iloc[i]["feature"] in features_df.columns else 0.0
                    }
                    for i in range(top_n)
                ]
            except Exception as e:
                print(f"获取特征重要性时出错: {e}")
        else:
            # 传统模型的预测方法
            predicted_price = float(MODEL.predict(features_df)[0])
            
            # 传统模型的特征重要性计算
            if hasattr(MODEL, 'feature_importances_'):
                importances = MODEL.feature_importances_
                indices = np.argsort(importances)[::-1]
                top_n = min(10, len(FEATURE_COLS))  # 最多返回前10个重要特征
                
                for i in range(top_n):
                    idx = indices[i]
                    feature_name = FEATURE_COLS[idx]
                    importance = importances[idx]
                    feature_value = float(features_df[feature_name].iloc[0]) if feature_name in features_df.columns else 0.0
                    
                    feature_importance.append({
                        "feature": feature_name,
                        "importance": float(importance),
                        "value": feature_value
                    })
        
        return {
            "predicted_price": predicted_price,
            "feature_importance": feature_importance
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测过程中出错: {str(e)}")

# 添加获取房产列表的API接口
@app.get("/api/properties", response_model=PropertyListResponse)
async def get_properties(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    property_type: Optional[str] = None,
    query: Optional[str] = None,
    model: Optional[str] = None
):
    """获取房产列表，支持分页和筛选"""
    
    global MODEL, FEATURE_COLS
    
    try:
        if PROPERTIES_DF is None:
            raise HTTPException(status_code=404, detail="房产数据尚未加载")
        
        # 复制一份当前模型和特征列
        current_model = MODEL
        current_feature_cols = FEATURE_COLS
        
        # 如果指定了不同的模型，尝试加载
        temp_model = None
        if model:
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(base_dir, "model", f"{model}")
                
                # 检查模型文件是否存在
                if not os.path.exists(model_path) and not os.path.exists(model_path + ".joblib"):
                    model_path = os.path.join(base_dir, "model", f"{model}.joblib")
                
                # 如果模型文件存在，加载它
                if os.path.exists(model_path):
                    if MODELS_AVAILABLE:
                        temp_model = ModelFactory.load_model(model_path)
                    else:
                        temp_model = joblib.load(model_path)
                    
                    # 使用临时模型的特征列（如果可用）
                    if MODELS_AVAILABLE and hasattr(temp_model, 'feature_names') and temp_model.feature_names:
                        current_feature_cols = temp_model.feature_names
            except Exception as e:
                print(f"加载指定模型失败: {str(e)}")
                # 继续使用默认模型
                pass
        
        # 创建过滤条件
        filtered_df = PROPERTIES_DF
        
        # 按房产类型过滤
        if property_type:
            filtered_df = filtered_df[filtered_df['property_type'] == property_type]
        
        # 按查询字符串过滤（支持地址和ID搜索）
        if query:
            query_condition = (
                filtered_df['prop_id'].astype(str).str.contains(query, case=False, na=False) |
                filtered_df['std_address'].str.contains(query, case=False, na=False)
            )
            filtered_df = filtered_df[query_condition]
        
        # 计算总记录数
        total = len(filtered_df)
        
        # 分页数据
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paged_df = filtered_df.iloc[start_idx:end_idx].copy()
        
        properties = []
        
        for _, row in paged_df.iterrows():
            # 实际的y_label价格
            actual_price = float(row.get('y_label', 0))
            
            # 计算房价预测
            pred_price = 0
            pred_model = temp_model if temp_model else current_model
            feature_cols_to_use = current_feature_cols
            
            if pred_model is not None and set(feature_cols_to_use).issubset(filtered_df.columns):
                try:
                    features = row[feature_cols_to_use]
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
                    pred_price = float(pred_model.predict(features_df)[0])
                     
                except Exception as e:
                    print(f"预测异常: {e}") 
            else:  
                print(f"模型不可用，使用随机预测: {pred_price}")
            
            # 创建属性字典
            features = {}
            for col in row.index:
                if col not in ['prop_id', 'std_address']:
                    val = row[col]
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        features[col] = float(val)
                    else:
                        features[col] = str(val) if not pd.isna(val) else None
            
            # 添加使用的模型名称到特征中
            if model:
                features['model_name'] = model
            
            properties.append(Property(
                prop_id=str(row['prop_id']),
                address=row['std_address'],
                predicted_price=pred_price,
                features=features
            ))
        
        return PropertyListResponse(
            total=total,
            page=page,
            page_size=page_size,
            properties=convert_numpy_types(properties)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取房产列表出错: {str(e)}")

# 添加获取单个房产详情的API接口
@app.get("/api/properties/{prop_id}", response_model=PropertyDetail)
async def get_property_detail(prop_id: str, model: Optional[str] = None):
    """获取单个房产的详情"""
    global MODEL, FEATURE_COLS
    
    if PROPERTIES_DF is None:
        raise HTTPException(status_code=404, detail="房产数据尚未加载")
    
    try:
        # 复制一份当前模型和特征列
        current_model = MODEL
        current_feature_cols = FEATURE_COLS
        
        # 如果指定了不同的模型，尝试加载
        temp_model = None
        if model:
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(base_dir, "model", f"{model}")
                
                # 检查模型文件是否存在
                if not os.path.exists(model_path) and not os.path.exists(model_path + ".joblib"):
                    model_path = os.path.join(base_dir, "model", f"{model}.joblib")
                
                # 如果模型文件存在，加载它
                if os.path.exists(model_path):
                    if MODELS_AVAILABLE:
                        temp_model = ModelFactory.load_model(model_path)
                    else:
                        temp_model = joblib.load(model_path)
                    
                    # 使用临时模型的特征列（如果可用）
                    if MODELS_AVAILABLE and hasattr(temp_model, 'feature_names') and temp_model.feature_names:
                        current_feature_cols = temp_model.feature_names
            except Exception as e:
                print(f"加载指定模型失败: {str(e)}")
                # 继续使用默认模型
                pass
        
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
        
        # 使用指定的模型预测价格和计算特征重要性
        pred_model = temp_model if temp_model else current_model
        feature_cols_to_use = current_feature_cols
        
        # 预测价格和计算特征重要性
        try:
            pred_price, feature_importance = predict_property_price(
                row=row, 
                model=pred_model, 
                feature_cols=feature_cols_to_use, 
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
                feature_cols=feature_cols_to_use if feature_cols_to_use else [], 
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
                if isinstance(val, (int, float)) and not np.isnan(val):
                    features[col] = float(val)
                else:
                    features[col] = str(val) if not pd.isna(val) else None
                    
        # 添加使用的模型名称到特征中
        if model:
            features['model_name'] = model
        
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
                    feature_cols=feature_cols_to_use if feature_cols_to_use else [],
                    pred_price=pred_price
                )
            
            # 检查并确保comparable_properties不为空
            if not property_detail.comparable_properties:
                property_detail.comparable_properties = generate_dummy_comparables(row, prop_id)
                
            return convert_numpy_types(property_detail)
            
        except Exception as e:
            print(f"创建PropertyDetail对象时出错: {e}")
            # 创建最基本的PropertyDetail对象
            property_detail = PropertyDetail(
                prop_id=str(row['prop_id']),
                address=row['std_address'],
                predicted_price=pred_price,
                features=features,
                feature_importance=generate_rule_based_importance(row, feature_cols_to_use if feature_cols_to_use else [], pred_price),
                comparable_properties=generate_dummy_comparables(row, prop_id),
                price_trends=generate_price_trends(pred_price),
                price_range=calculate_price_range(pred_price),
                neighborhood_stats=get_neighborhood_stats(pred_price, prop_area),
                confidence_interval=calculate_confidence_interval(pred_price),
                ai_explanation=get_model_explanation(pred_price, [], feature_cols_to_use if feature_cols_to_use else [])
            )
            
            return convert_numpy_types(property_detail)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取房产详情出错: {str(e)}")

@app.get("/api/properties/{prop_id}/pdf")
async def generate_property_pdf(prop_id: str):
    """生成并返回指定房产的PDF报告"""
    print(f"开始为房产 {prop_id} 生成PDF报告")
    try:
        # 检查全局数据是否已加载
        if PROPERTIES_DF is None:
            print("错误: 房产数据尚未加载")
            raise HTTPException(status_code=500, detail="房产数据尚未加载")
            
        # 从数据中获取房产详情
        print(f"正在查找房产ID: {prop_id}")
        property_data = PROPERTIES_DF[PROPERTIES_DF["prop_id"].astype(str) == prop_id]
        
        if property_data.empty:
            print(f"错误: 未找到房产ID {prop_id}")
            raise HTTPException(status_code=404, detail="未找到该房产")
        
        print(f"已找到房产记录，开始获取详细信息")
        # 获取完整的房产详情
        try:
            detail = await get_property_detail(prop_id)
            print(f"成功获取房产详情: {detail.address}")
        except Exception as e:
            print(f"获取房产详情失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"获取房产详情失败: {str(e)}")
        
        # 配置中文字体支持
        print("配置中文字体支持")
        # 在Windows中尝试使用系统字体
        font_path = None
        try:
            if platform.system() == 'Windows':
                # 尝试常见的系统字体路径
                possible_fonts = [
                    "C:/Windows/Fonts/simhei.ttf",  # 黑体
                    "C:/Windows/Fonts/simsun.ttc",  # 宋体
                    "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
                    "C:/Windows/Fonts/simfang.ttf", # 仿宋
                    "C:/Windows/Fonts/arial.ttf",   # Arial (作为后备)
                ]
                for path in possible_fonts:
                    if os.path.exists(path):
                        font_path = path
                        print(f"找到字体: {path}")
                        break
                
                if font_path is None:
                    print("警告: 未找到任何中文字体，将使用默认字体")
        except Exception as e:
            print(f"字体检测过程出错: {str(e)}")
        
        # 注册字体
        chinese_font_name = 'Helvetica'  # 默认字体
        try:
            if font_path:
                font_name = os.path.basename(font_path).split('.')[0]
                try:
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    chinese_font_name = font_name
                    print(f"成功注册字体: {font_name}")
                except Exception as font_error:
                    print(f"字体注册失败: {str(font_error)}，将使用默认字体")
        except Exception as e:
            print(f"字体处理过程出错: {str(e)}")
        
        # 创建一个内存中的PDF文件
        print("创建PDF文档")
        buffer = io.BytesIO()
        try:
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
            styles = getSampleStyleSheet()
            
            # 自定义中文标题和正文样式
            title_style = ParagraphStyle(
                'ChineseTitle',
                parent=styles['Title'],
                fontName=chinese_font_name
            )
            
            heading2_style = ParagraphStyle(
                'ChineseHeading2',
                parent=styles['Heading2'],
                fontName=chinese_font_name
            )
            
            normal_style = ParagraphStyle(
                'ChineseNormal',
                parent=styles['Normal'],
                fontName=chinese_font_name
            )
            
            elements = []
            
            print("添加PDF内容: 标题")
            # 添加标题
            try:
                title_text = f"房产估价报告: {detail.address}"
                elements.append(Paragraph(title_text, title_style))
                elements.append(Spacer(1, 0.25*inch))
            except Exception as e:
                print(f"添加标题时出错: {str(e)}")
                # 使用简单标题作为后备
                elements.append(Paragraph("房产估价报告", title_style))
                elements.append(Spacer(1, 0.25*inch))
            
            print("添加PDF内容: 基本信息")
            # 添加基本信息
            elements.append(Paragraph("基本信息", heading2_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # 创建基本信息表格数据
            try:
                basic_data = [
                    ["属性", "值"],
                    ["房产ID", str(detail.prop_id)],
                    ["地址", str(detail.address)],
                    ["预测价格", f"¥{detail.predicted_price:,.2f}"],
                ]
                
                # 添加特征值到表格
                for key, value in detail.features.items():
                    if key not in ["prop_id", "address"]:
                        # 格式化特征名称和值
                        formatted_key = str(key).replace("_", " ").title()
                        if isinstance(value, (int, float)) and key != "predicted_price":
                            formatted_value = f"{value:,.2f}" if '.' in str(value) else f"{value:,}"
                        else:
                            formatted_value = str(value)
                        basic_data.append([formatted_key, formatted_value])
            except Exception as e:
                print(f"准备基本信息数据时出错: {str(e)}")
                # 使用简单的基本数据作为后备
                basic_data = [
                    ["属性", "值"],
                    ["房产ID", str(detail.prop_id)],
                    ["地址", str(detail.address)],
                ]
            
            # 将表格数据转换为Paragraph对象以支持中文
            print("格式化表格数据")
            try:
                styled_basic_data = []
                for row in basic_data:
                    styled_row = []
                    for cell in row:
                        try:
                            styled_row.append(Paragraph(str(cell), normal_style))
                        except Exception as cell_error:
                            print(f"处理单元格数据时出错: {cell} - {str(cell_error)}")
                            styled_row.append(Paragraph("数据错误", normal_style))
                    styled_basic_data.append(styled_row)
                
                # 创建并设置表格样式
                basic_table = Table(styled_basic_data, colWidths=[2.5*inch, 3*inch])
                basic_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.gray),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                    ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                
                elements.append(basic_table)
                elements.append(Spacer(1, 0.25*inch))
            except Exception as e:
                print(f"创建基本信息表格时出错: {str(e)}")
                # 跳过表格，添加简单文本
                elements.append(Paragraph("基本信息无法显示", normal_style))
                elements.append(Spacer(1, 0.25*inch))
            
            # 添加特征影响部分
            if detail.feature_importance and len(detail.feature_importance) > 0:
                print("添加PDF内容: 特征影响因素")
                try:
                    elements.append(Paragraph("特征影响因素", heading2_style))
                    elements.append(Spacer(1, 0.1*inch))
                    
                    feature_data = [["特征", "影响值", "影响百分比"]]
                    for feature in detail.feature_importance[:10]:  # 仅显示前10个特征
                        feature_data.append([
                            str(feature.get("name", "")).replace("_", " ").title(),
                            f"{feature.get('value', 0):,.2f}",
                            f"{feature.get('percentage', 0):.2f}%"
                        ])
                    
                    # 将表格数据转换为Paragraph对象以支持中文
                    styled_feature_data = []
                    for row in feature_data:
                        styled_row = []
                        for cell in row:
                            styled_row.append(Paragraph(str(cell), normal_style))
                        styled_feature_data.append(styled_row)
                    
                    feature_table = Table(styled_feature_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                    feature_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (2, 0), colors.gray),
                        ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (2, 0), 'CENTER'),
                        ('BOTTOMPADDING', (0, 0), (2, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ]))
                    
                    elements.append(feature_table)
                    elements.append(Spacer(1, 0.25*inch))
                except Exception as e:
                    print(f"添加特征影响部分时出错: {str(e)}")
            
            # 添加可比较房产信息
            if detail.comparable_properties and len(detail.comparable_properties) > 0:
                print("添加PDF内容: 可比较房产")
                try:
                    elements.append(Paragraph("可比较房产", heading2_style))
                    elements.append(Spacer(1, 0.1*inch))
                    
                    comp_data = [["地址", "价格", "差异"]]
                    for comp in detail.comparable_properties[:5]:  # 仅显示前5个可比较房产
                        comp_data.append([
                            str(comp.get("address", "")),
                            f"¥{comp.get('price', 0):,.2f}",
                            f"{comp.get('price_diff_percent', 0):.2f}%"
                        ])
                    
                    # 将表格数据转换为Paragraph对象以支持中文
                    styled_comp_data = []
                    for row in comp_data:
                        styled_row = []
                        for cell in row:
                            styled_row.append(Paragraph(str(cell), normal_style))
                        styled_comp_data.append(styled_row)
                    
                    comp_table = Table(styled_comp_data, colWidths=[3*inch, 1.5*inch, 1*inch])
                    comp_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (2, 0), colors.gray),
                        ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (2, 0), 'CENTER'),
                        ('BOTTOMPADDING', (0, 0), (2, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ]))
                    
                    elements.append(comp_table)
                    elements.append(Spacer(1, 0.25*inch))
                except Exception as e:
                    print(f"添加可比较房产部分时出错: {str(e)}")
            
            # 添加价格区间信息
            if detail.price_range:
                print("添加PDF内容: 价格区间")
                try:
                    elements.append(Paragraph("价格区间", heading2_style))
                    elements.append(Spacer(1, 0.1*inch))
                    
                    price_range_data = [
                        ["最低价格", "预测价格", "最高价格"],
                        [
                            f"¥{detail.price_range.get('min', 0):,.2f}",
                            f"¥{detail.predicted_price:,.2f}",
                            f"¥{detail.price_range.get('max', 0):,.2f}"
                        ]
                    ]
                    
                    # 将表格数据转换为Paragraph对象以支持中文
                    styled_price_range_data = []
                    for row in price_range_data:
                        styled_row = []
                        for cell in row:
                            styled_row.append(Paragraph(str(cell), normal_style))
                        styled_price_range_data.append(styled_row)
                    
                    price_range_table = Table(styled_price_range_data, colWidths=[2*inch, 2*inch, 2*inch])
                    price_range_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (2, 0), colors.gray),
                        ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (2, 0), 'CENTER'),
                        ('BOTTOMPADDING', (0, 0), (2, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ALIGN', (0, 1), (2, 1), 'CENTER'),
                    ]))
                    
                    elements.append(price_range_table)
                    elements.append(Spacer(1, 0.25*inch))
                except Exception as e:
                    print(f"添加价格区间部分时出错: {str(e)}")
            
            # 添加报告生成时间
            print("添加PDF内容: 生成时间")
            try:
                from datetime import datetime
                now = datetime.now()
                date_style = ParagraphStyle(
                    "ChineseDateStyle", 
                    parent=normal_style,
                    alignment=1  # 1 = center
                )
                elements.append(Paragraph(f"报告生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')}", date_style))
            except Exception as e:
                print(f"添加生成时间时出错: {str(e)}")
            
            # 构建PDF文档
            print("构建PDF文档")
            doc.build(elements)
            buffer.seek(0)
            
            # 获取PDF内容
            pdf_content = buffer.getvalue()
            
            # 设置文件名
            filename = f"property_report_{prop_id}.pdf"
            
            print(f"PDF生成成功，大小: {len(pdf_content)} 字节")
            
            # 创建Response对象
            response = Response(
                content=pdf_content,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
            
            return response
        except Exception as e:
            print(f"PDF生成过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"PDF生成过程中出错: {str(e)}")
            
    except HTTPException as he:
        # 重新抛出HTTP异常
        raise he
    except Exception as e:
        print(f"生成PDF报告时出现未预期错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成PDF报告时出错: {str(e)}")

# 添加一个简化版的PDF生成API端点
@app.get("/api/properties/{prop_id}/pdf-simple")
async def generate_simple_property_pdf(prop_id: str):
    """生成并返回指定房产的简化PDF报告"""
    print(f"开始为房产 {prop_id} 生成简化PDF报告")
    try:
        # 检查全局数据是否已加载
        if PROPERTIES_DF is None:
            print("错误: 房产数据尚未加载")
            raise HTTPException(status_code=500, detail="房产数据尚未加载")
            
        # 从数据中获取房产详情
        print(f"正在查找房产ID: {prop_id}")
        property_data = PROPERTIES_DF[PROPERTIES_DF["prop_id"].astype(str) == prop_id]
        
        if property_data.empty:
            print(f"错误: 未找到房产ID {prop_id}")
            raise HTTPException(status_code=404, detail="未找到该房产")
        
        # 不使用get_property_detail函数，直接从数据中获取基本信息
        row = property_data.iloc[0]
        address = str(row.get('address', '未知地址'))
        
        # 使用简单的reportlab创建PDF
        buffer = io.BytesIO()
        
        # 创建一个简单的PDF文档，避免复杂的格式和字体问题
        from reportlab.pdfgen import canvas
        
        # 设置页面大小为A4
        c = canvas.Canvas(buffer, pagesize=A4)
        
        # 添加标题
        title = "房产估价报告"
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(A4[0]/2, A4[1] - 50, title)
        
        # 添加房产ID和地址
        c.setFont("Helvetica", 12)
        c.drawString(50, A4[1] - 100, f"房产ID: {prop_id}")
        c.drawString(50, A4[1] - 120, f"地址: {address}")
        
        # 添加预测价格（如果有）
        predicted_price = row.get('predicted_price', 0)
        if isinstance(predicted_price, (int, float)):
            price_str = f"¥{predicted_price:,.2f}"
        else:
            price_str = str(predicted_price)
        c.drawString(50, A4[1] - 140, f"预测价格: {price_str}")
        
        # 添加主要特征
        y_pos = A4[1] - 180
        c.drawString(50, y_pos, "主要特征:")
        y_pos -= 20
        
        feature_count = 0
        for col in property_data.columns:
            if col not in ['prop_id', 'address', 'predicted_price'] and feature_count < 10:
                try:
                    value = row[col]
                    if pd.notna(value):  # 只显示非空值
                        if isinstance(value, (int, float)):
                            value_str = f"{value:,.2f}" if '.' in str(value) else f"{value:,}"
                        else:
                            value_str = str(value)
                        
                        feature_name = col.replace('_', ' ').title()
                        c.drawString(70, y_pos, f"{feature_name}: {value_str}")
                        y_pos -= 15
                        feature_count += 1
                except Exception as e:
                    print(f"处理特征 {col} 时出错: {str(e)}")
        
        # 添加生成时间
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d %H:%M:%S')
        c.setFont("Helvetica-Italic", 10)
        c.drawString(50, 50, f"报告生成时间: {date_str}")
        
        # 保存PDF
        c.save()
        buffer.seek(0)
        
        # 获取PDF内容
        pdf_content = buffer.getvalue()
        
        # 设置文件名
        filename = f"property_report_simple_{prop_id}.pdf"
        
        print(f"简化PDF生成成功，大小: {len(pdf_content)} 字节")
        
        # 创建Response对象
        response = Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
        return response
        
    except Exception as e:
        print(f"生成简化PDF报告时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成简化PDF报告时出错: {str(e)}")

# 模型管理相关的API
@app.get("/api/models", response_model=ModelListResponse)
async def list_models():
    """获取所有可用模型的列表"""
    try:
        # 获取模型目录路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'model')
        
        # 查找所有metrics.json文件
        metrics_files = glob.glob(os.path.join(model_dir, "*_metrics.json"))
        
        models = []
        # 读取每个指标文件并提取信息
        for metrics_file in metrics_files:
            try:
                model_name = os.path.basename(metrics_file).replace("_metrics.json", "")
                
                # 检查是否是临时文件或不需要的文件
                if model_name.startswith("ensemble_") or model_name.startswith("stacking_"):
                    continue  # 跳过集成模型文件
                
                # 读取指标文件
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                
                # 检查此模型是否为当前激活的模型
                model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
                is_active = MODEL_PATH.name == f"{model_name}_model.joblib" if MODEL_PATH else False
                
                # 计算误差小于10%的样本比例（合并<5%和5-10%）
                error_under_10_percent = 0
                if (metrics_data.get("error_distribution") and 
                    metrics_data["error_distribution"].get("error_ranges")):
                    error_ranges = metrics_data["error_distribution"]["error_ranges"]
                    less_than_5 = error_ranges.get("<5%", 0)
                    between_5_and_10 = error_ranges.get("5-10%", 0)
                    error_under_10_percent = less_than_5 + between_5_and_10
                else:
                    # 如果没有error_ranges，尝试从metrics.json文件读取其他相关字段估算
                    if metrics_data.get("median_percentage_error"):
                        median_pct_error = metrics_data.get("median_percentage_error", 10)
                        # 根据中位误差估算一个误差<10%的比例
                        if median_pct_error < 5:
                            error_under_10_percent = 0.8  # 80%
                        elif median_pct_error < 10:
                            error_under_10_percent = 0.5  # 50%
                        else:
                            error_under_10_percent = 0.3  # 30%
                
                # 创建模型信息对象
                model_info = {
                    "name": model_name,
                    "path": f"../model/{model_name}_model.joblib",
                    "type": model_name.split("_")[0] if "_" in model_name else model_name,
                    "metrics": {
                        "rmse": metrics_data.get("rmse", 0),
                        "mae": metrics_data.get("mae", 0),
                        "median_percentage_error": metrics_data.get("median_percentage_error", 0),
                        "error_under_10_percent": round(error_under_10_percent * 100, 3)  # 转为百分比，保留3位小数
                    },
                    "created_at": os.path.getmtime(metrics_file),
                    "status": "active" if is_active else "available"
                }
                
                models.append(model_info)
            except Exception as e:
                print(f"处理metrics文件时出错 {metrics_file}: {str(e)}")
        
        # 按中位误差从小到大排序
        models = sorted(models, key=lambda x: x["metrics"]["median_percentage_error"], reverse=False)
        
        return ModelListResponse(models=models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")

@app.post("/api/models/train", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest):
    """训练新模型"""
    try:
        # 确保model目录存在
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        # 使用绝对路径
        model_path = os.path.join(model_dir, f"{request.model_type}_model.joblib")
        
        # 读取数据
        data_path = "../resources/house_samples_features.csv"
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"数据文件不存在: {data_path}")
        
        # 导入train_and_evaluate_model函数
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from train_models import train_and_evaluate_model
        
        # 训练模型
        metrics = train_and_evaluate_model(
            model_type=request.model_type,
            data_file=data_path,
            output_dir="../model",
            test_size=request.test_size,
            random_state=request.random_state,
            params=request.params
        )
        
        return ModelTrainingResponse(
            success=True,
            message=f"{request.model_type}模型训练成功",
            metrics=metrics,
            model_path=model_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")

@app.get("/api/models/{model_type}/params")
async def get_model_params(model_type: str):
    """获取指定模型类型的默认参数"""
    try:
        if not MODELS_AVAILABLE:
            raise HTTPException(status_code=400, detail="模型模块不可用，无法获取模型参数")
        
        # 创建临时模型实例以获取默认参数
        model = ModelFactory.create_model(model_type)
        
        return {
            "model_type": model_type,
            "params": model.get_params()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型参数失败: {str(e)}")

@app.delete("/api/models/{model_path:path}")
async def delete_model(model_path: str):
    """删除指定的模型文件"""
    try:
        # 使用绝对路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, "model", model_path)
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"模型文件不存在: {model_path}")
        
        # 删除模型文件
        os.remove(full_path)
        
        # 尝试删除关联的元数据文件
        meta_path = os.path.splitext(full_path)[0] + "_meta.json"
        if os.path.exists(meta_path):
            os.remove(meta_path)
        
        return {"success": True, "message": f"模型 {model_path} 已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")

@app.post("/api/models/{model_path:path}/activate")
async def activate_model(model_path: str):
    """激活指定的模型文件"""
    global MODEL, FEATURE_COLS
    
    try:
        # 使用绝对路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, "model", model_path)
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"模型文件不存在: {model_path}")
        
        if not MODELS_AVAILABLE:
            # 使用传统方式加载模型
            MODEL = joblib.load(full_path)
            
            # 尝试加载特征列
            feature_cols_path = Path("../model/feature_cols.joblib")
            if feature_cols_path.exists():
                FEATURE_COLS = joblib.load(feature_cols_path)
        else:
            # 使用模型工厂加载模型
            MODEL = ModelFactory.load_model(full_path)
            
            # 从模型中获取特征列
            if hasattr(MODEL, 'feature_names') and MODEL.feature_names is not None:
                FEATURE_COLS = MODEL.feature_names
        
        # 在激活后，同步模型指标到前端公共目录
        model_name = os.path.basename(full_path).split(".")[0]
        
        # 模型指标文件
        metrics_path = os.path.join(base_dir, "model", f"{model_name}_metrics.json")
        frontend_metrics_path = os.path.join(base_dir, "frontend", "public", "data", "model_metrics.json")
        
        # 如果存在模型特定的指标文件，将其复制到前端公共目录
        if os.path.exists(metrics_path):
            try:
                # 读取模型特定的指标文件
                with open(metrics_path, 'r') as f:
                    model_metrics = json.load(f)
                
                # 保存到前端公共目录
                os.makedirs(os.path.dirname(frontend_metrics_path), exist_ok=True)
                with open(frontend_metrics_path, 'w') as f:
                    json.dump(model_metrics, f, indent=2)
                    
                print(f"已将模型 {model_name} 的指标文件复制到前端公共目录")
            except Exception as e:
                print(f"同步模型指标文件时出错: {str(e)}")
        else:
            print(f"未找到模型 {model_name} 的指标文件: {metrics_path}")
        
        return {"success": True, "message": f"模型 {model_path} 已激活"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"激活模型失败: {str(e)}")

@app.get("/api/debug/model")
async def debug_model():
    """调试模型加载问题"""
    # 检查模型路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'model')
    model_path = os.path.join(model_dir, "xgb_model.joblib")
    
    # 收集诊断信息
    result = {
        "model_path": str(model_path),
        "model_exists": os.path.exists(model_path),
        "model_size": os.path.getsize(model_path) if os.path.exists(model_path) else 0,
        "model_permissions": oct(os.stat(model_path).st_mode)[-3:] if os.path.exists(model_path) else None,
        "global_model_loaded": MODEL is not None,
        "python_version": sys.version,
        "joblib_version": joblib.__version__ if hasattr(joblib, "__version__") else "unknown",
        "module_paths": sys.path,
        "model_dir_contents": os.listdir(model_dir)
    }
    
    # 尝试加载并诊断问题
    if os.path.exists(model_path):
        try:
            # 尝试使用不同方法加载
            result["load_attempts"] = []
            
            # 1. 标准joblib加载
            try:
                model_data = joblib.load(model_path)
                result["load_attempts"].append({
                    "method": "joblib.load",
                    "success": True,
                    "model_type": str(type(model_data)),
                    "model_keys": list(model_data.keys()) if isinstance(model_data, dict) else None
                })
            except Exception as e:
                result["load_attempts"].append({
                    "method": "joblib.load",
                    "success": False,
                    "error": str(e)
                })
            
            # 2. pickle加载
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    pickle_model = pickle.load(f)
                result["load_attempts"].append({
                    "method": "pickle.load",
                    "success": True,
                    "model_type": str(type(pickle_model)),
                    "model_keys": list(pickle_model.keys()) if isinstance(pickle_model, dict) else None
                })
            except Exception as e:
                result["load_attempts"].append({
                    "method": "pickle.load",
                    "success": False,
                    "error": str(e)
                })
                
            # 3. 读取原始字节
            try:
                with open(model_path, 'rb') as f:
                    first_100_bytes = f.read(100)
                result["first_100_bytes_hex"] = first_100_bytes.hex()
            except Exception as e:
                result["first_100_bytes_error"] = str(e)
                
        except Exception as e:
            result["diagnosis_error"] = str(e)
    
    return result

@app.get("/api/fix/model")
async def fix_model():
    """尝试修复模型加载问题"""
    global MODEL, TRAINED_MODEL
    
    # 收集结果
    result = {"fixes_attempted": []}
    
    # 模型路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'model')
    source_path = os.path.join(model_dir, "xgb_model.joblib")
    
    # 修复方法1: 重新拷贝并确保权限
    try:
        new_path = os.path.join(model_dir, "fixed_xgb_model.joblib")
        import shutil
        shutil.copy2(source_path, new_path)
        os.chmod(new_path, 0o644)  # 确保权限正确
        
        result["fixes_attempted"].append({
            "method": "file_copy_and_permissions",
            "source": source_path,
            "target": new_path,
            "success": os.path.exists(new_path)
        })
        
        # 尝试加载这个新文件
        try:
            model_data = joblib.load(new_path)
            
            # 尝试不同方式获取模型对象
            if isinstance(model_data, dict):
                if "model" in model_data:
                    MODEL = model_data["model"]
                    result["fixes_attempted"].append({
                        "method": "load_from_dict_model",
                        "success": True
                    })
                else:
                    # 寻找可能的模型对象
                    for key, value in model_data.items():
                        if hasattr(value, 'predict'):
                            MODEL = value
                            result["fixes_attempted"].append({
                                "method": "load_from_dict_with_predict",
                                "success": True,
                                "key_used": key
                            })
                            break
            else:
                MODEL = model_data
                result["fixes_attempted"].append({
                    "method": "direct_load",
                    "success": True
                })
                
            # 如果成功加载了模型
            if MODEL is not None:
                TRAINED_MODEL = MODEL
                result["model_loaded"] = True
                result["model_type"] = str(type(MODEL))
        except Exception as e:
            result["fixes_attempted"].append({
                "method": "load_new_file",
                "success": False,
                "error": str(e)
            })
    
    except Exception as e:
        result["fixes_attempted"].append({
            "method": "file_copy_and_permissions",
            "success": False,
            "error": str(e)
        })
    
    # 最后检查模型是否已加载
    result["final_model_loaded"] = MODEL is not None
    
    return result

@app.get("/api/models/{model_name}/metrics")
async def get_model_metrics(model_name: str):
    """获取指定模型的metrics.json文件内容"""
    try:
        # 使用绝对路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metrics_path = os.path.join(base_dir, "model", f"{model_name}_metrics.json")
        
        print(f"尝试读取模型指标文件: {metrics_path}")
        
        if not os.path.exists(metrics_path):
            print(f"找不到模型指标文件: {metrics_path}")
            raise HTTPException(status_code=404, detail=f"找不到模型 {model_name} 的指标文件")
        
        # 读取metrics.json文件
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        
        print(f"成功读取模型指标数据，数据键: {list(metrics_data.keys())}")
        
        # 兼容性处理：确保数据结构一致
        # 1. 如果数据嵌套在metrics内部，提取到顶层
        if "metrics" in metrics_data and isinstance(metrics_data["metrics"], dict):
            print("检测到嵌套metrics结构，正在调整...")
            metrics_content = metrics_data["metrics"]
            # 保留非metrics字段
            for key, value in metrics_data.items():
                if key != "metrics":
                    metrics_content[key] = value
            metrics_data = metrics_content
        
        # 2. 确保error_under_10_percent以百分比形式存在
        if "error_under_10_percent" in metrics_data:
            value = metrics_data["error_under_10_percent"]
            # 如果值小于1，说明可能是比例而非百分比
            if isinstance(value, (int, float)) and value < 1:
                print(f"转换error_under_10_percent从比例({value})到百分比({value*100})")
                metrics_data["error_under_10_percent"] = value * 100
        
        # 3. 对feature_importance进行特殊处理
        if "feature_importance" in metrics_data and isinstance(metrics_data["feature_importance"], list):
            # 确保每个feature都有正确的结构
            for i, item in enumerate(metrics_data["feature_importance"]):
                if isinstance(item, dict) and "feature" not in item and "name" in item:
                    # 兼容"name"作为"feature"的情况
                    item["feature"] = item["name"]
                    metrics_data["feature_importance"][i] = item
        
        # 4. 处理特殊浮点值（NaN, Infinity, -Infinity）
        def clean_float_values(obj):
            if isinstance(obj, float):
                # 检查是否为NaN或Infinity
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            elif isinstance(obj, dict):
                return {k: clean_float_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_float_values(item) for item in obj]
            return obj
        
        # 清理所有特殊浮点值
        metrics_data = clean_float_values(metrics_data)
        
        print(f"处理后的指标数据键: {list(metrics_data.keys())}")
        print(f"处理后的指标数据示例: {json.dumps(metrics_data, cls=NumpyEncoder, ensure_ascii=False)[:500]}...")
        
        return metrics_data
    except Exception as e:
        print(f"获取模型指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型指标失败: {str(e)}")

# Vercel部署支持
from mangum import Mangum
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8102))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)  # 设置reload=False