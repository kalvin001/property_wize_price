from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(title="房产估价API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应当设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class PropertyFeatures(BaseModel):
    features: Dict[str, Any]

class PredictionResult(BaseModel):
    predicted_price: float
    feature_importance: List[Dict[str, Any]]

# 全局变量
MODEL_PATH = Path("../model/xgb_model.joblib")  # 更新为XGBoost模型路径
FEATURE_COLS = []
MODEL = None

@app.on_event("startup")
async def startup_event():
    global MODEL, FEATURE_COLS
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
        # 加载特征列名
        feature_cols_path = Path("../model/feature_cols.joblib")
        if feature_cols_path.exists():
            FEATURE_COLS = joblib.load(feature_cols_path)
        print(f"XGBoost模型已加载，特征数量: {len(FEATURE_COLS)}")
    else:
        print("警告：模型文件不存在，请先运行分析脚本生成模型")

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
            import json
            with open(metrics_path, 'r') as f:
                model_metrics = json.load(f)
        
        data_info = {}
        if data_info_path.exists():
            import json
            with open(data_info_path, 'r') as f:
                data_info = json.load(f)
        
        return {
            "model_type": "XGBRegressor",  # 更新为XGBoost
            "features_count": len(FEATURE_COLS),
            "feature_names": FEATURE_COLS[:10] + ["..."] if FEATURE_COLS else [],
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
    
    import json
    with open(sample_path, 'r') as f:
        sample_properties = json.load(f)
    
    return sample_properties

@app.post("/api/predict")
async def predict_price(property_data: PropertyFeatures):
    """基于输入特征预测房价"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")
    
    # 准备特征数据
    features = property_data.features
    missing_features = [f for f in FEATURE_COLS if f not in features]
    if missing_features:
        return JSONResponse(
            status_code=400,
            content={"detail": f"缺少必要的特征: {', '.join(missing_features[:5])}..."}
        )
    
    # 构建特征向量 - 使用DataFrame以保留特征名称
    feature_df = pd.DataFrame([{feature: features.get(feature, 0) for feature in FEATURE_COLS}])
    
    # 预测价格
    try:
        predicted_price = float(MODEL.predict(feature_df)[0])
        
        # 获取特征重要性
        feature_importance = []
        if hasattr(MODEL, 'feature_importances_'):
            importances = MODEL.feature_importances_
            indices = np.argsort(importances)[-10:]  # 获取最重要的10个特征
            for i in reversed(indices):
                if i < len(FEATURE_COLS):
                    feature_importance.append({
                        "feature": FEATURE_COLS[i],
                        "importance": float(importances[i]),
                        "value": float(features.get(FEATURE_COLS[i], 0))
                    })
        
        return {
            "predicted_price": predicted_price,
            "feature_importance": feature_importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测出错: {str(e)}")

# Vercel部署支持
from mangum import Mangum
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True) 