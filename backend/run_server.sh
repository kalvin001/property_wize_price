#!/bin/bash
# Bash 启动脚本
echo "启动房价预测系统后端服务..."
echo "当前路径: $(pwd)"
echo "检查数据文件..."

RESOURCES_PATH="../resources"
MODEL_PATH="../model"

# 检查数据文件
if [ -f "$RESOURCES_PATH/house_samples_features.csv" ]; then
    echo "找到数据文件: $RESOURCES_PATH/house_samples_features.csv"
else
    echo "警告: 未找到数据文件 $RESOURCES_PATH/house_samples_features.csv"
fi

# 检查模型文件
if [ -f "$MODEL_PATH/xgb_model.joblib" ]; then
    echo "找到模型文件: $MODEL_PATH/xgb_model.joblib"
else
    echo "警告: 未找到模型文件 $MODEL_PATH/xgb_model.joblib"
fi

# 启动服务器
echo "启动FastAPI服务器..."
python main.py 