# PowerShell 启动脚本
Write-Host "启动房价预测系统后端服务..."
Write-Host "当前路径: $(Get-Location)"
Write-Host "检查数据文件..."

$resourcesPath = "../resources"
$modelPath = "../model"

# 检查数据文件
if (Test-Path "$resourcesPath/house_samples_features.csv") {
    Write-Host "找到数据文件: $resourcesPath/house_samples_features.csv"
} else {
    Write-Host "警告: 未找到数据文件 $resourcesPath/house_samples_features.csv"
}

# 检查模型文件
if (Test-Path "$modelPath/xgb_model.joblib") {
    Write-Host "找到模型文件: $modelPath/xgb_model.joblib"
} else {
    Write-Host "警告: 未找到模型文件 $modelPath/xgb_model.joblib"
}

# 启动服务器
Write-Host "启动FastAPI服务器..."
python main.py 