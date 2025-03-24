@echo off
echo ===================================
echo 开始训练高级PyTorch深度神经网络模型（带残差连接）
echo ===================================

REM 检测Python环境
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 错误：未检测到Python环境，请确保已安装Python并添加到PATH
    exit /b 1
)

REM 检查PyTorch是否已安装
python -c "import torch" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 警告：PyTorch未安装，尝试安装中...
    pip install torch tqdm
    if %ERRORLEVEL% neq 0 (
        echo 错误：PyTorch安装失败，请手动安装
        exit /b 1
    )
    echo PyTorch安装成功！
)

REM 设置默认参数
set DATA_FILE=resources/house_samples_features.csv
set OUTPUT_DIR=model
set MODEL_SIZE=medium

REM 解析命令行参数
:parse
if "%~1"=="" goto :execute
if /i "%~1"=="--data_file" (
    set DATA_FILE=%~2
    shift
    shift
    goto :parse
)
if /i "%~1"=="--output_dir" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto :parse
)
if /i "%~1"=="--model_size" (
    set MODEL_SIZE=%~2
    shift
    shift
    goto :parse
)
shift
goto :parse

:execute
echo 使用数据文件: %DATA_FILE%
echo 输出目录: %OUTPUT_DIR%
echo 模型大小: %MODEL_SIZE%

REM 运行训练脚本
python train_advanced_torch.py --data_file "%DATA_FILE%" --output_dir "%OUTPUT_DIR%" --model_size "%MODEL_SIZE%"

echo ===================================
echo 高级PyTorch深度神经网络模型训练完成
echo ===================================

pause 