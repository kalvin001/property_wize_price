# 房产价格预测系统

基于机器学习的房产价格预测系统，通过特征工程和多种模型训练来提供准确的房产价格估算。

## 项目结构

- `feature_engineering.py`: 特征工程脚本，用于处理原始数据并创建新特征
- `train_models.py`: 原始模型训练脚本
- `train_models_enhanced.py`: 增强版模型训练脚本，使用经过特征工程的数据
- `run_feature_engineering_and_models.py`: 完整工作流脚本，从特征工程到模型训练和评估
- `models/`: 模型定义目录
- `resources/`: 数据资源目录
- `model/`: 基础模型输出目录
- `model_enhanced/`: 增强模型输出目录

## 特征工程

特征工程脚本(`feature_engineering.py`)通过以下步骤优化原始数据：

1. **数据加载**: 读取原始数据集
2. **处理缺失值**: 根据各列特性处理缺失值
3. **新特征创建**: 生成多种新特征，包括：
   - 价格比率特征（如价格与土地价值比）
   - 区域特征（区域溢价指数等）
   - 空间特征组合
   - 属性特征衍生
4. **异常值处理**: 使用Z分数方法识别和处理异常值
5. **特征编码**: 对分类特征进行编码
6. **低方差特征移除**: 移除对预测贡献小的特征

## 模型训练

支持多种模型类型，包括：

- XGBoost
- LightGBM
- CatBoost
- 随机森林

### 基础版模型训练

```bash
python train_models.py --model_types xgboost lightgbm
```

### 增强版模型训练

使用特征工程后的数据：

```bash
python train_models_enhanced.py --data_file resources/house_samples_engineered.csv --output_dir model_enhanced --feature_selection
```

## 完整工作流

一键运行从特征工程到模型训练和评估的完整流程：

```bash
python run_feature_engineering_and_models.py --model_types xgboost lightgbm
```

可选参数：
- `--skip_feature_engineering`: 跳过特征工程步骤
- `--skip_outlier_handling`: 特征工程中跳过离群值处理
- `--feature_selection`: 启用特征选择
- `--top_n_features`: 指定使用的特征数量

## 评估指标

模型评估使用多种指标：

- RMSE (均方根误差)
- MAE (平均绝对误差)
- R² (决定系数)
- 平均百分比误差
- 中位百分比误差
- 误差分布（各百分位数和误差范围分布）

## 环境要求

- Python 3.8+
- pandas, numpy, scikit-learn
- XGBoost, LightGBM, CatBoost (可选)
- matplotlib, seaborn (可视化，可选)

## 使用说明

1. 确保已安装所有依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行特征工程：
   ```bash
   python feature_engineering.py
   ```

3. 训练模型：
   ```bash
   python train_models_enhanced.py
   ```

4. 或者一键执行完整流程：
   ```bash
   python run_feature_engineering_and_models.py
   ```

## 技术栈

### 后端

- **FastAPI**：高性能API框架
- **XGBoost**：高性能梯度提升树模型，用于房价预测
- **Scikit-Optimize**：贝叶斯优化库
- **SHAP**：用于模型解释性分析
- **Pandas & NumPy**：数据处理
- **Matplotlib**：结果可视化

### 前端

- **Next.js**：React框架，支持服务端渲染
- **Ant Design**：UI组件库
- **React**：前端库
- **CSS Modules**：组件级样式隔离

## 安装与运行

### 数据准备与模型训练

```bash
# 分析数据并训练模型
python run_feature_engineering_and_models.py

# 或直接运行底层脚本
python train_models_enhanced.py
```

### 后端

```bash
cd backend
pip install -r requirements.txt
python main.py
```

服务将在 http://localhost:8102 上运行。

### 前端

```bash
cd frontend
npm install
npm run dev
```

应用将在 http://localhost:8101 上运行。

## API端点

- GET `/`: 基本欢迎信息
- GET `/api/health`: 健康检查
- GET `/api/model/info`: 获取模型信息
- GET `/api/properties/sample`: 获取样本房产分析结果
- POST `/api/predict`: 基于输入特征预测房价

## 部署

本项目可以通过GitHub和Vercel进行部署。

### GitHub

1. 创建一个新的GitHub仓库
2. 将代码推送到该仓库（注意：模型文件不会被上传）

```bash
git init
git add .
git commit -m "update"
git branch -M main
git remote add origin git@github.com:kalvin001/property_wize_price.git
git push -u origin main
```

> **注意**：模型文件 (`model/*.joblib`) 不会被上传到GitHub，需要在部署环境中重新生成。

### Vercel

1. 在Vercel上创建新项目
2. 连接到GitHub仓库
3. 配置构建设置：
   - 前端构建命令: `cd frontend && npm install && npm run build`
   - 输出目录: `frontend/.next`
   - 构建前运行: `python train_models_enhanced.py` (生成模型文件) 