# 房产估价分析系统

这是一个基于机器学习的房产估价分析系统，使用XGBoost算法构建预测模型，并提供可解释性分析。

## 项目结构

```
project/
├── backend/                # FastAPI后端
│   ├── main.py             # 主应用文件
│   └── requirements.txt    # Python依赖
├── frontend/               # Next.js前端
│   ├── pages/              # 页面组件
│   ├── components/         # 可复用组件
│   ├── styles/             # 样式文件
│   ├── public/             # 静态资源
│   └── package.json        # npm依赖
├── model/                  # 模型文件（不包含在Git仓库中）
│   ├── xgb_model.joblib    # XGBoost模型
│   ├── feature_cols.joblib # 特征列名
│   └── optimizer_logs/     # 参数优化日志
└── resources/              # 数据资源
    └── house_samples_features.csv  # 房产特征数据
```

## 优化模型训练与分析

本项目强调模型的优化和可解释性，主要包括以下功能：

1. **参数优化**：支持网格搜索和贝叶斯优化两种方式进行XGBoost参数搜索
2. **特征选择**：使用递归特征消除(RFECV)优化特征子集
3. **优化过程可视化**：记录并可视化每次参数搜索的结果
4. **模型解释性分析**：使用SHAP值解释模型预测

### 运行优化和训练

使用`run_analysis.py`脚本可以轻松运行模型优化和训练：

```bash
# 使用默认配置(贝叶斯优化)
python run_analysis.py

# 使用快速配置(网格搜索，更少的参数组合)
python run_analysis.py --preset quick

# 使用彻底配置(更多迭代次数，更精细的参数搜索)
python run_analysis.py --preset thorough

# 使用自定义参数
python run_analysis.py --custom_args "--optimization bayes --cv_folds 5 --bayes_iterations 30"
```

### 可视化和分析优化结果

```bash
# 可视化最近一次的优化结果
python run_analysis.py --visualize

# 可视化指定目录的优化结果
python run_analysis.py --log_dir model/optimizer_logs/20230615_120000_bayes
```

可视化结果将保存在优化日志目录下的`analysis`子目录中，包括：
- 优化进度图：RMSE随迭代次数的变化
- 参数影响图：每个参数对模型性能的影响
- 迭代数据表：CSV格式的详细迭代记录

## 功能特点

1. **房产估价预测**：基于房产的多种特征，预测房产价格
2. **可解释性分析**：使用SHAP值解释哪些特征对价格有影响
3. **特征重要性排名**：展示最重要的房产特征及其影响程度
4. **样本分析**：展示典型房产的估价分析和误差解释
5. **参数优化记录**：记录每次优化的搜索轨迹和结果

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
python run_analysis.py

# 或直接运行底层脚本
python analyze_data.py --optimization bayes --feature_selection --bayes_iterations 20
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
   - 构建前运行: `python analyze_data.py` (生成模型文件) 