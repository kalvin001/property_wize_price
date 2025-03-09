# 房产估价分析系统

这是一个基于机器学习的房产估价分析系统，使用XGBoost算法构建预测模型，并提供可解释性分析。

## 项目结构

```
project/
├── backend/           # FastAPI后端
│   ├── main.py        # 主应用文件
│   └── requirements.txt  # Python依赖
├── frontend/          # React前端
│   ├── src/           # 源代码
│   ├── public/        # 静态资源
│   └── package.json   # npm依赖
├── model/             # 模型文件（不包含在Git仓库中）
│   ├── xgb_model.joblib  # XGBoost模型
│   └── feature_cols.joblib  # 特征列名
└── resources/         # 数据资源
    └── house_samples_features.csv  # 房产特征数据
```

## 功能特点

1. **房产估价预测**：基于房产的多种特征，预测房产价格
2. **可解释性分析**：使用SHAP值解释哪些特征对价格有影响
3. **特征重要性排名**：展示最重要的房产特征及其影响程度
4. **样本分析**：展示典型房产的估价分析和误差解释

## 技术栈

### 后端

- **FastAPI**：高性能API框架
- **XGBoost**：高性能梯度提升树模型，用于房价预测
- **SHAP**：用于模型解释性分析
- **Pandas & NumPy**：数据处理

### 前端

- **React**：前端框架
- **Ant Design**：UI组件库
- **TypeScript**：类型安全的JavaScript
- **Vite**：现代前端构建工具

## 安装与运行

### 数据准备与模型训练

```bash
# 分析数据并训练模型
python analyze_data.py
```

### 后端

```bash
cd backend
pip install -r requirements.txt
python main.py
```

服务将在 http://localhost:8000 上运行。

### 前端

```bash
cd frontend
npm install
npm start
```

应用将在 http://localhost:3001 上运行。

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
   - 输出目录: `frontend/dist`
   - 构建前运行: `python analyze_data.py` (生成模型文件) 