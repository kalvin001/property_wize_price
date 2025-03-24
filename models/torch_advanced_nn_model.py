import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib
from typing import List, Dict, Any, Tuple, Optional
import json
import math
from tqdm import tqdm, trange

from models.model_interface import ModelInterface
from models.torch_nn_model import TorchNNModel

class ResidualBlock(nn.Module):
    """
    残差块，用于构建残差网络
    """
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # 如果输入和输出维度不同，添加一个线性层进行调整
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        out += identity
        return self.relu(out)

class AdvancedHousePriceNN(nn.Module):
    """
    高级房价预测深度神经网络模型，使用残差连接
    """
    def __init__(self, input_size, hidden_sizes=(256, 256, 128, 128, 64), dropout_rate=0.3):
        super(AdvancedHousePriceNN, self).__init__()
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 残差块
        res_blocks = []
        for i in range(len(hidden_sizes) - 1):
            res_blocks.append(ResidualBlock(hidden_sizes[i], hidden_sizes[i+1]))
        
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        使用合适的初始化方法初始化网络权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        """前向传播"""
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x.squeeze()

class TorchAdvancedNNModel(TorchNNModel):
    """
    基于PyTorch的高级深度神经网络模型，使用残差连接
    继承自TorchNNModel，复用其中的方法
    """
    
    def __init__(self, 
                 hidden_layers=(256, 256, 128, 128, 64), 
                 dropout_rate=0.3, 
                 learning_rate=0.0005,
                 batch_size=64,
                 num_epochs=200,
                 patience=30,
                 model_path: str = None, 
                 metadata: Dict[str, Any] = None):
        """
        初始化高级PyTorch神经网络模型
        
        Args:
            hidden_layers: 隐藏层大小的元组
            dropout_rate: Dropout比率
            learning_rate: 学习率
            batch_size: 批次大小
            num_epochs: 训练轮数
            patience: 早停耐心值
            model_path: 模型保存路径
            metadata: 模型元数据
        """
        super().__init__(
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            model_path=model_path,
            metadata=metadata
        )
        
        # 更新元数据类型
        if not self._metadata:
            self._metadata = {
                "model_type": "torch_advanced_nn",
                "hidden_layers": hidden_layers,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "patience": patience
            }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        训练模型
        
        Args:
            X_train: 训练数据特征
            y_train: 训练数据标签
            **kwargs: 其他训练参数
        """
        # 保存特征名
        self._feature_names = list(X_train.columns)
        
        # 转换数据为PyTorch张量
        X = torch.tensor(X_train.values, dtype=torch.float32)
        y = torch.tensor(y_train.values, dtype=torch.float32)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 创建高级模型
        input_size = X_train.shape[1]
        self.model = AdvancedHousePriceNN(
            input_size=input_size,
            hidden_sizes=self.hidden_layers,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, verbose=True, min_lr=1e-6
        )
        
        # 训练模型
        print(f"开始训练高级PyTorch神经网络模型，共{self.num_epochs}轮...")
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # 使用tqdm显示训练进度
        progress_bar = trange(self.num_epochs, desc="训练进度", ncols=100)
        
        for epoch in progress_bar:
            self.model.train()
            running_loss = 0.0
            
            # 使用tqdm添加批次进度条
            batch_progress = tqdm(train_loader, desc=f"轮次 {epoch+1}/{self.num_epochs}", 
                                 leave=False, ncols=100)
            
            for inputs, targets in batch_progress:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播、计算损失
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播、优化
                loss.backward()
                # 梯度剪裁，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
                # 更新批次进度条描述
                batch_progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
            epoch_loss = running_loss / len(train_dataset)
            
            # 学习率调度器更新
            scheduler.step(epoch_loss)
            
            # 更新进度条描述
            progress_bar.set_postfix({
                "loss": f"{epoch_loss:.4f}", 
                "best_loss": f"{best_loss:.4f}",
                "patience": f"{patience_counter}/{self.patience}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
            })
            
            # 早停
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                progress_bar.set_postfix({
                    "loss": f"{epoch_loss:.4f}", 
                    "best_loss": f"{best_loss:.4f}",
                    "patience": f"{patience_counter}/{self.patience}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                    "状态": "✓ 更新最佳模型"
                })
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    progress_bar.set_postfix({
                        "loss": f"{epoch_loss:.4f}", 
                        "best_loss": f"{best_loss:.4f}",
                        "patience": f"{patience_counter}/{self.patience}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                        "状态": "! 早停触发"
                    })
                    print(f"早停触发，在{epoch+1}轮后停止训练")
                    break
        
        # 加载最佳模型状态
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 计算特征重要性
        print("计算特征重要性...")
        self._compute_feature_importance(X_train, y_train)
        
        # 更新元数据
        self._metadata.update({
            "training_epochs": epoch + 1,
            "final_loss": float(best_loss),
            "input_features": self._feature_names,
            "model_architecture": str(self.model)
        })
        
        print("高级PyTorch神经网络模型训练完成！")
    
    @classmethod
    def load(cls, path: str) -> 'TorchAdvancedNNModel':
        """
        从指定路径加载模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        # 确保文件存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # 读取元数据
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        # 加载模型数据
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.load(path, map_location=device)
        
        # 获取模型参数
        input_size = data.get('input_size')
        hidden_layers = data.get('hidden_layers')
        dropout_rate = data.get('dropout_rate')
        feature_names = data.get('feature_names')
        
        # 创建实例
        nn_model = cls(
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            model_path=path,
            metadata=metadata
        )
        
        # 创建模型并加载权重
        if input_size:
            nn_model.model = AdvancedHousePriceNN(
                input_size=input_size,
                hidden_sizes=hidden_layers,
                dropout_rate=dropout_rate
            ).to(device)
            nn_model.model.load_state_dict(data['model_state_dict'])
            nn_model.model.eval()  # 设置为评估模式
        
        # 设置特征名
        nn_model._feature_names = feature_names
        
        # 设置特征重要性
        if data.get('feature_importance'):
            nn_model._feature_importance = pd.DataFrame(data['feature_importance'])
        
        nn_model.device = device
        
        print(f"已加载高级PyTorch神经网络模型: {path}")
        return nn_model 

    # 覆盖父类的方法，增加进度显示
    def _compute_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        计算特征重要性（使用排列重要性方法）
        
        Args:
            X: 特征数据
            y: 标签数据
        """
        print("计算高级模型特征重要性...")
        
        # 获取基准性能
        baseline_pred = self.predict(X)
        baseline_mse = mean_squared_error(y, baseline_pred)
        
        # 计算每个特征的重要性
        importances = []
        
        # 添加进度条
        feature_progress = tqdm(X.columns, desc="特征重要性计算", ncols=100)
        
        for col in feature_progress:
            feature_progress.set_postfix({"当前特征": col})
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)
            permuted_pred = self.predict(X_permuted)
            permuted_mse = mean_squared_error(y, permuted_pred)
            
            # 重要性为混淆后误差与基准误差的差值
            importance = permuted_mse - baseline_mse
            importances.append(importance)
        
        # 归一化重要性
        importances = np.array(importances)
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        # 创建特征重要性DataFrame
        self._feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"特征重要性计算完成。最重要的5个特征: {', '.join(self._feature_importance['feature'].head(5).tolist())}") 