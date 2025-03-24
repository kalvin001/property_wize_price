import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import joblib
from typing import List, Dict, Any, Tuple, Optional
import json
import math
from tqdm import tqdm, trange
import time

from models.model_interface import ModelInterface

class HousePriceNN(nn.Module):
    """
    房价预测深度神经网络模型
    """
    def __init__(self, input_size, hidden_layers=(256, 128, 64), dropout_rate=0.2):
        """
        初始化神经网络模型
        
        Args:
            input_size: 输入特征数量
            hidden_layers: 隐藏层大小的元组
            dropout_rate: Dropout比率
        """
        super(HousePriceNN, self).__init__()
        
        # 创建层列表
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))  # 添加批归一化层，有助于稳定训练
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        # 添加输出层
        layers.append(nn.Linear(prev_size, 1))
        
        # 创建模型
        self.model = nn.Sequential(*layers)
        
        # 初始化权重，防止梯度爆炸
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        使用适当的初始化方法初始化权重，防止梯度爆炸
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            预测结果
        """
        return self.model(x).squeeze()


class TorchNNModel(ModelInterface):
    """
    基于PyTorch的深度神经网络模型
    """
    
    def __init__(self, 
                 hidden_layers=(128, 64, 32), 
                 dropout_rate=0.3, 
                 learning_rate=0.0005,  # 降低学习率
                 batch_size=32,
                 num_epochs=5,
                 patience=15,
                 model_path: str = None, 
                 metadata: Dict[str, Any] = None):
        """
        初始化PyTorch神经网络模型
        
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
        super().__init__(model_path, metadata)
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        
        self.model = None
        self.scaler = None
        self.target_scaler = None  # 目标值的缩放器
        self._feature_importance = None
        
        # 初始化元数据
        if not self._metadata:
            self._metadata = {
                "model_type": "torch_nn",
                "hidden_layers": hidden_layers,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "patience": patience
            }
        
        # 检测设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, validation_split=0.1, **kwargs) -> None:
        """
        训练模型，并使用验证集进行评估和早停
        
        Args:
            X_train: 训练数据特征
            y_train: 训练数据标签
            validation_split: 验证集比例，默认0.1
            **kwargs: 其他训练参数
        """
        # 保存特征名
        self._feature_names = list(X_train.columns)
        
        # 对目标值进行缩放（这是解决loss过大的关键）
        self.target_scaler = MinMaxScaler()
        y_train_scaled = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        # 打印缩放前后的目标值范围
        print(f"目标值缩放前范围: {y_train.min()} - {y_train.max()}")
        print(f"目标值缩放后范围: {y_train_scaled.min()} - {y_train_scaled.max()}")
        
        # 分割训练集和验证集
        if validation_split > 0:
            from sklearn.model_selection import train_test_split
            X_train_part, X_val, y_train_part, y_val = train_test_split(
                X_train, y_train_scaled, test_size=validation_split, random_state=42
            )
            print(f"训练集大小: {len(X_train_part)}, 验证集大小: {len(X_val)}")
            
            # 转换验证集为PyTorch张量
            X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            X_train_part, y_train_part = X_train, y_train_scaled
            X_val, y_val = None, None
            val_loader = None
            
        # 转换训练数据为PyTorch张量
        X_train_tensor = torch.tensor(X_train_part.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_part, dtype=torch.float32)  # 使用缩放后的标签
        
        # 创建训练数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 创建模型
        input_size = X_train.shape[1]
        self.model = HousePriceNN(
            input_size=input_size,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # 使用Huber损失代替MSE，对异常值更鲁棒
        criterion = nn.HuberLoss(delta=1.0)
        
        # 使用带权重衰减（L2正则化）的Adam优化器
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5  # 添加L2正则化
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
        )
        
        # 训练模型
        print(f"开始训练PyTorch神经网络模型，共{self.num_epochs}轮...")
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # 记录训练和验证损失
        train_losses = []
        val_losses = []
        
        # 使用tqdm显示训练进度
        progress_bar = trange(self.num_epochs, desc="训练进度", ncols=100)
        
        for epoch in progress_bar:
            # 训练阶段
            self.model.train()
            running_train_loss = 0.0
            
            # 使用tqdm添加批次进度条
            batch_progress = tqdm(train_loader, desc=f"轮次 {epoch+1}/{self.num_epochs}", 
                                 leave=False, ncols=5)
            
            for inputs, targets in batch_progress:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播、计算损失
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 检查损失值是否合理，防止梯度爆炸
                if not torch.isfinite(loss):
                    print(f"警告: 损失值为无穷大或NaN: {loss.item()}")
                    print("跳过此批次更新...")
                    continue
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 反向传播、优化
                loss.backward()
                optimizer.step()
                
                running_train_loss += loss.item() * inputs.size(0)
                
                # 更新批次进度条描述
                batch_progress.set_postfix({"loss": f"{loss.item():.6f}"})
            
            epoch_train_loss = running_train_loss / len(train_dataset)
            train_losses.append(epoch_train_loss)
            
            # 验证阶段
            if val_loader is not None:
                self.model.eval()
                running_val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        val_loss = criterion(outputs, targets)
                        running_val_loss += val_loss.item() * inputs.size(0)
                
                epoch_val_loss = running_val_loss / len(val_dataset)
                val_losses.append(epoch_val_loss)
                
                # 学习率调度器更新（基于验证损失）
                scheduler.step(epoch_val_loss)
                
                # 更新进度条描述
                progress_bar.set_postfix({
                    "train_loss": f"{epoch_train_loss:.6f}",
                    "val_loss": f"{epoch_val_loss:.6f}", 
                    "best_val_loss": f"{best_val_loss:.6f}",
                    "patience": f"{patience_counter}/{self.patience}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # 早停（基于验证损失）
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    progress_bar.set_postfix({
                        "train_loss": f"{epoch_train_loss:.6f}",
                        "val_loss": f"{epoch_val_loss:.6f}", 
                        "best_val_loss": f"{best_val_loss:.6f}",
                        "patience": f"{patience_counter}/{self.patience}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "状态": "✓ 更新最佳模型"
                    })
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        progress_bar.set_postfix({
                            "train_loss": f"{epoch_train_loss:.6f}",
                            "val_loss": f"{epoch_val_loss:.6f}", 
                            "best_val_loss": f"{best_val_loss:.6f}",
                            "patience": f"{patience_counter}/{self.patience}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                            "状态": "! 早停触发"
                        })
                        print(f"早停触发，在{epoch+1}轮后停止训练")
                        print(f"最佳验证损失: {best_val_loss:.6f}，发生在第{epoch+1-patience_counter}轮")
                        break
            else:
                # 如果没有验证集，则基于训练损失进行早停和调度
                scheduler.step(epoch_train_loss)
                
                # 更新进度条描述
                progress_bar.set_postfix({
                    "train_loss": f"{epoch_train_loss:.6f}", 
                    "best_loss": f"{best_val_loss:.6f}",
                    "patience": f"{patience_counter}/{self.patience}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # 早停（基于训练损失）
                if epoch_train_loss < best_val_loss:
                    best_val_loss = epoch_train_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    progress_bar.set_postfix({
                        "train_loss": f"{epoch_train_loss:.6f}", 
                        "best_loss": f"{best_val_loss:.6f}",
                        "patience": f"{patience_counter}/{self.patience}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "状态": "✓ 更新最佳模型"
                    })
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        progress_bar.set_postfix({
                            "train_loss": f"{epoch_train_loss:.6f}", 
                            "best_loss": f"{best_val_loss:.6f}",
                            "patience": f"{patience_counter}/{self.patience}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                            "状态": "! 早停触发"
                        })
                        print(f"早停触发，在{epoch+1}轮后停止训练")
                        print(f"最佳训练损失: {best_val_loss:.6f}，发生在第{epoch+1-patience_counter}轮")
                        break
        
        # 加载最佳模型状态
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"已加载最佳模型（第{epoch+1-patience_counter}轮）")
        
        # 最终验证集评估（如果有）
        if val_loader is not None:
            self.model.eval()
            all_val_preds = []
            
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    all_val_preds.append(outputs.cpu().numpy())
            
            # 合并所有批次的预测
            all_val_preds = np.concatenate(all_val_preds)
            
            # 如果使用了缩放，需要反转缩放
            if self.target_scaler is not None:
                all_val_preds = self.target_scaler.inverse_transform(all_val_preds.reshape(-1, 1)).flatten()
                y_val_orig = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            else:
                y_val_orig = y_val
            
            # 计算验证集评估指标
            val_mse = mean_squared_error(y_val_orig, all_val_preds)
            val_rmse = math.sqrt(val_mse)
            val_mae = mean_absolute_error(y_val_orig, all_val_preds)
            val_r2 = r2_score(y_val_orig, all_val_preds)
            
            print(f"\n验证集评估结果:")
            print(f"- RMSE: {val_rmse:.4f}")
            print(f"- MAE: {val_mae:.4f}")
            print(f"- R²: {val_r2:.4f}")
            
            # 更新元数据中的验证评估结果
            val_metrics = {
                "val_mse": float(val_mse),
                "val_rmse": float(val_rmse),
                "val_mae": float(val_mae),
                "val_r2": float(val_r2)
            }
        else:
            val_metrics = None
        
        # 计算特征重要性
        print("\n计算特征重要性...")
        self._compute_feature_importance(X_train, y_train)
        
        # 更新元数据
        self._metadata.update({
            "training_epochs": epoch + 1,
            "early_stopped_epoch": epoch + 1 - patience_counter if patience_counter >= self.patience else None,
            "final_train_loss": float(epoch_train_loss),
            "best_val_loss": float(best_val_loss),
            "train_losses": [float(loss) for loss in train_losses],
            "val_losses": [float(loss) for loss in val_losses] if val_losses else None,
            "val_metrics": val_metrics,
            "input_features": self._feature_names,
            "model_architecture": str(self.model),
            "target_scale_min": float(self.target_scaler.data_min_[0]),
            "target_scale_max": float(self.target_scaler.data_max_[0]),
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 尝试绘制损失曲线
        try:
            self._plot_training_curves(train_losses, val_losses)
        except Exception as e:
            print(f"绘制损失曲线时出错: {e}")
        
        print("PyTorch神经网络模型训练完成！")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 预测数据特征
            
        Returns:
            预测结果数组
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法进行预测")
        
        # 确保使用相同的特征列顺序
        if self._feature_names:
            X = X[self._feature_names].copy()
        
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        
        # 设置为评估模式并预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        # 如果使用了目标缩放，需要反向转换
        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_test: 测试数据特征
            y_test: 测试数据标签
            
        Returns:
            包含评估指标的字典，如 RMSE, MAE, R² 等
        """
        # 获取预测结果（predict方法已处理了缩放问题）
        y_pred = self.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 计算更多指标
        median_ae = np.median(np.abs(y_test - y_pred))
        
        # 记录评估结果
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "median_ae": float(median_ae),
            "r2": float(r2)
        }
        
        # 更新元数据
        self._metadata["metrics"] = metrics
        self._metadata["evaluation_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 打印评估结果
        print(f"模型评估结果:")
        print(f"- RMSE: {rmse:.4f}")
        print(f"- MAE: {mae:.4f}")
        print(f"- R²: {r2:.4f}")
        
        return metrics
    
    def _compute_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        计算特征重要性（使用排列重要性方法）
        
        Args:
            X: 特征数据
            y: 标签数据
        """
        print("计算特征重要性...")
        
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
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            包含特征名称和重要性分数的DataFrame
        """
        if self._feature_importance is None:
            # 如果没有计算过特征重要性，返回空DataFrame
            return pd.DataFrame(columns=['feature', 'importance'])
        
        return self._feature_importance
    
    def save(self, path: Optional[str] = None) -> str:
        """
        保存模型到文件
        
        Args:
            path: 保存路径，如果为None，则使用实例化时提供的路径
            
        Returns:
            实际保存路径
        """
        if path is None and self._model_path is None:
            raise ValueError("未指定保存路径")
        
        save_path = path or self._model_path
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 保存模型
        model_dict = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'feature_names': self._feature_names,
            'feature_importance': self._feature_importance,
            'metadata': self._metadata,
            'target_scaler': self.target_scaler  # 保存目标缩放器
        }
        
        # 使用joblib保存，支持大文件
        joblib.dump(model_dict, save_path)
        
        # 同时保存元数据到JSON文件，方便查看
        meta_path = os.path.splitext(save_path)[0] + "_meta.json"
        with open(meta_path, 'w') as f:
            # 过滤掉不能序列化的字段
            metadata_copy = self._metadata.copy()
            if 'model_architecture' in metadata_copy:
                metadata_copy['model_architecture'] = str(metadata_copy['model_architecture'])
            json.dump(metadata_copy, f, indent=2)
        
        print(f"模型已保存到: {save_path}")
        print(f"元数据已保存到: {meta_path}")
        
        return save_path
    
    @classmethod
    def load(cls, path: str) -> 'TorchNNModel':
        """
        从文件加载模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            加载的TorchNNModel实例
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        try:
            # 加载模型字典
            model_dict = joblib.load(path)
            
            # 提取元数据
            metadata = model_dict.get('metadata', {})
            
            # 创建模型实例
            model = cls(
                hidden_layers=metadata.get('hidden_layers', (256, 128, 64)),
                dropout_rate=metadata.get('dropout_rate', 0.2),
                learning_rate=metadata.get('learning_rate', 0.001),
                batch_size=metadata.get('batch_size', 64),
                num_epochs=metadata.get('num_epochs', 100),
                patience=metadata.get('patience', 10),
                model_path=path,
                metadata=metadata
            )
            
            # 恢复特征名
            model._feature_names = model_dict.get('feature_names', [])
            
            # 恢复特征重要性
            model._feature_importance = model_dict.get('feature_importance', None)
            
            # 恢复目标缩放器
            model.target_scaler = model_dict.get('target_scaler', None)
            
            # 如果有模型状态，恢复模型
            model_state_dict = model_dict.get('model_state_dict')
            if model_state_dict is not None:
                # 创建模型结构
                input_size = len(model._feature_names)
                model.model = HousePriceNN(
                    input_size=input_size,
                    hidden_layers=model.hidden_layers,
                    dropout_rate=model.dropout_rate
                ).to(model.device)
                
                # 加载模型参数
                model.model.load_state_dict(model_state_dict)
                model.model.eval()  # 设置为评估模式
            
            print(f"模型已成功从 {path} 加载")
            if model.target_scaler:
                print(f"目标缩放器已加载，缩放范围: {model.target_scaler.data_min_[0]} - {model.target_scaler.data_max_[0]}")
            
            return model
            
        except Exception as e:
            raise ValueError(f"加载模型时出错: {e}")
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns:
            模型参数字典
        """
        return {
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "patience": self.patience
        }
    
    def set_params(self, **params) -> None:
        """
        设置模型参数
        
        Args:
            **params: 模型参数
        """
        for param, value in params.items():
            if param == 'hidden_layers':
                self.hidden_layers = value
            elif param == 'dropout_rate':
                self.dropout_rate = value
            elif param == 'learning_rate':
                self.learning_rate = value
            elif param == 'batch_size':
                self.batch_size = value
            elif param == 'num_epochs':
                self.num_epochs = value
            elif param == 'patience':
                self.patience = value
        
        # 更新元数据
        self._metadata.update(params) 
    
    def _plot_training_curves(self, train_losses, val_losses=None):
        """
        绘制训练和验证损失曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表（可选）
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(train_losses) + 1)
            
            plt.plot(epochs, train_losses, 'b-', label='训练损失')
            if val_losses:
                plt.plot(epochs, val_losses, 'r-', label='验证损失')
            
            plt.title('训练与验证损失曲线')
            plt.xlabel('轮次')
            plt.ylabel('损失')
            plt.legend()
            plt.grid(True)
            
            # 保存图片
            curves_dir = os.path.join('model', 'curves')
            os.makedirs(curves_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            fig_path = os.path.join(curves_dir, f'loss_curves_{timestamp}.png')
            plt.savefig(fig_path)
            plt.close()
            
            print(f"损失曲线已保存到: {fig_path}")
            
            # 添加路径到元数据
            self._metadata["loss_curves_path"] = fig_path
            
        except ImportError:
            print("无法导入matplotlib，跳过绘制损失曲线")
        except Exception as e:
            print(f"绘制损失曲线时出错: {e}") 