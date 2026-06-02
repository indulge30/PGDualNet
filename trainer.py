# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from config import *  # 导入全局配置
from typing import Tuple
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_data: tuple, test_labels: torch.Tensor):
        """
        Args:
            model: 待训练的双分支模型
            train_loader: 训练集DataLoader（含polar, rgb, label）
            test_data: 测试集数据 tuple(polar_test, rgb_test)，均为 [N, C, H, W]
            test_labels: 测试集标签 [N]
        """
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.polar_test, self.rgb_test = test_data  # 测试集特征（已转移到DEVICE）
        self.test_labels = test_labels  # 测试集标签（CPU，用于计算指标）
        
        # 多GPU支持
        # if MULTI_GPU:
        #     print(f"使用 {torch.cuda.device_count()} 块GPU训练")
        #     self.model = nn.DataParallel(self.model, device_ids = [0,1])
        
        # 训练组件初始化
        self.criterion = nn.CrossEntropyLoss()  # 分类损失
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=True
        )
        self.scaler = GradScaler()  # 混合精度训练
        
        # 训练状态跟踪
        self.best_acc = 0.0  # 最佳测试集准确率
        self.early_stop_counter = 0  # 早停计数器
        self.steps_per_epoch = len(train_loader)

    def train_one_epoch(self, epoch: int) -> float:
        """训练单个epoch，返回平均总损失"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (polar_batch, rgb_batch, label_batch) in enumerate(self.train_loader):
            # 数据转移到DEVICE
            polar_batch = polar_batch.to(DEVICE)
            rgb_batch = rgb_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            # 梯度清零
            self.optimizer.zero_grad()

            # 混合精度训练
            with autocast():
                # 前向传播
                # selected_features, selection_mask, info = self.model.module.polar_branch.feature_selector(polar_batch)
                info = {'selection_prob':1}
                logits, corr_penalty = self.model(polar_batch, rgb_batch)
                # 计算损失（分类损失 + 相关性惩罚）
                ce_loss = self.criterion(logits, label_batch)
                # 处理无效惩罚值（避免NaN）
                if ONLY == False:
                    if torch.isnan(corr_penalty).any() or torch.isinf(corr_penalty).any():
                        total_batch_loss = ce_loss
                    else:
                        corr_penalty = corr_penalty.mean() / 50.0  # 惩罚值缩放
                        total_batch_loss = ce_loss + CORR_PENALTY_LAMBDA * corr_penalty
                else:
                    total_batch_loss = ce_loss
            # 反向传播与参数更新
            self.scaler.scale(total_batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 累计损失
            total_loss += total_batch_loss.item() * polar_batch.size(0)
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader.dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | 平均训练损失: {avg_loss:.4f}")
        return avg_loss , info

    def evaluate(self) -> Tuple[float, float, float]:
        """评估模型在测试集上的性能，返回（准确率、平均精确率、平均召回率）"""
        self.model.eval()    
        batch_size = 32  # 根据您的GPU内存调整这个值
        num_batches = (len(self.test_labels) + batch_size - 1) // batch_size
        
        all_preds = []
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.test_labels))
                
                # 获取当前批次的数据
                polar_batch = self.polar_test[start_idx:end_idx]
                rgb_batch = self.rgb_test[start_idx:end_idx]
                
                # 前向传播
                logits, _ ,_ = self.model(polar_batch, rgb_batch)
                y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(y_pred)
                
                # 释放内存
                del logits, y_pred, polar_batch, rgb_batch
                torch.cuda.empty_cache()  # 清理GPU缓存
          # 计算分类指标
        acc = accuracy_score(self.test_labels, all_preds)
        avg_precision = precision_score(self.test_labels, all_preds, average='macro', zero_division=0)
        avg_recall = recall_score(self.test_labels, all_preds, average='macro', zero_division=0)
        
        # 打印指标
        print("="*50)
        print(f"测试集性能 | 准确率: {acc:.4f} | 平均精确率: {avg_precision:.4f} | 平均召回率: {avg_recall:.4f}")
        print("="*50)
        
        return acc, avg_precision, avg_recall
        
    def train(self) -> None:
        """完整训练流程（含早停、模型保存）"""
        print("开始训练...")
        for epoch in range(EPOCHS):
            # 训练1个epoch
            _ , info = self.train_one_epoch(epoch)
            torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
            # 每10个epoch评估一次
            if (epoch + 1) % 10 == 0:
                current_acc, _ ,_ = self.evaluate()
                
                # 学习率调度（基于测试集性能，此处简化为基于训练损失，可按需修改）
                self.scheduler.step(current_acc)  # 也可传入训练损失：self.scheduler.step(avg_loss)
                # probs = info['selection_probs'].cpu().numpy()
                # print(probs)
           
                # 保存最佳模型
                if current_acc > self.best_acc:
                    self.best_acc = current_acc
                    self.early_stop_counter = 0  # 重置早停计数器
                
                    import os
                    # 检查保存目录是否存在，不存在则创建
                    save_dir = os.path.dirname(MODEL_SAVE_PATH)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                        print(f"📁 创建保存目录: {save_dir}")
                    
                    # 执行保存
                    torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
                    # 验证是否保存成功
                    if os.path.exists(MODEL_SAVE_PATH):
                        file_size = os.path.getsize(MODEL_SAVE_PATH) / 1024 / 1024  # 转为MB
                        print(f"✅ 模型保存成功！路径: {MODEL_SAVE_PATH} | 大小: {file_size:.2f}MB")
                    else:
                        print(f"❌ 模型保存失败！路径: {MODEL_SAVE_PATH}")
                else:
                    self.early_stop_counter += 1
                    print(f"❌ 测试集精度未提升，连续下降次数: {self.early_stop_counter}/{PATIENCE}")
                    