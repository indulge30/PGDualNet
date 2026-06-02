# data_utils.py
import numpy as np
import tifffile
import os
from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader
import torch
from config import *  # 导入全局配置
from typing import Tuple
import matplotlib.pyplot as plt


def patch(data: np.ndarray, patch_size: int = PATCH_SIZE, 
          m_H: int = IMAGE_HEIGHT, m_W: int = IMAGE_WIDTH) -> np.ndarray:
    """
    将图像切割为重叠patch（用于后续逐像素推理）
    优化点：使用NumPy向量化操作替代三重循环，速度提升10-100倍
    Args:
        data: 输入图像 [H, W, C]
        patch_size: patch大小（奇数）
        m_H: 原始图像高度
        m_W: 原始图像宽度
    Returns:
        new_patch_data: 切割后的patch [H, W, patch_size, patch_size, C]
    """
    channels = data.shape[-1]
    startid = (patch_size - 1) // 2  # 计算填充量
    
    # 1. 填充图像（保持与原逻辑一致的填充方式）
    data_padded = np.pad(
        data, 
        pad_width=(
            (startid, startid),  # 高度方向前后各填充startid
            (startid, startid),  # 宽度方向前后各填充startid
            (0, 0)               # 通道方向不填充
        ),
        mode='constant'  # 用0填充（与原逻辑一致）
    )
    
    # 2. 向量化生成所有patch（核心优化）
    # 生成每个patch的坐标网格
    # 输出形状: [m_H, m_W, patch_size, patch_size, channels]
    new_patch_data = np.lib.stride_tricks.sliding_window_view(
        data_padded, 
        window_shape=(patch_size, patch_size, channels)  # 滑动窗口大小
    )
    new_patch_data = new_patch_data.squeeze()  # 去掉多余的维度
    print(new_patch_data.shape)
    # 3. 裁剪到原始图像尺寸（移除因滑动窗口产生的多余维度）
    # 因为padding后尺寸是(m_H + 2*startid, m_W + 2*startid, channels)
    # 滑动窗口后正好得到(m_H, m_W, patch_size, patch_size, channels)
    return new_patch_data

def load_pos_and_labels(pos_path: str = TRAIN_POS_PATH, label_path: str = TRAIN_LABEL_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """加载训练/测试集的位置和标签"""
    positions = np.load(pos_path)
    labels = np.load(label_path)
    return positions, labels

def load_full_ground_truth(mat_path: str = LABEL_PATH) -> np.ndarray:
    mat_data = loadmat(mat_path)
    print(mat_data.keys())
    label = mat_data[LABEL_NAME]  # 假设 .mat 文件中标签数据的键是 'la
    return label
def load_polarization_features(mat_path: str = POLARIZATION_FEATURE_PATH) -> np.ndarray:
    """加载极化特征（从mat文件），返回 [H, W, CHAN]"""
    mat_data = loadmat(mat_path)
    # 按特征名提取并拼接通道
# 定义需要提取的变量列表（与保存时的变量名对应）
    variables = [
    'PdF3', 'PvF3', 'PsF3', 
     'PsY4', 'PdY4', 'PvY4', 'PcY4',
    'Ps6', 'Pd6', 'Pv6', 'Ph6', 'Pod', 'Pcd', 
     'Ps3D', 'Pd3D', 'Pv3D', 
     'alpha', 'beta', 'delta', 'gamma', 'lambda', 'H', 'Anisotropy'
    ]
    

# 存储特征的列表
    features_list = []
    for name in variables:
        # 提取特征并去除多余维度
        feat = np.squeeze(mat_data[name])
        # 确保是NumPy数组
        if not isinstance(feat, np.ndarray):
            feat = np.array(feat)
        # 增加通道维度
        features_list.append(feat[..., np.newaxis])
    
    # 关键：将列表转换为NumPy数组（拼接操作）
    combined_array = np.concatenate(features_list, axis=-1)
    print(combined_array.shape)
    combined_array = combined_array.reshape(IMAGE_HEIGHT,IMAGE_WIDTH,CHAN)
    # combined_array = combined_array[:,:,:3]


    # 现在可以安全地使用.shape属性了
    return combined_array

def load_rgb_image(mat_path: str = RGB_IMAGE_PATH) -> np.ndarray:
    """加载RGB图像（从mat文件），返回 [H, W, 3]"""
    mat_data = loadmat(mat_path)
    return np.squeeze(mat_data[RGB_NAME])  # [H, W, 3]


def plot_selection_probs(selection_probs, feature_names):
    if isinstance(selection_probs, torch.Tensor):
        selection_probs = selection_probs.detach().cpu().numpy()  # ✅ 转 numpy
    # --- 适配不同维度 ---
    if selection_probs.ndim == 2:   # [B, C]
        selection_probs = selection_probs.mean(axis=0)
    elif selection_probs.ndim == 3: # [C, H, W]
        selection_probs = selection_probs.mean(axis=(1, 2))

    plt.figure(figsize=(10, 4))
    idx = np.arange(len(feature_names))
    plt.bar(idx, selection_probs, color="skyblue")
    plt.xticks(idx, feature_names, rotation=90)
    plt.ylabel("Selection Probability")
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig('selection.jpg', dpi=300)
    plt.close()

def create_dataloader(positions: np.ndarray, labels: np.ndarray, 
                      patch_polar: np.ndarray, patch_rgb: np.ndarray, 
                      batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    """
    创建DataLoader（极化特征+RGB特征+标签）
    Args:
        positions: 样本位置 [N, 2]（H, W坐标）
        labels: 样本标签 [N]
        patch_polar: 极化特征patch [H, W, patch_size, patch_size, CHAN]
        patch_rgb: RGB特征patch [H, W, patch_size, patch_size, 3]
        batch_size: 批次大小
        shuffle: 是否打乱
    Returns:
        DataLoader: 包含 (polar_feat, rgb_feat, label) 的数据加载器
    """
    # 按位置提取patch
    polar_feats = patch_polar[positions[:, 0], positions[:, 1]]  # [N, patch_size, patch_size, CHAN]
    rgb_feats = patch_rgb[positions[:, 0], positions[:, 1]]      # [N, patch_size, patch_size, 3]
    
    # 转换为PyTorch格式 [B, C, H, W]
    polar_tensor = torch.from_numpy(polar_feats).float().permute(0, 3, 1, 2)
    rgb_tensor = torch.from_numpy(rgb_feats).float().permute(0, 3, 1, 2)
    label_tensor = torch.from_numpy(labels).long()
    
    # 创建数据集和加载器
    dataset = TensorDataset(polar_tensor, rgb_tensor, label_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)