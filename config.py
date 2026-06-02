# config.py
import torch
import numpy as np


## Deguo
# # -------------------------- 路径配置 --------------------------
# POLARIZATION_FEATURE_PATH = "../data/Deguo/other/wall_Pol_features4.mat"  # 极化特征文件Dual/data/Deguo/other/wall_Pol_features2.mat
# RGB_IMAGE_PATH = "../data/Deguo//other/RGB4.mat"                          # RGB图像文件
# TRAIN_POS_PATH = "../data/Deguo/train_indicesl.npy"            # 训练集位置
# TRAIN_LABEL_PATH = "../data/Deguo/train_labelsl.npy"           # 训练集标签
# TEST_POS_PATH = "../data/Deguo/test_indicesl.npy"              # 测试集位置
# TEST_LABEL_PATH = "../data/Deguo/test_labelsl.npy"             # 测试集标签
# LABEL_PATH = "../data/Deguo/label"
# MODEL_SAVE_PATH = "vit.pth"                          # 模型保存路径
# RGB_NAME = 'RGB_crop'
# # RGB_NAME = 'A1crop_data'
# LABEL_NAME = 'label'
# # -------------------------- 数据配置 --------------------------
# CHAN = 23  # 极化特征通道数
# PATCH_SIZE = 17  # 图像切块大小
# IMAGE_HEIGHT = 920  # 原始图像高度
# IMAGE_WIDTH = 500  # 原始图像宽度
# SPATIAL_CHANNELS = 3  # RGB通道数
# NUM_CLASSES = 5  # 作物类别数（wheat/beat/bigwheat/corn/yumai）
# ONLY = False
# # -------------------------- 特征配置 --------------------------
# FEATURE_NAMES = [
#     'PdF3', 'PvF3', 'PsF3', 
#     'PsY4', 'PdY4', 'PvY4', 'PcY4',
#     'Ps6', 'Pd6', 'Pv6', 'Ph6', 'Pod', 'Pcd', 
#     'Ps3D', 'Pd3D', 'Pv3D', 
#     'alpha', 'beta', 'delta', 'gamma', 'lambda', 'H', 'Anisotropy'
# ]
# CROP_TYPES = [0,1,2,3,4]
# COLOR = np.array([
#         # [255, 255, 255],  # 背景 - 白色
#         # [0, 0, 255],      # 蓝色
#         # [255, 0, 0],      # 红色
#         # [255, 100, 0],    # 橙色
#         # [0, 255, 0],      # 绿色
#         # [255, 255, 0] ,    # 黄色
#         # [128, 0, 128],    # 紫色
#         # [0, 255, 255]     # 青色
#         [0,0,255],
#         [255,255,0],
#         [255,180,0],
#         [255,0,0],
#         [0,255,0], 
#     ], dtype=np.uint8)

#Flevoland
# -------------------------- 路径配置 --------------------------
POLARIZATION_FEATURE_PATH = "../data/flevoland/Fle_Pol_features.mat"  # 极化特征文件
RGB_IMAGE_PATH = "../data/flevoland/RGB.mat"                          # RGB图像文件
TRAIN_POS_PATH = "../data/flevoland/train_indices_balanced.npy"            # 训练集位置
TRAIN_LABEL_PATH = "../data/flevoland/train_labels_balanced.npy"           # 训练集标签
TEST_POS_PATH = "../data/flevoland/test_indices_balanced.npy"              # 测试集位置
TEST_LABEL_PATH = "../data/flevoland/test_labels_balanced.npy"             # 测试集标签
LABEL_PATH = "../data/flevoland/label"
MODEL_SAVE_PATH = "dual_fle.pth"                          # 模型保存路径
RGB_NAME = 'rgb'  # RGB变量名
# -------------------------- 数据配置 --------------------------
CHAN = 23 # 极化特征通道数
PATCH_SIZE = 17  # 图像切块大小
IMAGE_HEIGHT = 750  # 原始图像高度
IMAGE_WIDTH = 1024  # 原始图像宽度
SPATIAL_CHANNELS = 3  # RGB通道数
NUM_CLASSES = 15  # 作物类别数（wheat/beat/bigwheat/corn/yumai）
LABEL_NAME = 'data'
ONLY = False

COLOR = np.array( [
    [255, 0, 0],  # Red; Steambeans
    [90, 11, 226],  # Purple; Peas
    [0, 131, 74],  # Green; Forest
    [0, 252, 255],  # Teal; Lucerne
    [255, 182, 228],  # Pink; Wheat
    [184, 0, 255],  # Magenta; Beet
    [254, 254, 0],  # Yellow; Potatoes
    [170, 138, 79],  # Brown; Bare Soil
    [1, 254, 3],  # Light green; Grass
    [255, 127, 0],  # Orange; Rapeseed
    [146, 0, 1],  # Bordeaux; Barley
    [191, 191, 255],  # Lila; Wheat 2
    [191, 255, 192],  # Marine Green; Wheat 3
    [0, 0, 254],  # Blue; Water
    [255, 217, 160]  # Beige; Buildings
    ], dtype=np.uint8)




# -------------------------- 特征配置 --------------------------
FEATURE_NAMES = [
    'PdF3', 'PvF3', 'PsF3', 
     'PsY4', 'PdY4', 'PvY4', 'PcY4',
    'Ps6', 'Pd6', 'Pv6', 'Ph6', 'Pod', 'Pcd', 
     'Ps3D', 'Pd3D', 'Pv3D', 
     'alpha', 'beta', 'delta', 'gamma', 'lambda', 'H', 'Anisotropy'
]
# CROP_TYPES = ["Stem beans", "Peas", "Forest", "Lucerne", "Wheat", "Beet", 
#                "Potatoes", "Bare soil", "Grassess", "Rapeseed", "Barley", 
#                "Wheat 2", "Wheat 3", "Water", "Buildings"]
CROP_TYPES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]


# ###ESAR
# POLARIZATION_FEATURE_PATH = "../data/Esar/esar_Pol_features.mat"  # 极化特征文件
# RGB_IMAGE_PATH = "../data/Esar/rgb.mat"                          # RGB图像文件
# TRAIN_POS_PATH = "../data/Esar/train_indices_balanced.npy"            # 训练集位置
# TRAIN_LABEL_PATH = "../data/Esar/train_labels_balanced.npy"           # 训练集标签
# TEST_POS_PATH = "../data/Esar/test_indices_balanced.npy"              # 测试集位置
# TEST_LABEL_PATH = "../data/Esar/test_labels_balanced.npy"             # 测试集标签
# LABEL_PATH = "../data/Esar/NewLabel"
# LABEL_NAME = 'label'
# MODEL_SAVE_PATH = "Dual_Esar.pth"                          # 模型保存路径
# RGB_NAME = 'RGB'  # RGB变量名
# # -------------------------- 数据配置 --------------------------
# CHAN = 23 # 极化特征通道数
# PATCH_SIZE = 33  # 图像切块大小
# IMAGE_HEIGHT = 1300  # 原始图像高度
# IMAGE_WIDTH = 1200  # 原始图像宽度
# SPATIAL_CHANNELS = 3  # RGB通道数
# NUM_CLASSES = 5  # 类别数

# ONLY = False

# COLOR = np.array( [
#     [255, 0, 0],  # Red; Steambeans
#     [90, 11, 226],  # Purple; Peas
#     [0, 131, 74],  # Green; Forest
#     [0, 252, 255],  # Teal; Lucerne
#     [255, 182, 228],  # Pink; Wheat
#     ], dtype=np.uint8)




# # -------------------------- 特征配置 --------------------------
# FEATURE_NAMES = [
#     'PdF3', 'PvF3', 'PsF3', 
#      'PsY4', 'PdY4', 'PvY4', 'PcY4',
#     'Ps6', 'Pd6', 'Pv6', 'Ph6', 'Pod', 'Pcd', 
#      'Ps3D', 'Pd3D', 'Pv3D', 
#      'alpha', 'beta', 'delta', 'gamma', 'lambda', 'H', 'Anisotropy'
# ]
# CROP_TYPES = [1,2,3,4,5]





# 特征分组（物理意义分类）
FEATURE_GROUPS = {
    'surface': [2, 3, 7, 11, 13],      # PsF3, PsY4, Ps6, Pod, Ps3D
    'volume': [1, 5, 9, 15],           # PvF3, PvY4, Pv6, Pv3D
    'double_bounce': [0, 4, 8, 12, 14],# PdF3, PdY4, Pd6, Pcd, Pd3D
    'helix': [6, 10],                  # PcY4, Ph6
    'other': [16, 17, 18, 19, 20, 21, 22] # 其他特征
}

# 特征组先验权重（物理重要性）
GROUP_PRIORS = {
    'surface': 0.7,
    'volume': 0.8,
    'double_bounce': 0.6,
    'helix': 0.4,
    'other': 0.3
}




# -------------------------- 训练配置 --------------------------
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # 早停耐心值
CORR_PENALTY_LAMBDA = 0.01  # 相关性惩罚系数
REDUCE_LR_FACTOR = 0.5  # 学习率衰减因子
REDUCE_LR_PATIENCE = 5  # 学习率衰减耐心值
TRAIN = True
# -------------------------- 设备配置 --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTI_GPU = torch.cuda.device_count() > 1  # 是否使用多GPU
# MULTI_GPU = False  # 是否使用多GPU