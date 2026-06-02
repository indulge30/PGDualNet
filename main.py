import torch
from config import *
from data_utils import (
    patch, load_pos_and_labels, load_polarization_features,
    load_rgb_image, create_dataloader
)
from model_components import PhysicsInformedDualBranchNet,Base
from trainer import ModelTrainer

import numpy as np
from sklearn.model_selection import train_test_split

def main():
    # -------------------------- 1. 加载数据 --------------------------
    print("加载数据...")
    # 加载原始图像
    sar_image = load_polarization_features()  # 极化特征 [920, 500, 23]
    pauli_image = load_rgb_image()            # RGB图像 [920, 500, 3]
    # 切割为patch
    print("生成图像patch...")
    patch_polar = patch(sar_image, patch_size=PATCH_SIZE)  # [920, 500, 33, 33, 23]
    patch_rgb = patch(pauli_image, patch_size=PATCH_SIZE)   # [920, 500, 33, 33, 3]
    
    # 加载训练/测试集位置和标签
    train_pos, train_labels = load_pos_and_labels(TRAIN_POS_PATH, TRAIN_LABEL_PATH)
    test_pos, test_labels = load_pos_and_labels(TEST_POS_PATH, TEST_LABEL_PATH)
    unique_classes1 = np.unique(train_labels)
    unique_classes2 = np.unique(test_labels)
    print(unique_classes1,unique_classes2)

    print(f"训练样本数: {len(train_pos)}, 测试样本数: {len(test_pos)}")
    
    # 创建数据加载器
    train_loader = create_dataloader(
        positions=train_pos,
        labels=train_labels,
        patch_polar=patch_polar,
        patch_rgb=patch_rgb,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # 准备测试集数据（转移到DEVICE）
    polar_test = patch_polar[test_pos[:, 0], test_pos[:, 1]]  # [N, 33, 33, 23]
    rgb_test = patch_rgb[test_pos[:, 0], test_pos[:, 1]]      # [N, 33, 33, 3]
    # 转换为PyTorch格式并转移到DEVICE
    polar_test = torch.from_numpy(polar_test).float().permute(0, 3, 1, 2).to(DEVICE)
    rgb_test = torch.from_numpy(rgb_test).float().permute(0, 3, 1, 2).to(DEVICE)
    test_data = (polar_test, rgb_test)
    # -------------------------- 2. 初始化模型 --------------------------
    print("初始化模型...")
    # model = Base()
    model = PhysicsInformedDualBranchNet()

    # -------------------------- 3. 训练或加载模型 --------------------------
    if TRAIN:  # TRAIN 是 config.py 中定义的布尔值（True/False）
        print("开始训练流程...")
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            test_data=test_data,
            test_labels=test_labels
        )
        trainer.train()
        print(f"训练完成，最佳测试集准确率: {trainer.best_acc:.4f}")
    else:
        print("加载预训练模型...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)
        # 可选：加载后进行一次评估
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,  # 训练集加载器仅用于初始化，不实际训练
            test_data=test_data,
            test_labels=test_labels
        )
        print("预训练模型评估结果:")
        trainer.evaluate()

    # -------------------------- 4. 特征重要性分析（可选） --------------------------
    if not TRAIN:  # 仅在推理模式下分析
        print("提取特征重要性...")
        feature_importance = model.get_feature_importance().cpu().numpy()
        # 打印每个特征的重要性分数
        for name, importance in zip(FEATURE_NAMES, feature_importance):
            print(f"{name}: {importance:.4f}")


if __name__ == "__main__":
    main()
