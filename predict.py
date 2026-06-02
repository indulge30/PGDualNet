import torch
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from data_utils import patch  # 复用之前的patch切割函数
from model_components import PhysicsInformedDualBranchNet,Base
from config import *  # 导入全局配置
from typing import Tuple,Dict
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import torch.nn as nn
from data_utils import (
    patch, load_polarization_features,
    load_rgb_image, create_dataloader,load_full_ground_truth,plot_selection_probs
)

def load_pretrained_model(model_path: str = MODEL_SAVE_PATH) -> nn.Module:
    """加载预训练的双分支模型"""
    # 初始化模型
    model = PhysicsInformedDualBranchNet()
    # model = Base()
    # 加载权重（处理多GPU保存的权重）
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    # 若模型是多GPU训练的（权重键含"module."），则适配单GPU加载
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    print(MODEL_SAVE_PATH)
    model.load_state_dict(state_dict)
    # 设置为推理模式
    model = model.to(DEVICE)
    model.eval()
    print(f"✅ 预训练模型加载成功！路径: {model_path}")
    return model

def reconstruct_selected_maps(selected_maps_list, coords, full_shape, patch_size):
    """
    把所有patch的selected_maps拼接成整幅图
    selected_maps_list: list of [B, C, H, W] 张量
    coords: 对应patch左上角坐标的列表 [(i, j), ...]
    full_shape: (C, H_full, W_full) 全图的形状
    patch_size: patch大小 (ph, pw)
    """
    C, H_full, W_full = full_shape
    ph, pw = patch_size
    full_maps = torch.zeros((C, H_full, W_full))
    count_maps = torch.zeros((C, H_full, W_full))

    for fmap, (i, j) in zip(selected_maps_list, coords):
        # fmap: [C, ph, pw]，注意取单个patch
        full_maps[:, i:i+ph, j:j+pw] += fmap
        count_maps[:, i:i+ph, j:j+pw] += 1

    # 避免除0
    count_maps[count_maps == 0] = 1
    full_maps = full_maps / count_maps
    return full_maps

def infer_full_image(model: nn.Module, polar_full: np.ndarray, rgb_full: np.ndarray) -> np.ndarray:
    """
    全图分类推理：对完整极化图像和RGB图像进行逐patch推理，返回全图分类结果
    Args:
        model: 预训练模型（推理模式）
        polar_full: 完整极化图像 [H, W, C_polar]
        rgb_full: 完整RGB图像 [H, W, C_rgb]
    Returns:
        full_pred: 全图分类结果 [H, W]（每个像素对应类别标签）
    """
    H, W = polar_full.shape[0], polar_full.shape[1]
    C_polar = polar_full.shape[-1]
    C_rgb = rgb_full.shape[-1]
    
    # 1. 对全图切割patch（复用data_utils的patch函数，输出形状：[H, W, patch_size, patch_size, C]）
    print("🔄 切割全图为patch...")
    polar_patches = patch(polar_full, patch_size=PATCH_SIZE)  # [H, W, 33, 33, 23]
    rgb_patches = patch(rgb_full, patch_size=PATCH_SIZE)      # [H, W, 33, 33, 3]
    
    # 2. 初始化全图预测结果数组
    full_pred = np.zeros((H, W), dtype=np.int32)  # 存储每个像素的类别标签
    batch_size = 64  # 推理批次大小（根据显存调整，越大越快）
    
    # 3. 生成所有像素的坐标（用于批量取patch）
    coords = [(i, j) for i in range(H) for j in range(W)]
    total_patches = len(coords)
    print(f"📊 全图共 {total_patches} 个patch，分批次推理（batch_size={batch_size}）...")
    full_maps = np.zeros((23, H, W), dtype=np.float32)
    count_maps = np.zeros((1, H, W), dtype=np.float32)
    all_selection_probs = []  
    ph, pw = PATCH_SIZE, PATCH_SIZE

    # 4. 批量推理
    with torch.no_grad():  # 禁用梯度，加速推理
        for idx in range(0, total_patches, batch_size):
            # 取当前批次的坐标
            batch_coords = coords[idx:idx+batch_size]
            # 提取当前批次的polar和rgb patch
            batch_polar = []
            batch_rgb = []
            for (i, j) in batch_coords:
                # 从全图patch数组中取单个patch：[33, 33, C]
                polar_patch = polar_patches[i, j]
                rgb_patch = rgb_patches[i, j]
                batch_polar.append(polar_patch)
                batch_rgb.append(rgb_patch)
            
            # 转换为PyTorch格式：[B, C, H, W]
            batch_polar = torch.from_numpy(np.array(batch_polar)).float().permute(0, 3, 1, 2).to(DEVICE)
            batch_rgb = torch.from_numpy(np.array(batch_rgb)).float().permute(0, 3, 1, 2).to(DEVICE)
            
            # 模型推理（仅取分类logits，忽略相关性惩罚）
            logits, prob , selected_features= model(batch_polar, batch_rgb)
            
            # 取概率最大的类别作为预测结果
            batch_pred = torch.argmax(logits, dim=1).cpu().numpy()  # [B]
            batch_prob_mean = prob.mean(dim=(2, 3))  # [B, C]
            all_selection_probs.append(batch_prob_mean.cpu().numpy())
            # 将批次结果写入全图预测数组
            for k, (i, j) in enumerate(batch_coords):
                fmap = selected_features[k].detach().cpu().numpy()  # [C, ph, pw]

                h, w = fmap.shape[1], fmap.shape[2]

                # 裁剪，避免超出边界
                h = min(h, full_maps.shape[1] - i)
                w = min(w, full_maps.shape[2] - j)

                full_maps[:, i:i+h, j:j+w] += fmap[:, :h, :w]
                count_maps[:, i:i+h, j:j+w] += 1
                full_pred[i:i+ph, j:j+pw] = batch_pred[k]
        # plot_selection_probs(prob, FEATURE_NAMES)
        final_selected_maps = full_maps / np.maximum(count_maps, 1)
        all_selection_probs = np.vstack(all_selection_probs)

# # 调用分析函数
#         analyze_selection_probs(all_selection_probs, CROP_TYPES , FEATURE_NAMES,CROP_TYPES)
#         plot_feature_maps(final_selected_maps, FEATURE_NAMES)

    return full_pred

import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt


def plot_feature_maps(final_selected_maps, feature_names):
    """
    可视化每个特征的空间分布
    Args:
        final_selected_maps: [C, H, W]
        feature_names: 特征名称
    """
    C, H, W = final_selected_maps.shape
    cols = 5
    rows = int(np.ceil(C / cols))

    plt.figure(figsize=(3*cols, 3*rows))
    for c in range(C):
        plt.subplot(rows, cols, c+1)
        plt.imshow(final_selected_maps[c], cmap="viridis")
        plt.title(feature_names[c], fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig('featuremap')
import numpy as np
import matplotlib.pyplot as plt

def analyze_selection_probs(selection_probs, labels, feature_names, class_names):
    """
    selection_probs: [N, C] 每个样本的特征选择概率
    labels: [N] 样本类别
    feature_names: list[str]
    class_names: list[str]
    """
    num_classes = len(class_names)
    num_features = len(feature_names)
    
    # 计算每类的平均选择概率
    class_feature_means = np.zeros((num_classes, num_features))
    for cls in range(num_classes):
        idx = (labels == cls)
        
        class_feature_means[cls] = selection_probs[idx].mean(axis=0)
    
    # 画热力图
    plt.figure(figsize=(10, 6))
    plt.imshow(class_feature_means, cmap="viridis", aspect="auto")
    plt.colorbar(label="Selection Probability")
    plt.xticks(range(num_features), feature_names, rotation=90, fontsize=8)
    plt.yticks(range(num_classes), class_names)
    plt.title("Feature Selection Probability per Class")
    plt.tight_layout()
    plt.savefig('chosemap.jpg')
    
    return class_feature_means


def visualize_all_features(selected_maps, feature_names=None):
    """
    可视化所有特征
    selected_maps: Tensor [B, num_features, H, W]
    feature_names: 可选，特征名列表
    """
    B, num_features, H, W = selected_maps.shape
    cols = 6
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()

    for i in range(num_features):
        fmap = selected_maps[0, i].cpu().detach().numpy()
        axes[i].imshow(fmap, cmap="viridis")
        title = f"F{i+1}"
        if feature_names is not None:
            title = feature_names[i]
        axes[i].set_title(title, fontsize=8)
        axes[i].axis("off")

    for i in range(num_features, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

def create_color_map() -> Tuple[np.ndarray, dict]:
    """创建类别-颜色映射（适配5种作物）"""
    # 定义5种作物的颜色（可根据需求调整，确保区分度高）
    color_map = COLOR
    # 类别名称映射（对应crop_types）
    class_map = {i: CROP_TYPES[i] for i in range(len(CROP_TYPES))}
    return color_map, class_map
import seaborn as sns
def calculate_confusion_matrix(y_true, y_pred, classes=None, normalize=False):
    """
    计算并返回混淆矩阵
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    classes: 类别名称列表
    normalize: 是否对矩阵进行归一化
    """
    #     # 只保留标签为0-4的样本

    print("Unique values in y_true:", np.unique(y_true))
    print("Unique values in y_pred:", np.unique(y_pred))
    print("CROP_TYPES:", CROP_TYPES)
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    print(y_true.shape,y_pred.shape)

    valid_indices = np.where((y_true >= 0) & (y_true <= 4) & (y_pred >= 0) & (y_pred <= 4))
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]
    y_true = y_true_filtered.reshape(-1)
    y_pred = y_pred_filtered.reshape(-1)
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=CROP_TYPES)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
    
    return cm

def calculate_iou(cm):
    """
    计算每个类别的IoU和平均IoU(mIoU)
    """
    # 计算每个类别的IoU
    iou_per_class = np.zeros(len(cm))
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        iou_per_class[i] = iou
    
    # 计算mIoU (排除没有样本的类别)
    valid_classes = np.sum(cm, axis=1) > 0
    miou = np.mean(iou_per_class[valid_classes]) if np.any(valid_classes) else 0
    
    return iou_per_class, miou
def print_classification_report(cm, classes=None):
    """
    打印分类报告，包括精确率、召回率和F1分数
    """
    # 计算各项指标
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # 计算IoU和mIoU
    iou_per_class, miou = calculate_iou(cm)
    
    # 处理除零错误
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    
    # 计算总体准确率
    accuracy = np.trace(cm) / np.sum(cm)
    
    # 打印结果
    print(f"总体准确率: {accuracy:.4f}")
    print(f"平均交并比(mIoU): {miou:.4f}")
    print("\n各类别指标:")
    print(f"{'类别':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'IoU':<10} {'样本数':<10}")
    print("-" * 60)
    
    for i in range(len(cm)):
        class_name = classes[i] if classes else f"Class {i}"
        print(f"{class_name:<10} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {iou_per_class[i]:<10.4f} {np.sum(cm[i]):<10}")
    
    # 打印宏平均和加权平均
    print("\n平均值:")
    print(f"{'宏平均':<10} {np.mean(precision):<10.4f} {np.mean(recall):<10.4f} {np.mean(f1):<10.4f} {np.mean(iou_per_class):<10.4f}")
    print(f"{'加权平均':<10} {np.average(precision, weights=np.sum(cm, axis=1)):<10.4f} "
          f"{np.average(recall, weights=np.sum(cm, axis=1)):<10.4f} "
          f"{np.average(f1, weights=np.sum(cm, axis=1)):<10.4f} "
          f"{np.average(iou_per_class, weights=np.sum(cm, axis=1)):<10.4f}")

def visualize_metrics(cm: np.ndarray, class_names,save_dir: str) -> None:
    """可视化并保存性能指标（混淆矩阵、类别指标表格）"""
    # class_names = metrics["class_names"]
    
    # # 1. 保存文本格式的分类报告
    # report_path = os.path.join(save_dir, "classification_report.txt")
    # with open(report_path, "w") as f:
    #     f.write("=== 全图分类性能报告 ===\n\n")
    #     f.write(f"全局准确率: {metrics['total_accuracy']:.4f}\n\n")
    #     f.write("类别详细指标:\n")
    #     for cls in class_names:
    #         f.write(f"- {cls}:\n")
    #         f.write(f"  样本数量: {metrics['class_counts'][cls]}\n")
    #         f.write(f"  准确率: {metrics['class_accuracy'][cls]:.4f}\n")
    #         f.write(f"  精确率: {metrics['precision'][cls]:.4f}\n")
    #         f.write(f"  召回率: {metrics['recall'][cls]:.4f}\n")
    #         f.write(f"  F1分数: {metrics['f1'][cls]:.4f}\n\n")
        
    #     # 附加sklearn的详细报告
    #     f.write("\n=== 详细分类报告 ===\n")
    #     y_pred = np.concatenate([[cls]*cnt for cls, cnt in enumerate(metrics['class_counts'].values())])
    #     y_true = np.concatenate([[i]*metrics['class_counts'][cls] for i, cls in enumerate(class_names)])
    #     f.write(classification_report(y_true, y_pred, target_names=class_names))
    # print(f"📝 分类报告保存路径: {report_path}")
    
    # 2. 绘制混淆矩阵热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth")
    # plt.title("混淆矩阵")
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"📊 混淆矩阵保存路径: {cm_path}")


def save_classification_result(full_pred: np.ndarray, full_gt: np.ndarray, save_dir: str = "classification_results") -> None:
    """
    保存全图分类结果：
    1. 可视化彩色图（PNG）
    2. 原始类别标签图（NPY格式，便于后续分析）
    3. 类别说明文档（TXT）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 保存原始类别标签数组（NPY格式，可后续加载分析）
    npy_path = os.path.join(save_dir, "full_classification_labels.npy")
    np.save(npy_path, full_pred)
    print(f"💾 原始标签图保存路径: {npy_path}")
    
    # 2. 生成并保存可视化彩色图
    color_map, class_map = create_color_map()
    # 将类别标签映射为彩色图像
    H, W = full_pred.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            class_idx = full_pred[i, j]
            # 确保类别索引在颜色映射范围内
            if 0 <= class_idx < len(color_map):
                color_img[i, j] = color_map[class_idx]
            else:
                color_img[i, j] = color_map[0]  # 未知类别用背景色
    
    # 保存彩色图（PNG格式，压缩率高）
    png_path = os.path.join(save_dir, "full_classification_color.png")
    # 用cv2保存（避免matplotlib的坐标轴问题）
    cv2.imwrite(png_path, cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))  # 转换为BGR适配cv2
    print(f"🖼️ 彩色分类图保存路径: {png_path}")
    
    # 3. 计算并保存类别性能指标
    if full_gt is not None:
        cm = calculate_confusion_matrix(full_pred, full_gt, CROP_TYPES)
        print_classification_report(cm, classes=CROP_TYPES)
        types = ['Barley','Corn','Potatoes','Sugarbeat','Wheat']
        visualize_metrics(cm,class_names=types,save_dir='./')


def plot_selection_importance(selection_probs, feature_names, save_path="feature_importance.pdf"):
    plt.figure(figsize=(10, 4))
    idx = np.arange(len(feature_names))
    plt.bar(idx, selection_probs, color="skyblue")
    plt.xticks(idx, feature_names, rotation=90)
    plt.ylabel("Selection Probability")
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
def run_full_image_classification() -> None:
    """全图分类主流程：加载数据→加载模型→推理→保存结果"""
    print("="*60)
    print("📌 开始全图分类流程")
    print("="*60)
    
    # 1. 加载完整原始数据（极化特征+RGB图像）
    print("\n1. 加载完整数据...")
    polar_full = load_polarization_features()  # [H, W, 23]（从data_utils导入）
    rgb_full = load_rgb_image()                # [H, W, 3]（从data_utils导入）
    full_gt = load_full_ground_truth()         # 新增：加载全图真实标签 [H, W]

    print(f"   极化图像形状: {polar_full.shape}")
    print(f"   RGB图像形状: {rgb_full.shape}")
    
    # 2. 加载预训练模型
    print("\n2. 加载预训练模型...")
    model = load_pretrained_model(MODEL_SAVE_PATH)
    pre = True
    # 3. 全图推理
    if pre == True:
        print("\n3. 全图推理...")
        full_pred = infer_full_image(model, polar_full, rgb_full)
    else:
        full_pred = np.load('deguo/Dual/full_classification_labels.npy')
        print("\n3. 读取已有分类结果...")
    # 4. 保存分类结果
    print("\n4. 保存分类结果...")

    save_classification_result(full_pred, full_gt)
    
    print("\n" + "="*60)
    print("🎉 全图分类流程完成！结果保存在 'classification_results' 目录")
    print("="*60)

if __name__ == "__main__":
    run_full_image_classification()