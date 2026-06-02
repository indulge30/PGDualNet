# model_components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from config import *  # 导入配置


# -------------------------- 基础组件 --------------------------
class BasicBlock(nn.Module):
    """残差块（用于空间分支）"""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

#全局特征选择
class FixedSelectionGumbelSelector(nn.Module):
    """固定选择概率的特征选择器"""
    def __init__(
        self,
        num_features: int,
        feature_groups: Dict[str, List[int]],
        group_priors: Dict[str, float],
        feature_names: List[str],
        scene_context: str = "agriculture",
        correlation_strength: float = 0.8,
        reliability_weight: float = 1.0,
        entropy_weight: float = 0.01  # 添加熵正则化权重
    ):
        super().__init__()
        self.num_features = num_features
        self.feature_groups = feature_groups
        self.group_priors = group_priors
        self.feature_names = feature_names
        self.scene_context = scene_context
        self.correlation_strength = correlation_strength
        self.reliability_weight = reliability_weight
        self.entropy_weight = entropy_weight

        # 初始化物理先验和相关矩阵
        self.register_buffer('physics_priors', self._initialize_physics_priors())
        self.register_buffer('correlation_matrix', self._initialize_correlation_matrix())
        self.register_buffer('reliability_weights', self._initialize_reliability_weights())

        # 可学习的选择logits - 每个特征的初始选择概率
        self.selection_logits = nn.Parameter(torch.zeros(num_features))

    def _initialize_physics_priors(self) -> torch.Tensor:
        """初始化物理先验（基于分组）"""
        priors = torch.ones(self.num_features) * 0.5  # 默认先验
        for group_name, indices in self.feature_groups.items():
            group_prior = self.group_priors.get(group_name, 0.5)
            for idx in indices:
                if idx < self.num_features:
                    priors[idx] = group_prior
        return priors

    def _initialize_correlation_matrix(self) -> torch.Tensor:
        """初始化特征相关性矩阵（基于物理意义）"""
        correlation_matrix = torch.eye(self.num_features)
        for group_name, indices in self.feature_groups.items():
            for i in indices:
                for j in indices:
                    if i != j and i < self.num_features and j < self.num_features:
                        correlation_matrix[i, j] = 0.7
        if self.scene_context == "agriculture":
            surface_indices = self.feature_groups.get('surface', [])
            volume_indices = self.feature_groups.get('volume', [])
            for i in surface_indices:
                for j in volume_indices:
                    if i < self.num_features and j < self.num_features:
                        correlation_matrix[i, j] = -0.6
                        correlation_matrix[j, i] = -0.6
        return correlation_matrix
    
    def _initialize_reliability_weights(self) -> torch.Tensor:
        """初始化可靠性权重 - 基于特征物理属性"""
        weights = torch.ones(self.num_features)

        surface_indices = self.feature_groups.get('surface', [])
        for idx in surface_indices:
            if idx < self.num_features:
                weights[idx] = 1.2  # surface 更可靠

        volume_indices = self.feature_groups.get('volume', [])
        for idx in volume_indices:
            if idx < self.num_features:
                weights[idx] = 1.0  # volume 中等可靠性

        other_indices = self.feature_groups.get('other', [])
        for idx in other_indices:
            if idx < self.num_features:
                weights[idx] = 0.8  # other 较低可靠性

        return weights

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """固定选择概率的特征选择"""
        # 计算选择概率
        selection_probs = torch.sigmoid(self.selection_logits)
        
        # 生成掩码（训练用连续松弛，推理用硬选择）
        if self.training:
            selection_mask = selection_probs
        else:
            # 推理时选择概率最大的特征
            _, top_indices = torch.topk(selection_probs, k=int(self.num_features * 0.6))  # 选前60%
            selection_mask = torch.zeros_like(selection_probs)
            selection_mask[top_indices] = 1.0
        
        # 应用选择掩码
        batch_size, num_channels, height, width = features.shape
        selection_mask_expanded = selection_mask.view(1, num_channels, 1, 1)
        selection_mask_expanded = selection_mask_expanded.expand(batch_size, num_channels, height, width)
        selected_features = features * selection_mask_expanded 
        # 计算多样性约束损失
        diversity_loss = self._compute_diversity_constraint(selection_probs)

        info = {
            'selection_probs': selection_probs.detach(),
            'selected_count': (selection_mask > 0.5).sum().item(),
            'diversity_loss': diversity_loss,
        }
        return selected_features, selection_mask, info 

    def _compute_diversity_constraint(self, selection_probs: torch.Tensor) -> torch.Tensor:
        """计算多家族多样性约束损失"""
        loss = 0.0
        loss += 0.1 * torch.sum(selection_probs)
        for group_name, feature_indices in self.feature_groups.items():
            group_probs = selection_probs[feature_indices]
            if len(group_probs) > 0:
                threshold = min(2, len(group_probs))  # 至少选择2个或全部
                group_loss = F.relu(threshold - torch.sum(group_probs))
                loss += 0.2 * group_loss
        active_groups = 0
        for group_name, feature_indices in self.feature_groups.items():
            if torch.any(selection_probs[feature_indices] > 0.3):  # 松弛的激活条件
                active_groups += 1
        loss -= 0.3 * active_groups
        entropy_loss = -torch.sum(selection_probs * torch.log(selection_probs + 1e-8))
        loss += self.entropy_weight * entropy_loss
        return loss

#局部空间适应
class LocalGroupAdaptiveSelection(nn.Module):
    """
    局部组适应特征选择器
    - 输入: [B, C, H, W]
    - 输出: [B, C, H, W]
    - 支持 feature_groups + group_priors
    """
    def __init__(self, num_features: int,
                 feature_groups: List[List[int]],
                 group_priors: Dict[str, float],
                 feature_names: List[str],
                 reduction: int = 16):
        super().__init__()
        self.num_features = num_features
        self.feature_groups = feature_groups
        self.group_priors = group_priors
        self.feature_names = feature_names

        # ===== 局部通道选择器 (1x1 conv, 带空间适应性) =====
        self.conv1 = nn.Conv2d(num_features, num_features // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features // reduction, num_features, kernel_size=1)

        # ===== 组约束权重 =====
        self.group_bias = self._build_group_bias()     # shape: [C]

        self.sigmoid = nn.Sigmoid()

    def _build_group_bias(self) -> torch.Tensor:
        """根据分组信息构建 group mask (每个通道属于哪个组)"""
        bias = torch.zeros(self.num_features)
        for g_name, channels in self.feature_groups.items():
            prior = self.group_priors.get(g_name, 0.0)  # 没有先验则默认为0
            for ch in channels:
                bias[ch] = prior
        return bias


    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        return:
          selected_features: 加权特征 [B, C, H, W]
          selection_probs: 局部选择概率 [B, C, H, W]
          info: 调试信息
        """
        B, C, H, W = x.shape

        # ===== 基础局部选择 =====
        logits = self.conv2(self.relu(self.conv1(x)))  # [B, C, H, W]

        # ===== 加上 group bias =====
        group_bias = self.group_bias.view(1, C, 1, 1).to(x.device)  # [1,C,1,1]
        logits = logits + group_bias

        # ===== 概率化 =====
        selection_probs = self.sigmoid(logits)

        
        # ===== 组约束 (保证同组通道一致性) =====
        # 方法：同组内取均值，广播回各通道
        selection_probs_grouped = selection_probs.clone()
        for g_name, channels in self.feature_groups.items():
            if len(channels) == 0:
                continue
            group_mask = selection_probs[:, channels, :, :]  # [B, |G|, H, W]
            # group_mean = group_mask.mean(dim=1, keepdim=True)  # [B,1,H,W]
            # selection_probs_grouped[:, channels, :, :] = group_mean
            group_mean = group_mask.mean(dim=1, keepdim=True)  # [B,1,H,W]
            selection_probs_grouped[:, channels, :, :] = (
                0.5 * group_mask + 0.5 * group_mean
            )
        # ===== 应用选择 =====
            selected_features = x * selection_probs_grouped

        # ===== 输出信息 =====
        info = {
            "selection_probs": selection_probs_grouped.detach(),
            "mask_mean": selection_probs_grouped.mean().item(),
            "mask_max": selection_probs_grouped.max().item(),
        }

        return selected_features, selection_probs_grouped, info

#极化特征高级处理
class PolarimetricProcessingMLP(nn.Module):
    """
    极化特征处理MLP - 带有残差连接
    处理选择的极化特征并提取高级表示
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_dims: List[int] = [64, 128, 64],
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        # 构建MLP层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            ])
            in_channels = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 残差连接
        self.residual = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dims[-1], kernel_size=1),
            nn.BatchNorm2d(hidden_dims[-1])
        ) if input_channels != hidden_dims[-1] else nn.Identity()
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # 通过MLP
        x = self.mlp(x)
        
        # 残差连接
        if isinstance(self.residual, nn.Identity):
            identity = identity
        else:
            identity = self.residual(identity)
        
        # 确保尺寸匹配
        if x.shape != identity.shape:
            identity = F.interpolate(identity, size=x.shape[2:], mode='nearest')
        
        return self.activation(x + identity)
# -------------------------- 分支网络 --------------------------
#空间分支
class SpatialBranch(nn.Module):
    """空间分支（处理RGB图像）"""
    def __init__(self, input_channels: int = SPATIAL_CHANNELS):
        super().__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差层（保持特征图尺寸）
        self.in_channels = 32
        self.layer1 = self._make_res_layer(32, 2)  # 2个残差块，输出32通道
        self.layer2 = self._make_res_layer(64, 2)  # 2个残差块，输出64通道
        
        self._initialize_weights()  # 权重初始化

    def _make_res_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """构建残差层（含下采样适配通道数）"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Kaiming初始化卷积层，常数初始化BN层"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 [B, 3, H, W] → 输出 [B, 64, H, W]"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

#极化分支
class PolarimetricBranch(nn.Module):
    """简化版极化分支（去除多尺度特征提取）"""
    def __init__(self,
                 input_channels: int,
                 output_channels: int = 64,
                 use_multi_scale: bool = False):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_multi_scale = use_multi_scale
        
        # 特征选择器
        # self.feature_selector = FixedSelectionGumbelSelector(
        #     num_features=input_channels,
        #     feature_groups=FEATURE_GROUPS,
        #     group_priors=GROUP_PRIORS,
        #     feature_names=FEATURE_NAMES
        # )
        self.feature_selector = LocalGroupAdaptiveSelection(
            num_features=input_channels,
            feature_groups=FEATURE_GROUPS,
            group_priors=GROUP_PRIORS,
            feature_names=FEATURE_NAMES
        )
        # 特征提取：简化为单尺度卷积处理
        self.feature_processor = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # 高级特征提炼
        self.refinement = PolarimetricProcessingMLP(
            input_channels=output_channels,
            hidden_dims=[output_channels * 2, output_channels],
            dropout_rate=0.1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, _, height, width = x.shape

        # 特征选择
        selected_features, selection_mask, selection_info = self.feature_selector(x)

        # 直接进行单尺度处理
        processed_features = self.feature_processor(selected_features)

        # 高级特征提炼
        refined_features = self.refinement(processed_features)

        output_info = {
            **selection_info,
            'selected_mask': selection_mask,
            'input_variance': torch.var(x).item(),
            'output_variance': torch.var(refined_features).item()
        }

        return refined_features, output_info, selected_features

# -------------------------- 融合与分类 --------------------------
class ClassificationHead(nn.Module):
    """分类头（从融合特征输出类别概率）"""
    def __init__(self, in_channels: int = 64 ,  num_classes: int = NUM_CLASSES):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化→[B, 64, 1, 1]
        self.fc = nn.Linear(in_channels, num_classes)  # 全连接→[B, 5]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # [B, 64]
        x = self.fc(x)
        return x

class ConcatenateFusion(nn.Module):
    """Simple concatenation-based feature fusion."""
    
    def __init__(self, spatial_dim: int, polarimetric_dim: int):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.polarimetric_dim = polarimetric_dim
    
    def forward(
        self, 
        spatial_features: torch.Tensor, 
        polarimetric_features: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate spatial and polarimetric features."""
        return torch.cat([spatial_features, polarimetric_features], dim=1)

class AttentionFusion(nn.Module):
    """空间注意力融合模块 - 适用于特征图输入"""
    
    def __init__(self, channels: int = 64, reduction: int = 16):
        super().__init__()
        self.channels = channels
        
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力机制（可选）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 输出卷积
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, spatial_feat: torch.Tensor, polar_feat: torch.Tensor) -> torch.Tensor:
        """应用空间注意力融合"""
        # 拼接特征
        combined = torch.cat([spatial_feat, polar_feat], dim=1)  # [B, 2*C, H, W]
        
        # 通道注意力
        channel_weights = self.channel_attention(combined)  # [B, 2*C, 1, 1]
        channel_weighted = combined * channel_weights
        
        # 空间注意力（可选）
        avg_pool = torch.mean(channel_weighted, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(channel_weighted, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_weights = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))  # [B, 1, H, W]
        
        # 应用空间注意力
        spatially_weighted = channel_weighted * spatial_weights
        
        # 输出投影
        output = self.output_conv(spatially_weighted)
        
        return output

class CrossAttentionFusion(nn.Module):
    """基于交叉注意力的融合模块（修复版）"""
    
    def __init__(self, channels: int = 64, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert self.head_dim * num_heads == channels, "channels必须能被num_heads整除"
        
        # 查询、键、值的线性变换
        self.q_linear = nn.Linear(channels, channels)
        self.k_linear = nn.Linear(channels, channels)
        self.v_linear = nn.Linear(channels, channels)
        
        # 输出投影
        self.out_proj = nn.Linear(channels, channels)
        
        # 丢弃层
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        
        # 位置编码参数
        self.pos_enc = None
        self.cached_size = None
        
        # 层归一化
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )

    def _create_2d_sincos_pos_embed(self, height, width, dim, temperature=10000.0):
        """创建2D正弦余弦位置编码"""
        if self.pos_enc is not None and self.cached_size == (height, width):
            return self.pos_enc
            
        # 生成网格
        y_embed = torch.arange(height, dtype=torch.float32).unsqueeze(1)
        x_embed = torch.arange(width, dtype=torch.float32).unsqueeze(0)
        
        # 计算位置编码
        dim_t = torch.arange(dim // 2, dtype=torch.float32)
        dim_t = temperature ** (2 * dim_t / dim)
        
        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        
        # 应用正弦和余弦函数
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)
        
        # 组合x和y的位置编码
        pos_embed = torch.cat([pos_y.unsqueeze(-2).expand(-1, width, -1), 
                              pos_x.unsqueeze(-3).expand(height, -1, -1)], dim=-1)
        
        # 调整形状
        pos_embed = pos_embed.flatten(0, 1).unsqueeze(0)  # [1, H*W, C]
        
        # 缓存位置编码
        self.pos_enc = pos_embed
        self.cached_size = (height, width)
        
        return pos_embed

    def forward(self, spatial_feat: torch.Tensor, polar_feat: torch.Tensor) -> torch.Tensor:
        """
        输入：空间特征 [B, 64, H, W] + 极化特征 [B, 64, H, W]
        输出：融合特征 [B, 64, H, W]
        """
        B, C, H, W = spatial_feat.shape
        
        # 重塑为序列形式 [B, N, C]，其中 N = H * W
        spatial_seq = spatial_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        polar_seq = polar_feat.flatten(2).transpose(1, 2)      # [B, H*W, C]
        
        # 创建位置编码
        pos_encoding = self._create_2d_sincos_pos_embed(H, W, C).to(spatial_seq.device)
        
        # 添加位置编码
        spatial_seq = spatial_seq + pos_encoding
        polar_seq = polar_seq + pos_encoding
        
        # 保存残差连接
        residual = spatial_seq
        
        # 空间特征作为查询，极化特征作为键和值
        Q = self.q_linear(spatial_seq)  # [B, N, C]
        K = self.k_linear(polar_seq)    # [B, N, C]
        V = self.v_linear(polar_seq)    # [B, N, C]
        
        # 多头注意力计算
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)  # [B, H, N, D]
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, C)  # [B, N, C]
        
        # 输出投影 + 残差连接 + 层归一化
        attn_output = self.out_proj(attn_output)
        attn_output = self.norm1(attn_output + residual)
        
        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(attn_output)
        output = self.norm2(ffn_output + attn_output)
        
        # 重塑回特征图形式
        output = output.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        return output

# -------------------------- 完整模型 --------------------------
class Base(nn.Module):  
    def __init__(self , fusion_type: str = "attention"):
        super().__init__()
        self.spatial_branch = SpatialBranch()          # RGB分支
        self.polar_branch = PolarimetricBranch(input_channels = CHAN)
        self.classifier = ClassificationHead()
        self.selection_info = {}
        self.classifier = ClassificationHead()          # 分类头

    def forward(self, polar_input: torch.Tensor, spatial_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入：极化特征 [B, 23, H, W] + RGB特征 [B, 3, H, W]
        输出：类别logits [B, 5] + 相关性惩罚值
        """
        # 提取分支特征
        spatial_feat = self.spatial_branch(spatial_input)
        polar_feat, selection_info,selected_features = self.polar_branch(polar_input)
        # 保存选择信息
        self.selection_info = selection_info
        
        # 融合与分类
        logits = self.classifier(spatial_feat)
        
        if not self.training:  # eval 阶段
            return logits, selection_info['selection_probs'],selected_features
        else:
            return logits, selection_info['selection_probs']

    def get_selection_info(self) -> Dict:
        """获取最新特征选择信息（如选择概率、温度）"""
        return self.selection_info

    def get_feature_importance(self) -> torch.Tensor:
        """获取特征重要性（基于选择器logits的sigmoid值）"""
        return torch.sigmoid(self.polar_branch.feature_selector.selection_logits.detach())


class PhysicsInformedDualBranchNet(nn.Module):
    """物理信息双分支网络（空间+极化）"""
    def __init__(self , fusion_type: str = "attetion"):
        super().__init__()
        self.spatial_branch = SpatialBranch()          # RGB分支
        # self.polar_branch = PolarimetricBranch(input_channels = CHAN ,feature_groups = FEATURE_GROUPS,
        #                                        group_priors = GROUP_PRIORS,feature_names = FEATURE_NAMES)        # 极化分支
        self.polar_branch = PolarimetricBranch(input_channels = CHAN)
        if fusion_type == "concatenate":
            self.fusion = ConcatenateFusion(spatial_dim=64, polarimetric_dim=64)
        elif fusion_type == "attetion":
            self.fusion = AttentionFusion()
        elif fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion()
        else:
            # 默认使用原始通道注意力
            self.fusion = ConcatenateFusion(spatial_dim=64, polarimetric_dim=64)
        
        self.classifier = ClassificationHead()
        self.selection_info = {}
        
        # 添加用于监控的方法（从原版复制）
        self.selection_info = {}
        self.classifier = ClassificationHead()          # 分类头
        self.selection_info = {}  # 存储最新特征选择信息

    def forward(self, polar_input: torch.Tensor, spatial_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入：极化特征 [B, 23, H, W] + RGB特征 [B, 3, H, W]
        输出：类别logits [B, 5] + 相关性惩罚值
        """
        # 提取分支特征
        spatial_feat = self.spatial_branch(spatial_input)
        polar_feat, selection_info,selected_features = self.polar_branch(polar_input)
        # 保存选择信息
        self.selection_info = selection_info
        # 融合与分类
        fused_feat = self.fusion(spatial_feat, polar_feat)

        logits = self.classifier(fused_feat)
        
        if not self.training:  # eval 阶段
            return logits, selection_info['selection_probs'],selected_features
        else:
            return logits, selection_info['selection_probs']
