import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import os

# CBAM
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                      padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_feat = self.avg_pool(x).view(b, c)
        max_feat = self.max_pool(x).view(b, c)
        ca = self.mlp(avg_feat) + self.mlp(max_feat)
        ca = self.sigmoid_channel(ca).view(b, c, 1, 1)
        x = x * ca

        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp, _ = x.max(dim=1, keepdim=True)
        sa = torch.cat([avg_sp, max_sp], dim=1)
        sa = self.conv_spatial(sa)
        sa = self.sigmoid_spatial(sa)
        return x * sa

# 双向交叉注意力融合
class BiCrossAttentionFusion(nn.Module):
    def __init__(self, dim_glcm=64, dim_img=2048, embed_dim=512, num_heads=8):
        super().__init__()
        self.glcm2img_query_proj = nn.Linear(dim_glcm, embed_dim)
        self.glcm2img_key_proj = nn.Linear(dim_img, embed_dim)
        self.glcm2img_value_proj = nn.Linear(dim_img, embed_dim)

        self.img2glcm_query_proj = nn.Linear(dim_img, embed_dim)
        self.img2glcm_key_proj = nn.Linear(dim_glcm, embed_dim)
        self.img2glcm_value_proj = nn.Linear(dim_glcm, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 512)  # 最终输出512维，兼容分类器
        )

    def forward(self, glcm_feat, img_feat):

        glcm_q = self.glcm2img_query_proj(glcm_feat).unsqueeze(1)
        img_k = self.glcm2img_key_proj(img_feat).unsqueeze(1)
        img_v = self.glcm2img_value_proj(img_feat).unsqueeze(1)
        glcm2img_attn, _ = self.attn(glcm_q, img_k, img_v)
        glcm2img_out = glcm2img_attn.squeeze(1)  # [B, 512]

        img_q = self.img2glcm_query_proj(img_feat).unsqueeze(1)
        glcm_k = self.img2glcm_key_proj(glcm_feat).unsqueeze(1)
        glcm_v = self.img2glcm_value_proj(glcm_feat).unsqueeze(1)
        img2glcm_attn, _ = self.attn(img_q, glcm_k, glcm_v)
        img2glcm_out = img2glcm_attn.squeeze(1)  # [B, 512]

        bi_fused = torch.cat([glcm2img_out, img2glcm_out], dim=1)  # [B, 1024]
        final_fused = self.fusion_proj(bi_fused)  # [B, 512]

        return final_fused

# 自适应温度模块
class EnhancedDynamicConfusionClassifier(nn.Module):
    def __init__(self, in_features, num_classes, init_temp=1.0,
                 update_interval=5, misclassify_thresh=0.1, min_samples=50):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
        self.in_features = in_features
        self.base_temp = nn.Parameter(torch.tensor(init_temp))
        self.temp_min = 0.5
        self.temp_max = 2.0

        self.confusion_pairs = OrderedDict()
        self.update_interval = update_interval
        self.misclassify_thresh = misclassify_thresh
        self.min_samples = min_samples

        self.uncertainty_estimator = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_features, in_features),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(in_features, in_features // 2),
            nn.BatchNorm1d(in_features // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 2, 1),
            nn.Sigmoid()
        )

        self.pair_temps = nn.ParameterDict()
        self.history_cm = []
        self.max_history = 3

        self.temp_reg_weight = 1e-4

    def forward(self, x):
        logits = self.fc(x)
        batch_size = logits.shape[0]

        uncertainty = self.uncertainty_estimator(x)  # [B,1]
        # 根据样本不确定性动态调整温度
        global_temp = self.temp_min + (self.temp_max - self.temp_min) * (1 - uncertainty)
        global_temp = torch.clamp(global_temp + self.base_temp, self.temp_min, self.temp_max)
        final_temp = global_temp.clone()
        scaled_logits = logits / final_temp
        if self.training:
            temp_reg = self.temp_reg_weight * (self.base_temp ** 2)
            self.temp_reg = temp_reg
        else:
            self.temp_reg = 0.0

        return scaled_logits

    def update_confusion_pairs(self, y_true, y_pred):
        return list(self.confusion_pairs.keys())

    def _update_pair_temps(self, new_pairs):
        pass

    def get_temperature_stats(self):
        stats = {
            "base_temp": self.base_temp.item(),
            "pair_temps": "消融实验：混淆对加权已禁用",
            "temp_range": (self.temp_min, self.temp_max),
            "confusion_pairs_status": "disabled"
        }
        return stats

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in list(state_dict.keys()):
            if key.startswith("pair_temps."):
                del state_dict[key]
        return state_dict
    def load_state_dict(self, state_dict, strict=True):
        for key in list(state_dict.keys()):
            if key.startswith("pair_temps."):
                del state_dict[key]
                print(f"⚠️  消融实验：混淆对参数：{key}")

        super().load_state_dict(state_dict, strict=strict)

    def get_regularization_loss(self):
        return self.temp_reg if hasattr(self, 'temp_reg') else 0.0
# GLCM-ViT分支
class LocalWindowSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1,
                 window_size=1, alpha=1.0):
        super().__init__()
        assert window_size >= 1, "window_size 必须 >= 1"
        self.window_size = int(window_size)
        self.alpha = float(alpha)

        self.norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens):
        identity = tokens
        t = self.norm(tokens)

        n = t.size(1)
        idx = torch.arange(n, device=t.device)
        dist = (idx[None, :] - idx[:, None]).abs()
        attn_mask = dist > self.window_size

        out, _ = self.mha(
            t, t, t,
            attn_mask=attn_mask,
            need_weights=False
        )
        out = self.drop(out)
        return identity + self.alpha * out

class PureGLCMBranch(nn.Module):
    def __init__(
        self,
        input_dim=256,
        num_groups=16,
        group_dim=8,
        hidden_dim=384,
        output_dim=64,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        gate_dropout=0.2,
        gate_floor=0.10,
        local_window_size=1,
        local_attn_heads=4,
        local_attn_alpha=1.0
    ):
        super().__init__()

        assert num_groups * group_dim == input_dim, (
            f"分组参数不匹配：{num_groups}×{group_dim}≠{input_dim}"
        )
        assert embed_dim % num_heads == 0, (
            f"embed_dim={embed_dim} 必须能被 num_heads={num_heads} 整除"
        )
        assert embed_dim % local_attn_heads == 0, (
            f"embed_dim={embed_dim} 必须能被 local_attn_heads={local_attn_heads} 整除"
        )

        self.input_dim = input_dim
        self.num_groups = num_groups
        self.group_dim = group_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.gate_floor = float(gate_floor)
        self.num_distances = 4
        self.num_angles = 8
        assert self.num_distances * self.num_angles == self.num_groups, (
            "距离数 × 方向数 必须等于 num_groups"
        )

        self.in_norm = nn.LayerNorm(input_dim)

        # 1. 距离-方向组合自适应权重
        self.group_weight = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_groups),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Sigmoid()
        )

        self.token_embed = nn.Linear(group_dim, embed_dim)

        # 3. 局部窗口自注意力
        self.local_window_attn = LocalWindowSelfAttention(
            embed_dim=embed_dim,
            num_heads=local_attn_heads,
            dropout=dropout,
            window_size=local_window_size,
            alpha=local_attn_alpha
        )

        self.mid_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 32),
            nn.GELU()
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_groups, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, glcm_feat):
        if glcm_feat.dim() != 2 or glcm_feat.size(1) != self.input_dim:
            raise ValueError(
                f"GLCM输入维度错误：期望 [B,{self.input_dim}]，"
                f"实际 {tuple(glcm_feat.shape)}"
            )

        x = self.in_norm(glcm_feat)
        b = x.size(0)

        grouped = x.view(
            b,
            self.num_distances,
            self.num_angles,
            self.group_dim
        )  # [B,4,8,8]

        grouped = grouped.reshape(
            b,
            self.num_groups,
            self.group_dim
        )  # [B,32,8]

        group_weights = self.group_weight(x).unsqueeze(-1)  # [B,32,1]

        if self.gate_floor > 0:
            group_weights = (
                self.gate_floor
                + (1.0 - self.gate_floor) * group_weights
            )

        weighted_groups = grouped * group_weights  # [B,32,8]

        tokens = self.token_embed(weighted_groups)  # [B,32,128]

        # 5. 局部窗口自注意力
        tokens = self.local_window_attn(tokens)  # [B,32,128]

        glcm_mid = self.mid_proj(tokens)  # [B,32,32]
        glcm_mid = glcm_mid.transpose(1, 2)  # [B,32,32]
        glcm_mid = glcm_mid.unsqueeze(-1)  # [B,32,32,1]

        cls = self.cls_token.expand(b, -1, -1)  # [B,1,128]
        vit_tokens = torch.cat([cls, tokens], dim=1)  # [B,33,128]

        vit_tokens = self.pos_drop(
            vit_tokens + self.pos_embed
        )

        vit_tokens = self.encoder(vit_tokens)
        cls_feat = self.norm(vit_tokens[:, 0])  # [B,128]

        final_feat = self.output_proj(cls_feat)  # [B,64]

        return final_feat, glcm_mid

# Layer2交叉融合
class Layer2CrossFusion(nn.Module):
    def __init__(self, img_channels=512, glcm_channels=32, reduction=8):
        super().__init__()
        self.glcm_adapter = nn.Sequential(
            nn.Conv2d(glcm_channels, img_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(img_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # [B,512,32,1] → [B,512,1,1]
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(img_channels, img_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(img_channels // reduction, img_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, img_feat, glcm_feat):
        glcm_aligned = self.glcm_adapter(glcm_feat)  # [B,512,1,1]

        attn_weight = self.attention(img_feat)  # [B,512,1,1]
        fused_weight = attn_weight * glcm_aligned

        out = img_feat * fused_weight + img_feat
        return self.dropout(out)

# 主模型
class ResNet50WithGLCM(nn.Module):
    def __init__(self, num_classes=4, pretrained_backbone=False, backbone_pretrain_path=None):
        super().__init__()

        base = resnet50(weights=None)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.drop1 = nn.Dropout2d(p=0.2)
        self.layer2 = base.layer2
        self.drop2 = nn.Dropout2d(p=0.2)
        self.layer3 = base.layer3
        self.drop3 = nn.Dropout2d(p=0.4)
        self.layer4 = base.layer4
        self.drop4 = nn.Dropout2d(p=0.4)
        self.avgpool = base.avgpool
        self.cbam = CBAM(channels=2048)

        self.layer2_fusion = Layer2CrossFusion(img_channels=512, glcm_channels=32)

        # 3. GLCM-ViT分支
        self.glcm_branch = PureGLCMBranch(
            input_dim=256,
            num_groups=32,
            group_dim=8,
            hidden_dim=384,
            output_dim=64
        )

        # 4. 双向交叉注意力融合
        self.cross_attention = BiCrossAttentionFusion(
            dim_glcm=64,
            dim_img=2048,
            embed_dim=512,
            num_heads=8
        )

        # 5. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.6),
            EnhancedDynamicConfusionClassifier(
                in_features=256,
                num_classes=num_classes,
                init_temp=1.0,
                update_interval=5,
                misclassify_thresh=0.1,
                min_samples=50
            )
        )

        if backbone_pretrain_path is not None and os.path.exists(backbone_pretrain_path):
            self._load_backbone_pretrain(backbone_pretrain_path)

        self.update_interval = self.classifier[-1].update_interval
        self.num_classes = num_classes

    def forward(self, x, glcm):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.drop1(x)

        x_layer2 = self.layer2(x)
        x_layer2 = self.drop2(x_layer2)  # [B,512,H,W]

        glcm_final, glcm_mid = self.glcm_branch(glcm)  # glcm_mid: [B,32,32,1]

        x_layer2_fused = self.layer2_fusion(x_layer2, glcm_mid)  # [B,512,H,W]

        x = self.layer3(x_layer2_fused)
        x = self.drop3(x)
        x = self.layer4(x)
        x = self.drop4(x)
        x = self.avgpool(x)  # [B,2048,1,1]

        x = self.cbam(x)
        x = x.view(x.size(0), -1)  # [B,2048]

        fused = self.cross_attention(glcm_feat=glcm_final, img_feat=x)  # [B,512]
        return self.classifier(fused)

    def _load_backbone_pretrain(self, pretrain_path):
        print(f"📌 加载自定义backbone预训练权重：{pretrain_path}")
        pretrain_state_dict = torch.load(pretrain_path, map_location='cpu')

        backbone_keys = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        backbone_state_dict = {}
        for key, value in pretrain_state_dict.items():
            if any(key.startswith(k) for k in backbone_keys):
                backbone_state_dict[key] = value

        load_result = self.load_state_dict(backbone_state_dict, strict=False)
        print(f"✅ Backbone权重加载完成：")
        print(f"  - 成功加载参数数：{len(load_result.matched_keys)}")
        print(f"  - 忽略参数数（非backbone）：{len(load_result.unexpected_keys)}")
        print(f"  - 缺失参数数（若有）：{len(load_result.missing_keys)}")

    def freeze_layers(self, freeze_backbone_until=-1):
        backbone_components = [
            ("conv1", self.conv1),
            ("bn1", self.bn1),
            ("relu", self.relu),
            ("maxpool", self.maxpool),
            ("layer1", self.layer1),
            ("drop1", self.drop1),
            ("layer2", self.layer2),
            ("drop2", self.drop2),
            ("layer3", self.layer3),
            ("drop3", self.drop3),
            ("layer4", self.layer4),
            ("drop4", self.drop4),
            ("avgpool", self.avgpool),
        ]

        if freeze_backbone_until >= 0:
            freeze_n = min(freeze_backbone_until, len(backbone_components))
            print(f"📌 冻结backbone前 {freeze_n} 个组件")

            for idx, (name, module) in enumerate(backbone_components):
                requires_grad = idx >= freeze_n
                for param in module.parameters():
                    param.requires_grad = requires_grad

                if any(True for _ in module.parameters()):
                    status = "解冻" if requires_grad else "冻结"
                    print(f"  - {status} {name}")
        else:
            print("📌 不冻结backbone任何层（全量训练）")
            for _, module in backbone_components:
                for param in module.parameters():
                    param.requires_grad = True

        print("📌 layer2融合模块、GLCM-ViT分支、双向交叉注意力、分类器：开启梯度更新")
        for module in [
            self.layer2_fusion,
            self.glcm_branch,
            self.cross_attention,
            self.cbam,
            self.classifier,
        ]:
            for param in module.parameters():
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

        print("✅ 冻结配置完成：")
        print(f"  - 总参数量：{total_params:,}")
        print(f"  - 可训练参数量：{trainable_params:,}")
        print(f"  - 可训练比例：{trainable_params / total_params * 100:.2f}%")

    def _get_component_idx_by_param_idx(self, param_idx):
        component_param_counts = [
            2,
            2,
            0,
            0,
            6,
            0,
            12,
            0,
            24,
            0,
            18,
            0,
            0
        ]

        cumulative = 0
        for component_idx, count in enumerate(component_param_counts):
            cumulative += count
            if param_idx < cumulative:
                return component_idx
        return len(component_param_counts) - 1

    def get_trainable_params_groups(self, lr_backbone=1e-5, lr_others=1e-4):
        backbone_trainable = []
        others_trainable = []

        backbone_param_names = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        for name, param in self.named_parameters():
            if param.requires_grad:
                is_backbone = any(
                    name == pn or name.startswith(pn + ".")
                    for pn in backbone_param_names
                )
                if is_backbone:
                    backbone_trainable.append(param)
                else:
                    others_trainable.append(param)

        params_groups = [
            {"params": backbone_trainable, "lr": lr_backbone, "weight_decay": 1e-5},
            {"params": others_trainable, "lr": lr_others, "weight_decay": 1e-5}
        ]

        print(f"📊 学习率分组配置：")
        print(f"  - Backbone可训练参数数：{sum(p.numel() for p in backbone_trainable):,}，学习率：{lr_backbone}")
        print(f"  - 其他可训练参数数：{sum(p.numel() for p in others_trainable):,}，学习率：{lr_others}")
        return params_groups

    def update_confusion_pairs(self, y_true, y_pred):
        return self.classifier[-1].update_confusion_pairs(y_true, y_pred)

    def get_temperature_stats(self):
        return self.classifier[-1].get_temperature_stats()

    def get_regularization_loss(self):
        return self.classifier[-1].get_regularization_loss()

if __name__ == "__main__":
    print("=== 源数据集（6类）模型初始化 ===")
    model_source = ResNet50WithGLCM(num_classes=6, pretrained_backbone=False)
    print(f"源模型输出维度：{model_source(torch.randn(2, 3, 224, 224), torch.randn(2, 256)).shape}")
    print(f"源模型类别数：{model_source.num_classes}\n")

    print("=== 目标数据集（4类）模型初始化 ===")
    model_target = ResNet50WithGLCM(
        num_classes=4,
        pretrained_backbone=False,
        backbone_pretrain_path="path/to/source_dataset_trained_weights.pth"
    )
    # 冻结backbone，仅训练分类器和GLCM分支（适配4类）
    model_target.freeze_layers(freeze_backbone_until=13)
    # 获取分组学习率参数
    params_groups = model_target.get_trainable_params_groups(lr_backbone=1e-6, lr_others=1e-3)
    print(f"目标模型输出维度：{model_target(torch.randn(2, 3, 224, 224), torch.randn(2, 256)).shape}")
    print(f"目标模型类别数：{model_target.num_classes}")
    print(f"温度统计：{model_target.get_temperature_stats()}")