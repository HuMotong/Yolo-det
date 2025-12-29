import torch
import torch.nn as nn
import math
from .blocks import FeatureExtractor
from .attention import BidirectionalCrossAttention
from .weight_net import PyramidFeatureFusion, WeightGenerator, OutputHead


class PolarizationSR(nn.Module):
    """偏振超分辨率网络"""

    def __init__(self, cfg):
        super(PolarizationSR, self).__init__()

        # 特征提取网络
        self.linear_extractor = FeatureExtractor(1, cfg.model.init_features)
        self.circular_extractor = FeatureExtractor(1, cfg.model.init_features)

        # 双向交叉注意力
        self.cross_attention = BidirectionalCrossAttention(
            cfg.model.init_features,
            cfg.model.num_heads,
            patch_size=16
        )

        # 特征融合
        self.pyramid_fusion = PyramidFeatureFusion(cfg.model.init_features)

        # 权重生成器
        self.weight_generator = WeightGenerator(cfg.model.init_features * 3)

        # 上采样模块
        self.upsampler = nn.Sequential(
            nn.Conv2d(cfg.model.init_features, cfg.model.init_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),  # 上采样2倍
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cfg.model.init_features, cfg.model.init_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),  # 再次上采样2倍
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 输出头
        self.output_head = OutputHead(cfg.model.init_features)

    def forward(self, linear_pol, circular_pol):
        # 特征提取
        linear_feat = self.linear_extractor(linear_pol)
        circular_feat = self.circular_extractor(circular_pol)

        # 交叉注意力
        linear_attn, circular_attn = self.cross_attention(linear_feat, circular_feat)

        # 特征融合
        mixed_feat = self.pyramid_fusion(linear_attn + circular_attn)

        # 生成每个输出头的权重
        weights = self.weight_generator(linear_feat, circular_feat, mixed_feat)

        # 对每个输出头使用独立的权重进行特征融合
        output_features = {}
        for head_name, head_weights in weights.items():
            weighted_feat = (
                    head_weights[:, [0], :, :] * linear_feat +
                    head_weights[:, [1], :, :] * circular_feat +
                    head_weights[:, [2], :, :] * mixed_feat
            )
            output_features[head_name] = weighted_feat

        # 上采样
        up_features = {k: self.upsampler(v) for k, v in output_features.items()}

        # 生成输出
        outputs = {}
        outputs['s0'] = self.output_head.s0_head(up_features['s0'])
        outputs['dop'] = self.output_head.dop_head(up_features['dop'])
        outputs['aop'] = self.output_head.aop_head(up_features['aop'])
        outputs['ellip'] = self.output_head.ellip_head(up_features['ellip'])

        return outputs
