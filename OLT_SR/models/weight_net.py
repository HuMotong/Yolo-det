import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PyramidFeatureFusion(nn.Module):
    """特征金字塔融合模块"""

    def __init__(self, in_channels):
        super(PyramidFeatureFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels * 3, in_channels, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # 多尺度池化
        pool1 = F.adaptive_avg_pool2d(x, (x.size(2) // 2, x.size(3) // 2))
        pool2 = F.adaptive_avg_pool2d(x, (x.size(2) // 4, x.size(3) // 4))

        # 上采样对齐
        up1 = F.interpolate(pool1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        up2 = F.interpolate(pool2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # 特征拼接和融合
        out = torch.cat([x, up1, up2], dim=1)
        out = self.lrelu(self.conv1x1(out))
        return out


class WeightGenerator(nn.Module):
    """动态权重生成网络 - 为每个输出头生成独立权重"""

    def __init__(self, in_channels):
        super(WeightGenerator, self).__init__()

        # 共享特征提取
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 为不同输出类型创建独立的权重生成器
        self.s0_weight = nn.Sequential(
            nn.Conv2d(in_channels // 4, 3, 1, 1, 0),
            nn.Softmax(dim=1)
        )

        self.aop_weight = nn.Sequential(
            nn.Conv2d(in_channels // 4, 3, 1, 1, 0),
            nn.Softmax(dim=1)
        )

        self.dop_weight = nn.Sequential(
            nn.Conv2d(in_channels // 4, 3, 1, 1, 0),
            nn.Softmax(dim=1)
        )

        self.ellip_weight = nn.Sequential(
            nn.Conv2d(in_channels // 4, 3, 1, 1, 0),
            nn.Softmax(dim=1)
        )

    def _create_weight_predictor(self, in_channels):
        """创建单个权重预测器"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels // 2, 3, 1, 1, 0)  # 3个权重通道：线性、圆形、混合
        )

    def forward(self, linear_feat, circular_feat, mixed_feat):
        # 拼接特征
        feat_cat = torch.cat([linear_feat, circular_feat, mixed_feat], dim=1)
        shared_feat = self.shared_conv(feat_cat)

        # 生成各类型的权重
        weights = {
            's0': self.s0_weight(shared_feat),  # 偏向线偏振和混合特征
            'aop': self.aop_weight(shared_feat),  # 偏向线偏振和混合特征
            'dop': self.dop_weight(shared_feat),  # 偏向混合特征
            'ellip': self.ellip_weight(shared_feat)  # 偏向混合特征
        }

        return weights


class OutputHead(nn.Module):
    """输出头网络"""

    def __init__(self, in_channels):
        super(OutputHead, self).__init__()

        # 新增通道调整层
        self.channel_adjust = nn.Conv2d(in_channels, 64, 3, 1, 1)

        self.s0_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1)
        )

        self.dop_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.aop_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

        self.ellip_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

        # # S0光强输出头
        # self.s0_head = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(in_channels // 2, 1, 3, 1, 1)
        # )
        #
        # # 全偏振度输出头
        # self.dop_head = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(in_channels // 2, 1, 3, 1, 1),
        #     nn.Sigmoid()
        # )
        #
        # # 偏振角输出头
        # self.aop_head = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(in_channels // 2, 1, 3, 1, 1),
        #     nn.Tanh()  # 映射到[-1,1]
        # )
        #
        # # 椭圆率输出头
        # self.ellip_head = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(in_channels // 2, 1, 3, 1, 1),
        #     nn.Tanh()  # 映射到[-1,1]
        # )

    def forward(self, x):
        return {
            's0': self.s0_head(x),
            'dop': self.dop_head(x),
            'aop': self.aop_head(x) * math.pi,  # 映射到[-π,π]
            'ellip': self.ellip_head(x) * (math.pi / 4)  # 映射到[-π/4,π/4]
        }
