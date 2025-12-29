import torch
import torch.nn as nn
import torch.nn.functional as F


class RRDB(nn.Module):
    """残差密集块"""

    def __init__(self, channels, growth_rate=32):
        super(RRDB, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(5):
            dense_layer = nn.Sequential(
                nn.Conv2d(channels + i * growth_rate, growth_rate, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True)
            )
            self.layers.append(dense_layer)

        self.conv1x1 = nn.Conv2d(channels + 5 * growth_rate, channels, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(inputs, 1))
            inputs.append(out)

        out = self.conv1x1(torch.cat(inputs, 1))
        return x + out * 0.2


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        y = torch.sigmoid(self.conv(x_cat))
        return x * y


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, init_features=64):
        super(FeatureExtractor, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, init_features, 3, 2, 1),  # 下采样到1/2
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.body = nn.ModuleList([
            RRDB(init_features) for _ in range(4)
        ])

        self.conv_body = nn.Conv2d(init_features, init_features, 3, 2, 1)  # 进一步下采样到1/4
        self.channel_attention = ChannelAttention(init_features)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        feat = self.conv_first(x)  # 1/2分辨率
        body_feat = feat

        for block in self.body:
            body_feat = block(body_feat)

        body_feat = self.conv_body(body_feat)  # 1/4分辨率
        body_feat = self.channel_attention(body_feat)
        body_feat = self.spatial_attention(body_feat)

        return body_feat

