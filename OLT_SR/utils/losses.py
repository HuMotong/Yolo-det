import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SSIM(nn.Module):
    """结构相似度损失"""

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

    def create_window(self, window_size):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size)
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()


class PolarizationLoss(nn.Module):
    """偏振超分辨率损失函数"""

    def __init__(self):
        super(PolarizationLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM()

    def weight_guidance_loss(self, weights):
        """添加权重导向损失"""
        loss = 0

        # S0和AoP的权重应该偏向线偏振
        for k in ['s0', 'aop']:
            loss += F.l1_loss(weights[k][:, 0], torch.ones_like(weights[k][:, 0]) * 0.5)  # 线偏振权重应较大
            loss += F.l1_loss(weights[k][:, 1], torch.zeros_like(weights[k][:, 1]) * 0.1)  # 圆偏振权重应较小
            loss += F.l1_loss(weights[k][:, 2], torch.ones_like(weights[k][:, 2]) * 0.4)  # 混合特征权重适中

        # DoP和Ellip的权重应该偏向混合特征
        for k in ['dop', 'ellip']:
            loss += F.l1_loss(weights[k][:, 0], torch.ones_like(weights[k][:, 0]) * 0.3)  # 线偏振权重适中
            loss += F.l1_loss(weights[k][:, 1], torch.ones_like(weights[k][:, 1]) * 0.3)  # 圆偏振权重适中
            loss += F.l1_loss(weights[k][:, 2], torch.ones_like(weights[k][:, 2]) * 0.4)  # 混合特征权重较大

        return loss

    def forward(self, pred, target, weights):
        # 基础重建损失
        loss_s0 = self.l1_loss(pred['s0'], target['s0']) + 0.5 * self.ssim_loss(pred['s0'], target['s0'])
        loss_dop = self.l1_loss(pred['dop'], target['dop']) + 0.3 * self.ssim_loss(pred['dop'], target['dop'])
        loss_aop = self.l1_loss(pred['aop'], target['aop']) + 0.7 * self.ssim_loss(pred['aop'], target['aop'])
        loss_ellip = self.l1_loss(pred['ellip'], target['ellip']) + 0.2 * self.ssim_loss(pred['ellip'], target['ellip'])

        # 斯托克斯不等式约束
        s0_sq = pred['s0'].pow(2)
        s1_sq = (pred['s0'] * pred['dop'] * torch.cos(2 * pred['aop'])).pow(2)
        s2_sq = (pred['s0'] * pred['dop'] * torch.sin(2 * pred['aop'])).pow(2)
        s3_sq = (pred['s0'] * pred['dop'] * torch.sin(2 * pred['ellip'])).pow(2)

        loss_stokes = F.relu(s1_sq + s2_sq + s3_sq - s0_sq + 1e-6).mean()

        # 偏振角周期性约束
        loss_angle = (1 - torch.cos(2 * (pred['aop'] - target['aop']))).mean()

        # 椭圆率边界约束
        loss_ellip_bound = F.relu(torch.abs(pred['ellip']) - math.pi / 4).mean()

        # 全偏振度归一化约束
        dop_norm = torch.sqrt(s1_sq + s2_sq + s3_sq) / (pred['s0'] + 1e-6)
        loss_dop_norm = F.mse_loss(dop_norm, pred['dop'])

        # 权重导向损失
        loss_weight = self.weight_guidance_loss(weights)

        # 总损失
        total_loss = (
                loss_s0 + loss_dop + loss_aop + loss_ellip +  # 基础重建损失
                0.5 * loss_stokes +  # 斯托克斯约束
                0.3 * loss_angle +  # 角度周期性约束
                0.2 * loss_ellip_bound +  # 椭圆率约束
                0.1 * loss_dop_norm +  # 偏振度约束
                0.1 * loss_weight  # 权重导向约束
        )

        return {
            'total_loss': total_loss,
            's0_loss': loss_s0,
            'dop_loss': loss_dop,
            'aop_loss': loss_aop,
            'ellip_loss': loss_ellip,
            'stokes_loss': loss_stokes,
            'angle_loss': loss_angle,
            'ellip_bound_loss': loss_ellip_bound,
            'dop_norm_loss': loss_dop_norm,
            'weight_loss': loss_weight
        }
