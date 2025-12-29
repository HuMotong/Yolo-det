import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os


class PolarizationDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, is_train=True):
        """
        参数:
            root_dir (str): 数据根目录
            patch_size (int): 训练patch大小
            is_train (bool): 是否为训练模式
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.is_train = is_train

        # 获取所有图像路径
        self.linear_pol_paths = sorted(
            [os.path.join(root_dir, 'linear', f)
             for f in os.listdir(os.path.join(root_dir, 'linear'))])
        self.circular_pol_paths = sorted(
            [os.path.join(root_dir, 'circular', f)
             for f in os.listdir(os.path.join(root_dir, 'circular'))])
        self.ground_truth_paths = sorted(
            [os.path.join(root_dir, 'gt', f)
             for f in os.listdir(os.path.join(root_dir, 'gt'))])

    def __len__(self):
        return len(self.linear_pol_paths)

    def __getitem__(self, idx):
        # 读取图像
        linear_pol = self._load_image(self.linear_pol_paths[idx])
        circular_pol = self._load_image(self.circular_pol_paths[idx])
        gt_images = self._load_ground_truth(self.ground_truth_paths[idx])

        # 训练时随机裁剪
        if self.is_train:
            # 随机翻转
            if np.random.random() > 0.5:
                linear_pol = np.flip(linear_pol, axis=0)
                circular_pol = np.flip(circular_pol, axis=0)
                gt_images = np.flip(gt_images, axis=0)

            # 随机旋转
            k = np.random.randint(0, 4)
            linear_pol = np.rot90(linear_pol, k)
            circular_pol = np.rot90(circular_pol, k)
            gt_images = np.rot90(gt_images, k)

        # 转换为tensor
        linear_pol = torch.from_numpy(linear_pol).float()
        circular_pol = torch.from_numpy(circular_pol).float()
        gt_images = torch.from_numpy(gt_images).float()

        return {
            'linear_pol': linear_pol,
            'circular_pol': circular_pol,
            'gt_images': gt_images
        }

    def _load_image(self, path):
        """加载图像并归一化"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        return img

    def _load_ground_truth(self, path):
        """加载地面真值（S0, DoP, AoP, 椭圆率）"""
        gt = np.load(path)  # 假设ground truth以.npy格式存储
        return gt.astype(np.float32)

    def _random_crop(self, linear, circular, gt):
        """随机裁剪"""
        h, w = linear.shape
        top = np.random.randint(0, h - self.patch_size)
        left = np.random.randint(0, w - self.patch_size)

        linear = linear[top:top + self.patch_size, left:left + self.patch_size]
        circular = circular[top:top + self.patch_size, left:left + self.patch_size]
        gt = gt[..., top:top + self.patch_size, left:left + self.patch_size]

        return linear, circular, gt

def build_dataloader(cfg, is_train=True):
    """构建数据加载器"""
    dataset = PolarizationDataset(
        root_dir=cfg.data.train_path if is_train else cfg.data.val_path,
        patch_size=cfg.data.patch_size,
        is_train=is_train
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=is_train,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=is_train
    )

    return dataloader
