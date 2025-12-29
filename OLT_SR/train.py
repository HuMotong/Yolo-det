import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import logging

from models.model import PolarizationSR
from utils.losses import PolarizationLoss
from data.dataset import build_dataloader


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建模型
        self.model = PolarizationSR(cfg).to(self.device)

        # 创建优化器和调度器
        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.train.epochs,
            eta_min=1e-7
        )

        # 创建损失函数
        self.criterion = PolarizationLoss().to(self.device)

        # 创建数据加载器
        self.train_loader = build_dataloader(cfg, is_train=True)
        self.val_loader = build_dataloader(cfg, is_train=False)

        # 创建日志
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.cfg.train.save_path, 'train.log')),
                logging.StreamHandler()
            ]
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            # 准备数据
            linear_pol = batch['linear_pol'].to(self.device)
            circular_pol = batch['circular_pol'].to(self.device)
            gt_images = {k: v.to(self.device) for k, v in batch['gt_images'].items()}

            # 前向传播
            self.optimizer.zero_grad()
            pred = self.model(linear_pol, circular_pol)

            # 计算损失
            loss_dict = self.criterion(pred, gt_images)
            loss = loss_dict['total_loss']

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 更新进度条
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # 更新学习率
        self.scheduler.step()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 准备数据
                linear_pol = batch['linear_pol'].to(self.device)
                circular_pol = batch['circular_pol'].to(self.device)
                gt_images = {k: v.to(self.device) for k, v in batch['gt_images'].items()}

                # 前向传播
                pred = self.model(linear_pol, circular_pol)

                # 计算损失
                loss_dict = self.criterion(pred, gt_images)
                total_loss += loss_dict['total_loss'].item()

        return total_loss / len(self.val_loader)

    def train(self):
        best_loss = float('inf')

        for epoch in range(self.cfg.train.epochs):
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss = self.validate()

            # 记录日志
            logging.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': best_loss
                }, os.path.join(self.cfg.train.save_path, 'best_model.pth'))
