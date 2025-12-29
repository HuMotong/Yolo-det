import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
import os
from collections import OrderedDict

from models.model import PolarizationSR


class Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建模型并加载权重
        self.model = PolarizationSR(cfg).to(self.device)
        self.load_checkpoint()
        self.model.eval()

    def load_checkpoint(self):
        checkpoint = torch.load(
            os.path.join(self.cfg.train.save_path, 'best_model.pth'),
            map_location=self.device
        )
        # 处理多GPU训练的模型权重加载
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # 删除'module.'前缀
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def process_large_image(self, linear_pol, circular_pol, patch_size=1024, overlap=32):
        """处理大尺寸图像的分块处理函数"""
        H, W = linear_pol.shape[2:]

        # 计算需要的填充
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size

        # 对图像进行填充
        linear_pol = F.pad(linear_pol, (0, pad_w, 0, pad_h), mode='reflect')
        circular_pol = F.pad(circular_pol, (0, pad_w, 0, pad_h), mode='reflect')

        # 获取填充后的尺寸
        pH, pW = linear_pol.shape[2:]

        # 初始化输出字典
        outputs = {
            's0': torch.zeros((1, 1, pH * 4, pW * 4), device=self.device),
            'dop': torch.zeros((1, 1, pH * 4, pW * 4), device=self.device),
            'aop': torch.zeros((1, 1, pH * 4, pW * 4), device=self.device),
            'ellip': torch.zeros((1, 1, pH * 4, pW * 4), device=self.device),
            'count': torch.zeros((1, 1, pH * 4, pW * 4), device=self.device)
        }

        # 分块处理
        for h in range(0, pH - overlap, patch_size - overlap):
            for w in range(0, pW - overlap, patch_size - overlap):
                # 提取patch
                h_end = min(h + patch_size, pH)
                w_end = min(w + patch_size, pW)
                h_start = max(0, h_end - patch_size)
                w_start = max(0, w_end - patch_size)

                linear_patch = linear_pol[:, :, h_start:h_end, w_start:w_end]
                circular_patch = circular_pol[:, :, h_start:h_end, w_start:w_end]

                # 处理patch
                with torch.no_grad():
                    pred = self.model(linear_patch, circular_patch)

                # 计算输出位置
                out_h_start = h_start
                out_h_end = h_end * 4
                out_w_start = w_start
                out_w_end = w_end * 4

                # out_h_start = h_start * 4
                # out_h_end = h_end * 4
                # out_w_start = w_start * 4
                # out_w_end = w_end * 4

                # 累加结果
                for k in pred.keys():
                    outputs[k][:, :, out_h_start:out_h_end, out_w_start:out_w_end] += pred[k]
                outputs['count'][:, :, out_h_start:out_h_end, out_w_start:out_w_end] += 1

        # 平均重叠区域
        for k in pred.keys():
            outputs[k] = outputs[k] / outputs['count']

        # 裁剪到原始尺寸
        final_outputs = {}
        for k in pred.keys():
            final_outputs[k] = outputs[k][:, :, :H * 4, :W * 4]

        return final_outputs

    def test_single_image(self, linear_pol_path, circular_pol_path, save_path):
        """测试单张图像"""
        # 读取图像
        linear_pol = cv2.imread(linear_pol_path, cv2.IMREAD_GRAYSCALE)
        circular_pol = cv2.imread(circular_pol_path, cv2.IMREAD_GRAYSCALE)

        # 归一化
        linear_pol = torch.from_numpy(linear_pol).float().unsqueeze(0).unsqueeze(0) / 255.0
        circular_pol = torch.from_numpy(circular_pol).float().unsqueeze(0).unsqueeze(0) / 255.0

        # 转移到设备
        linear_pol = linear_pol.to(self.device)
        circular_pol = circular_pol.to(self.device)

        # 处理图像
        outputs = self.process_large_image(linear_pol, circular_pol)

        # 保存结果
        os.makedirs(save_path, exist_ok=True)
        for k, v in outputs.items():
            # 转换为numpy数组
            img = v.cpu().numpy()[0, 0]

            # 根据不同输出类型进行后处理
            if k == 's0':
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            elif k == 'dop':
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            elif k == 'aop':
                img = ((img + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            elif k == 'ellip':
                img = ((img + np.pi / 4) / (np.pi / 2) * 255).astype(np.uint8)

            # 保存图像
            cv2.imwrite(os.path.join(save_path, f'{k}.png'), img)

            # 如果是角度类型，额外保存热图
            if k in ['aop', 'ellip']:
                heatmap = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
                cv2.imwrite(os.path.join(save_path, f'{k}_heatmap.png'), heatmap)

    def test_folder(self, test_folder, save_folder):
        """测试文件夹中的所有图像"""
        linear_folder = os.path.join(test_folder, 'linear')
        circular_folder = os.path.join(test_folder, 'circular')

        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(linear_folder) if f.endswith(('.png', '.jpg', '.bmp'))])

        for img_file in tqdm(image_files, desc='Testing'):
            linear_path = os.path.join(linear_folder, img_file)
            circular_path = os.path.join(circular_folder, img_file)
            save_path = os.path.join(save_folder, img_file.split('.')[0])

            self.test_single_image(linear_path, circular_path, save_path)
