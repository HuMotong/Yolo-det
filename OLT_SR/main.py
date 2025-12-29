import argparse
import os
from configs.config import cfg
from train import Trainer
from test import Tester


def parse_args():
    parser = argparse.ArgumentParser(description='Polarization Super-Resolution')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='train or test mode')
    parser.add_argument('--config', type=str, default='configs/config.py',
                        help='config file path')
    parser.add_argument('--test_folder', type=str, default='./test_data',
                        help='folder containing test images')
    parser.add_argument('--save_folder', type=str, default='./results',
                        help='folder for saving results')
    return parser.parse_args()


def main():
    args = parse_args()

    # 创建必要的文件夹
    os.makedirs(cfg.train.save_path, exist_ok=True)
    os.makedirs(args.save_folder, exist_ok=True)

    if args.mode == 'train':
        trainer = Trainer(cfg)
        trainer.train()
    else:
        tester = Tester(cfg)
        tester.test_folder(args.test_folder, args.save_folder)


if __name__ == '__main__':
    main()
