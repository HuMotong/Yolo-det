from easydict import EasyDict

cfg = EasyDict()

# 数据配置
cfg.data = EasyDict()
cfg.data.train_path = './data/train'
cfg.data.val_path = './data/val'
cfg.data.patch_size = 256
cfg.data.batch_size = 8
cfg.data.num_workers = 4

# 模型配置
cfg.model = EasyDict()
cfg.model.init_features = 64
cfg.model.num_blocks = 4
cfg.model.num_heads = 4
cfg.model.embedding_dim = 128
cfg.model.scale_factor = 1

# 训练配置
cfg.train = EasyDict()
cfg.train.epochs = 100
cfg.train.lr = 1e-4
cfg.train.weight_decay = 1e-4
cfg.train.save_path = './checkpoints'
