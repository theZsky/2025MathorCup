import paddle
import os


class BaseDataModule:
    def __init__(self, num_workers=4, batch_size=32):
        """
        初始化数据模块

        Args:
            num_workers: 数据加载的工作进程数
            batch_size: 批处理大小
        """
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None

        # 如果子类没定义 val_data，就默认给 None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup_transforms(self, train_transform=None, val_transform=None, test_transform=None):
        """
        设置数据转换和增强
        """
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    @property
    def train_dataset(self) -> paddle.io.Dataset:
        """
        注意：基类中不再 raise，而是直接返回 self.train_data
        如果子类没有赋值 self.train_data，这里就是 None
        """
        return self.train_data

    @property
    def val_dataset(self) -> paddle.io.Dataset:
        return self.val_data

    @property
    def test_dataset(self) -> paddle.io.Dataset:
        return self.test_data

    def train_dataloader(self, **kwargs) -> paddle.io.DataLoader:
        """训练数据加载器"""
        collate_fn = getattr(self, 'collate_fn', None)
        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': True,
            'drop_last': True
        }
        dataloader_kwargs.update(kwargs)

        return paddle.io.DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            **dataloader_kwargs
        )

    def val_dataloader(self, **kwargs) -> paddle.io.DataLoader:
        """验证数据加载器"""
        collate_fn = getattr(self, 'collate_fn', None)
        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': False,
            'drop_last': False
        }
        dataloader_kwargs.update(kwargs)

        return paddle.io.DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            **dataloader_kwargs
        )

    def test_dataloader(self, **kwargs) -> paddle.io.DataLoader:
        """测试数据加载器"""
        collate_fn = getattr(self, 'collate_fn', None)
        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': False,
            'drop_last': False
        }
        dataloader_kwargs.update(kwargs)

        return paddle.io.DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            **dataloader_kwargs
        )

    def apply_transform(self, data, transform):
        """应用数据转换和增强"""
        if transform is not None:
            return transform(data)
        return data
