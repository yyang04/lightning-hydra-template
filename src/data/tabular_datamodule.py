from lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class HydraDataModule(LightningDataModule):
    def __init__(self,
                 df: pd.DataFrame,
                 cfg,
                 label_col: str = 'label',
                 batch_size: int = 32,
                 num_workers: int = 4,
                 val_size: float = 0.1,
                 test_size: float = 0.1,
                 seed: int = 42):

        super().__init__()
        self.raw_df = df
        self.cfg = cfg
        self.label_col = label_col
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

        # 占位符
        self.preprocessor = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.meta_data = None  # 用于传递给模型构建 Embedding

    def setup(self, stage=None):
        """
        这里执行划分、拟合、转换逻辑。
        Lightning 会自动在主进程调用一次，不用担心多卡重复处理。
        """
        # 1. 划分数据集 (Train / Val / Test)
        # 先分出 Test
        train_val_df, test_df = train_test_split(
            self.raw_df, test_size=self.test_size, random_state=self.seed
        )
        # 再从剩余的分出 Val
        # 注意：这里重新计算 val_size 比例
        relative_val_size = self.val_size / (1 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=relative_val_size, random_state=self.seed
        )

        print(f"Data Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # 2. 初始化并拟合预处理器 (只在训练集上 Fit!)
        # 你的 HydraDataPreprocessor 需要在这里实例化
        self.preprocessor = HydraDataPreprocessor(self.cfg)

        print("Fitting preprocessor on Train set...")
        self.preprocessor.fit(train_df)

        # 保存元数据，模型初始化需要用到
        # 比如 vocab_size 等信息
        self.meta_data = self.preprocessor.meta_data

        # 3. 转换数据 (Transform)
        # 注意：extract_labels 是一个辅助函数(下面定义)，把 label 从 feature 中分离

        # 处理 Train
        train_feats = self.preprocessor.transform(train_df)
        train_labels = train_df[self.label_col].values.astype(np.float32) if self.label_col in train_df else None

        # 处理 Val
        val_feats = self.preprocessor.transform(val_df)
        val_labels = val_df[self.label_col].values.astype(np.float32) if self.label_col in val_df else None

        # 处理 Test
        test_feats = self.preprocessor.transform(test_df)
        test_labels = test_df[self.label_col].values.astype(np.float32) if self.label_col in test_df else None

        # 4. 构建 Dataset
        self.train_dataset = HydraDataset(train_feats, train_labels)
        self.val_dataset = HydraDataset(val_feats, val_labels)
        self.test_dataset = HydraDataset(test_feats, test_labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 训练集必须打乱
            num_workers=self.num_workers,
            pin_memory=True,  # 配合 GPU 训练加速
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def save_preprocessor(self, path):
        """暴露一个方法用于保存训练好的 preprocessor"""
        if self.preprocessor:
            self.preprocessor.save(path)