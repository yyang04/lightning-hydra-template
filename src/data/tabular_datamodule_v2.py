import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class HydraDataset(Dataset):
    def __init__(self, processed_data, labels=None):
        """
        processed_data: 你的预处理 transform 返回的字典
        labels: 标签数据 (numpy array), 可选
        """
        # 1. 连续特征矩阵 (N, num_cont) -> Float32
        # 使用 .size 判断是否存在该类特征，防止报错
        self.x_cont = processed_data['continuous_data']

        # 2. 离散特征矩阵 (N, num_cat) -> Int64
        self.x_cat = processed_data['categorical_data']

        # 3. 序列特征字典 {'seq_name': array(N, T)}
        self.seq_data = processed_data['sequence_data']
        # 提前把 key 列表存下来，固定顺序
        self.seq_features = list(self.seq_data.keys())

        # 4. 标签
        self.labels = labels

    def __len__(self):
        # 以连续特征的行数为准，或者离散特征的行数
        if self.x_cont is not None and len(self.x_cont) > 0:
            return len(self.x_cont)
        return len(self.x_cat)

    def __getitem__(self, idx):
        item = {}

        # --- 离散单值特征 ---
        if self.x_cat is not None and self.x_cat.size > 0:
            item['x_cat'] = torch.from_numpy(self.x_cat[idx])

        # --- 连续单值特征 ---
        if self.x_cont is not None and self.x_cont.size > 0:
            item['x_cont'] = torch.from_numpy(self.x_cont[idx])

        # --- 序列特征 ---
        for name in self.seq_features:
            item[name] = torch.from_numpy(self.seq_data[name][idx])

        # --- 标签 ---
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


class HydraModel(nn.Module):
    def __init__(self, meta_data, embedding_dim=32):
        super().__init__()
        self.meta_data = meta_data
        self.feature_info = meta_data['feature_info']

        # ------------------------------------------------
        # 1. 构建离散特征 Embedding 层 (单值)
        # ------------------------------------------------
        self.cat_embeddings = nn.ModuleDict()
        self.cat_input_dims = []  # 记录离散特征的名字和顺序

        # 遍历配置中定义的特征顺序 (假设你知道离散特征列表)
        # 这里为了演示，我们遍历 meta_data，实际工程中通常结合 cfg.features.discrete_features 遍历
        for name, info in self.feature_info.items():
            if info['processor_type'] == 'discrete':
                vocab_size = info['stats']['vocab_size']
                # 创建 Embedding: (Vocab+1, Dim)
                self.cat_embeddings[name] = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.cat_input_dims.append(name)

        # ------------------------------------------------
        # 2. 构建离散序列特征 Embedding 层
        # ------------------------------------------------
        self.seq_embeddings = nn.ModuleDict()
        self.seq_features = []

        for name, info in self.feature_info.items():
            if info['processor_type'] == 'discrete_sequence':
                vocab_size = info['stats']['vocab_size']
                self.seq_embeddings[name] = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.seq_features.append(name)

        # ------------------------------------------------
        # 3. 计算 MLP 输入的总维度
        # ------------------------------------------------
        # 离散单值总维数 + 连续单值数量 + 序列特征(经过Pooling后的)维数

        # 统计连续特征数量
        num_continuous = sum(1 for info in self.feature_info.values() if info['processor_type'] == 'continuous')

        total_input_dim = (len(self.cat_input_dims) * embedding_dim) + \
                          num_continuous + \
                          (len(self.seq_features) * embedding_dim)  # 假设序列Pooling后也是 emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # 假设是二分类 CTR 任务
        )

    def forward(self, inputs):
        """
        inputs: Dataset 返回的字典 batch
        """
        feature_list = []

        if 'x_cat' in inputs:
            x_cat = inputs['x_cat']  # shape: (B, num_cat_features)

            for i, feature_name in enumerate(self.cat_input_dims):
                col_data = x_cat[:, i]
                emb = self.cat_embeddings[feature_name](col_data)  # (B, Emb_Dim)
                feature_list.append(emb)

        if 'x_cont' in inputs:
            feature_list.append(inputs['x_cont'])  # (B, num_cont)

        for feature_name in self.seq_features:
            seq_data = inputs[feature_name]  # (B, Seq_Len)
            seq_emb = self.seq_embeddings[feature_name](seq_data)  # (B, Seq_Len, Emb_Dim)
            seq_pooled = torch.mean(seq_emb, dim=1)  # (B, Emb_Dim)
            feature_list.append(seq_pooled)

        concat_features = torch.cat(feature_list, dim=1)

        logits = self.mlp(concat_features)
        return logits.squeeze(-1)


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class HydraDataModule(pl.LightningDataModule):
    def __init__(
            self,
            df: pd.DataFrame,
            cfg,
            label_col: str = 'label',
            batch_size: int = 32,
            num_workers: int = 4,
            val_size: float = 0.1,
            test_size: float = 0.1,
            seed: int = 42
    ):
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
        train_val_df, test_df = train_test_split(self.raw_df, test_size=self.test_size, random_state=self.seed)
        # 再从剩余的分出 Val
        # 注意：这里重新计算 val_size 比例
        relative_val_size = self.val_size / (1 - self.test_size)
        train_df, val_df = train_test_split(train_val_df, test_size=relative_val_size, random_state=self.seed)

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