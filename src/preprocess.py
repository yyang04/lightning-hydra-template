import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Any, Optional
import json
from pathlib import Path


class HydraDataPreprocessor:
    """基于 Hydra 配置的数据预处理器"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.processors = {}
        self.meta_data = {
            'feature_info': {},
            'processing_config': OmegaConf.to_container(cfg.processing),
            'hydra_config': OmegaConf.to_container(cfg)
        }

    def _initialize_processors(self):
        """根据配置初始化处理器"""
        # 离散值处理器
        for feature in self.cfg.features.discrete_features:
            self.processors[feature] = DiscreteProcessor(
                feature,
                self.cfg.processing.discrete
            )

        # 连续值处理器
        for feature in self.cfg.features.continuous_features:
            self.processors[feature] = ContinuousProcessor(
                feature,
                self.cfg.processing.continuous
            )

        # 离散序列处理器
        if self.cfg.features.discrete_sequence_features:
            for feature in self.cfg.features.discrete_sequence_features:
                self.processors[feature] = DiscreteSequenceProcessor(
                    feature,
                    self.cfg.processing.sequences
                )

        # 连续序列处理器
        if self.cfg.features.continuous_sequence_features:
            for feature in self.cfg.features.continuous_sequence_features:
                self.processors[feature] = ContinuousSequenceProcessor(
                    feature,
                    self.cfg.processing.sequences
                )

    def fit(self, df: pd.DataFrame):
        """拟合所有处理器"""
        self._initialize_processors()

        for feature_name, processor in self.processors.items():
            if feature_name in df.columns:
                print(f"Fitting processor for: {feature_name}")
                processor.fit(df[feature_name])
                self.meta_data['feature_info'][feature_name] = processor.get_meta_data()

    def transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """转换数据"""
        processed_df = df.copy()
        sequence_data = {}

        for feature_name, processor in self.processors.items():
            if feature_name not in df.columns:
                print(f"Warning: Feature {feature_name} not found in dataframe")
                continue

            if hasattr(processor, 'sequence_type'):
                # 序列数据单独处理
                transformed = processor.transform(df[feature_name])
                sequence_data[feature_name] = transformed
            else:
                # 表格数据
                processed_df[feature_name] = processor.transform(df[feature_name])

        return {
            'tabular_data': processed_df,
            'sequence_data': sequence_data,
            'meta_data': self.meta_data
        }

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """拟合并转换数据"""
        self.fit(df)
        return self.transform(df)

    def save_meta_data(self, filepath: str):
        """保存元数据"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.meta_data, f, indent=2, ensure_ascii=False)


class DiscreteProcessor:
    """离散值处理器"""

    def __init__(self, feature_name: str, config: Dict):
        self.feature_name = feature_name
        self.config = config
        self.value_to_idx = {}
        self.stats = {}

    def fit(self, data: pd.Series):
        # 统计信息
        self.stats['null_count'] = data.isnull().sum()
        self.stats['null_ratio'] = self.stats['null_count'] / len(data)
        self.stats['unique_count'] = data.nunique()

        # 构建编码映射 (从1开始，0留给空值和未知值)
        valid_values = data.dropna().unique()
        self.value_to_idx = {value: idx + 1 for idx, value in enumerate(valid_values)}
        self.stats['vocab_size'] = len(valid_values) + 1  # +1 for null/unknown

    def transform(self, data: pd.Series) -> pd.Series:
        def map_value(x):
            if pd.isna(x):
                return self.config.get('fillna_value', 0)
            return self.value_to_idx.get(x, 0)  # 未知值映射为0

        return data.map(map_value)

    def get_meta_data(self) -> Dict[str, Any]:
        return {
            'processor_type': 'discrete',
            'vocab_size': self.stats['vocab_size'],
            'stats': self.stats,
            'value_mapping': self.value_to_idx
        }


class ContinuousProcessor:
    """连续值处理器"""

    def __init__(self, feature_name: str, config: Dict):
        self.feature_name = feature_name
        self.config = config
        self.scaler = None
        self.stats = {}

    def fit(self, data: pd.Series):
        # 统计信息
        self.stats['null_count'] = data.isnull().sum()
        self.stats['null_ratio'] = self.stats['null_count'] / len(data)
        self.stats['zero_count'] = (data == 0).sum()
        self.stats['zero_ratio'] = self.stats['zero_count'] / len(data)
        self.stats['mean'] = data.mean()
        self.stats['std'] = data.std()
        self.stats['min'] = data.min()
        self.stats['max'] = data.max()

        # 初始化标准化器
        scaler_type = self.config.get('scaler', 'standard')
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()

        # 拟合标准化器（使用非空值）
        non_null_data = data.dropna().values.reshape(-1, 1)
        if len(non_null_data) > 0:
            self.scaler.fit(non_null_data)

    def transform(self, data: pd.Series) -> pd.Series:
        transformed = data.copy()
        non_null_mask = ~data.isnull()

        if self.scaler is not None and non_null_mask.any():
            values = data[non_null_mask].values.reshape(-1, 1)
            transformed_values = self.scaler.transform(values).flatten()
            transformed[non_null_mask] = transformed_values

        # 处理空值
        if non_null_mask.any():
            fillna_strategy = self.config.get('fillna_strategy', 'mean')
            if fillna_strategy == 'mean':
                fill_value = transformed[non_null_mask].mean()
            elif fillna_strategy == 'median':
                fill_value = transformed[non_null_mask].median()
            elif fillna_strategy == 'zero':
                fill_value = 0
            else:
                fill_value = 0

            transformed[~non_null_mask] = fill_value

        return transformed

    def get_meta_data(self) -> Dict[str, Any]:
        return {
            'processor_type': 'continuous',
            'scaler_type': self.config.get('scaler', 'standard'),
            'stats': self.stats
        }


class DiscreteSequenceProcessor:
    """离散序列处理器"""

    def __init__(self, feature_name: str, config: Dict):
        self.feature_name = feature_name
        self.config = config
        self.sequence_type = True
        self.value_to_idx = {}
        self.stats = {}

    def fit(self, data: pd.Series):
        self.stats['null_count'] = data.isnull().sum()
        self.stats['null_ratio'] = self.stats['null_count'] / len(data)

        # 收集所有序列中的唯一值
        all_values = set()
        sequence_lengths = []

        for sequence in data.dropna():
            if isinstance(sequence, (list, np.ndarray)):
                all_values.update(sequence)
                sequence_lengths.append(len(sequence))

        self.stats['unique_count'] = len(all_values)
        self.stats['vocab_size'] = len(all_values) + 1
        self.stats['avg_sequence_length'] = np.mean(sequence_lengths) if sequence_lengths else 0
        self.stats['max_sequence_length'] = max(sequence_lengths) if sequence_lengths else 0

        # 构建编码映射
        self.value_to_idx = {value: idx + 1 for idx, value in enumerate(all_values)}

    def transform(self, data: pd.Series) -> List[List[int]]:
        max_len = self.config.get('max_sequence_length', 50)
        padding_value = self.config.get('padding_value', 0)

        processed_sequences = []
        for sequence in data:
            if pd.isna(sequence) or sequence is None:
                processed_sequences.append([padding_value] * max_len)
            else:
                encoded = [self.value_to_idx.get(x, 0) for x in sequence]
                # 填充或截断序列
                if len(encoded) < max_len:
                    encoded = encoded + [padding_value] * (max_len - len(encoded))
                else:
                    encoded = encoded[:max_len]
                processed_sequences.append(encoded)

        return processed_sequences

    def get_meta_data(self) -> Dict[str, Any]:
        return {
            'processor_type': 'discrete_sequence',
            'stats': self.stats,
            'value_mapping': self.value_to_idx
        }


class ContinuousSequenceProcessor:
    """连续序列处理器"""

    def __init__(self, feature_name: str, config: Dict):
        self.feature_name = feature_name
        self.config = config
        self.sequence_type = True
        self.scaler = None
        self.stats = {}

    def fit(self, data: pd.Series):
        self.stats['null_count'] = data.isnull().sum()
        self.stats['null_ratio'] = self.stats['null_count'] / len(data)

        # 收集所有序列值用于计算统计量
        all_values = []
        sequence_lengths = []

        for sequence in data.dropna():
            if isinstance(sequence, (list, np.ndarray)):
                all_values.extend(sequence)
                sequence_lengths.append(len(sequence))

        self.stats['avg_sequence_length'] = np.mean(sequence_lengths) if sequence_lengths else 0
        self.stats['max_sequence_length'] = max(sequence_lengths) if sequence_lengths else 0

        if all_values:
            self.stats['mean'] = np.mean(all_values)
            self.stats['std'] = np.std(all_values)
            self.scaler = StandardScaler()
            self.scaler.fit(np.array(all_values).reshape(-1, 1))

    def transform(self, data: pd.Series) -> List[List[float]]:
        max_len = self.config.get('max_sequence_length', 50)
        padding_value = self.config.get('padding_value', 0.0)

        processed_sequences = []
        for sequence in data:
            if pd.isna(sequence) or sequence is None:
                processed_sequences.append([padding_value] * max_len)
            else:
                if self.scaler is not None:
                    sequence_array = np.array(sequence).reshape(-1, 1)
                    standardized = self.scaler.transform(sequence_array).flatten().tolist()
                else:
                    standardized = sequence

                # 填充或截断序列
                if len(standardized) < max_len:
                    standardized = standardized + [padding_value] * (max_len - len(standardized))
                else:
                    standardized = standardized[:max_len]
                processed_sequences.append(standardized)

        return processed_sequences

    def get_meta_data(self) -> Dict[str, Any]:
        return {
            'processor_type': 'continuous_sequence',
            'stats': self.stats
        }


@hydra.main(version_base="1.3", config_path="../configs/preprocess", config_name="config")
def main(cfg: DictConfig):
    """主处理函数"""
    print("Loading configuration...")
    print(OmegaConf.to_yaml(cfg))

    # 创建示例数据（在实际应用中，你会从文件加载）
    sample_data = {
        'user_id': [1, 2, 3, 4, 5],
        'age': [25, 30, 35, None, 40],
        'category': ['A', 'B', 'A', 'C', None],
        'price': [100.5, 200.3, 0, 150.2, 180.7],
        'income': [50000, 60000, 75000, 55000, 80000],
        'click_history': [
            [101, 102, 103],
            [102, 104],
            [101, 103, 105],
            [],
            [102, 103]
        ]
    }

    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)

    # 初始化预处理器
    preprocessor = HydraDataPreprocessor(cfg)

    # 执行预处理
    processed_data = preprocessor.fit_transform(df)

    print("\nProcessed tabular data:")
    print(processed_data['tabular_data'])

    print("\nProcessed sequence data:")
    for feature, sequences in processed_data['sequence_data'].items():
        print(f"{feature}: {sequences[:2]}...")  # 只显示前两个序列

    # 保存元数据
    preprocessor.save_meta_data('feature_meta_data.json')
    print(f"\nMeta data saved to: feature_meta_data.json")

    # 在 PyTorch 中使用元数据创建 embedding 层
    print("\nEmbedding information for PyTorch:")
    for feature, info in processed_data['meta_data']['feature_info'].items():
        if info['processor_type'] in ['discrete', 'discrete_sequence']:
            vocab_size = info['stats']['vocab_size']
            embedding_dim = min(600, round(1.6 * vocab_size ** 0.56))  # 经验公式
            print(f"{feature}: vocab_size={vocab_size}, recommended embedding_dim={embedding_dim}")


if __name__ == "__main__":
    main()