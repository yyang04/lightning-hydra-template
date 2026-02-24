import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import joblib
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import logging

from src.feature.base import BaseProcessor

logger = logging.getLogger(__name__)


class DataPreprocessor:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.processors: Dict[str, BaseProcessor] = {}
        self._initialize_processors()

    def _initialize_processors(self):
        if hasattr(self.cfg, 'features'):
            for feature in self.cfg.features:
                processor = instantiate(feature)
                self.processors[processor.feature] = processor

    def fit(self, df: pd.DataFrame):
        for feature_name, processor in self.processors.items():
            if processor.source in df.columns:
                processor.fit(df[processor.source])

        dump_dict = {
            feature_name: processor.save()
            for feature_name, processor in self.processors.items()
        }
        self.save(dump_dict)

    def load(self):
        dump_dict = OmegaConf.load(self.cfg.dump_path)
        for feature_name, processor in self.processors.items():
            if feature_name in dump_dict:
                print(feature_name, dump_dict[feature_name])
                processor.load(dump_dict[feature_name])

    def save(self, dump_dict):
        filepath = Path(self.cfg.dump_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(dump_dict, filepath)

    def transform(self, df: pd.DataFrame):
        processed_df = pd.DataFrame()
        for feature_name, processor in self.processors.items():
            if processor.source in df.columns:
                print(feature_name)
                processed_df[feature_name] = processor.transform(df[processor.source])
        return processed_df

        # continuous_list = []
        # discrete_list = []
        # sequence_data = {}
        #
        # for feature in self.cfg.features.continuous_features:
        #     if feature in self.processors:
        #         arr = self.processors[feature].transform(df[feature])
        #         continuous_list.append(arr.reshape(-1, 1))
        #
        # for feature in self.cfg.features.discrete_features:
        #     if feature in self.processors:
        #         arr = self.processors[feature].transform(df[feature])
        #         discrete_list.append(arr.reshape(-1, 1))
        #
        # for feature in [self.cfg.features.discrete_sequence_features, self.cfg.features.continuous_sequence_features]:
        #     if feature in self.processors:
        #         sequence_data[feature] = self.processors[feature].transform(df[feature])
        #
        # X_cont = np.hstack(continuous_list) if continuous_list else np.array([])
        # X_cat = np.hstack(discrete_list) if discrete_list else np.array([])
        #
        # return {
        #     'continuous_data': X_cont,
        #     'categorical_data': X_cat,
        #     'sequence_data': sequence_data,
        #     'meta_data': self.meta_data
        # }

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.fit(df)
        return self.transform(df)

    def save_meta_data(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.meta_data, f, indent=2, ensure_ascii=False)

    # def save(self, path: str):
    #     Path(path).parent.mkdir(parents=True, exist_ok=True)
    #     joblib.dump(self, path)
    #     logger.info(f"Preprocessor saved to {path}")









class DiscreteSequenceProcessor:
    def __init__(self, feature_name: str, config: Dict):
        self.feature_name = feature_name
        self.config = config
        self.sequence_type = True
        self.value_to_idx = {}
        self.stats = {}

    def fit(self, data: pd.Series):
        self.stats['null_count'] = data.isnull().sum()
        self.stats['null_ratio'] = self.stats['null_count'] / len(data)

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
                if len(encoded) < max_len:  # 需要在后面填充 padding value
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
    def __init__(self, feature_name: str, config: Dict):
        self.feature_name = feature_name
        self.config = config
        self.sequence_type = True
        self.scaler = None
        self.stats = {}

    def fit(self, data: pd.Series):
        self.stats['null_count'] = data.isnull().sum()
        self.stats['null_ratio'] = self.stats['null_count'] / len(data)
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


@hydra.main(version_base="1.3", config_path="../configs/preprocess", config_name="titanic.yaml")
def main(cfg: DictConfig):
    print("Loading configuration...")
    print(OmegaConf.to_yaml(cfg))

    df = pd.read_csv(cfg.input_path)
    dataPreprocessor = DataPreprocessor(cfg)
    dataPreprocessor.fit(df)
    # dataPreprocessor.load()
    processed: DataFrame = dataPreprocessor.transform(df)
    filepath = Path(cfg.output_path)
    processed.to_parquet(filepath, index=False)
    return

    # sample_data = {
    #     'user_id': [1, 2, 3, 4, 5],
    #     'age': [25, 30, 35, None, 40],
    #     'category': ['A', 'B', 'A', 'C', None],
    #     'price': [100.5, 200.3, 0, 150.2, 180.7],
    #     'income': [50000, 60000, 75000, 55000, 80000],
    #     'click_history': [
    #         [101, 102, 103],
    #         [102, 104],
    #         [101, 103, 105],
    #         [],
    #         [102, 103]
    #     ]
    # }
    #
    # df = pd.DataFrame(sample_data)
    # print("Original data:")
    # print(df)

    # preprocessor = DataPreprocessor(cfg)
    # processed_data = preprocessor.fit_transform(df)
    #
    # print("\nProcessed tabular data:")
    # print(processed_data['tabular_data'])
    #
    # print("\nProcessed sequence data:")
    # for feature, sequences in processed_data['sequence_data'].items():
    #     print(f"{feature}: {sequences[:2]}...")  # 只显示前两个序列
    #
    # # 保存元数据
    # preprocessor.save_meta_data('feature_meta_data.json')
    # print(f"\nMeta data saved to: feature_meta_data.json")
    #
    # # 在 PyTorch 中使用元数据创建 embedding 层
    # print("\nEmbedding information for PyTorch:")
    # for feature, info in processed_data['meta_data']['feature_info'].items():
    #     if info['processor_type'] in ['discrete', 'discrete_sequence']:
    #         vocab_size = info['stats']['vocab_size']
    #         embedding_dim = min(600, round(1.6 * vocab_size ** 0.56))  # 经验公式
    #         print(f"{feature}: vocab_size={vocab_size}, recommended embedding_dim={embedding_dim}")


if __name__ == "__main__":
    main()
