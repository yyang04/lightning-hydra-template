import numpy as np
import pandas as pd
from pandas import DataFrame

from src.preprocess import DataPreprocessor


class CategoricalFeature:
    def __init__(self, features, group, dtype):
        self.features = features
        self.group = group  # group.npy
        self.output_path = f"{group}.npy"
        self.dtype = dtype  # int16

    def save(self, df: DataFrame, dataProcessor: DataPreprocessor):
        final_df = pd.DataFrame()
        field_dims = []
        field_names = []
        for feature in self.features:
            if feature in df.columns:
                final_df[feature] = df[feature]
                processor = dataProcessor.processors[feature]
                field_dims.append(processor.stats['vocab_size'])
                field_names.append(processor.stats['feature_name'])

        np.save(f"{self.group}.npy", final_df.value)
        # save_meta



    def save_meta(self):
        pass

    def fit(self, data: pd.Series):
        self.stats['null_count'] = data.isnull().sum().item()
        self.stats['null_ratio'] = self.stats['null_count'] / len(data)
        self.stats['unique_count'] = data.nunique()
        valid_values = data.dropna().unique().tolist()
        offset = 1 if self.handle_unknown else 0
        self.value_to_idx = {value: idx + offset for idx, value in enumerate(valid_values)}
        self.stats['vocab_size'] = len(valid_values) + offset  # +1 for null/unknown

    def transform(self, data: pd.Series) -> pd.Series:
        def map_value(x):
            if pd.isna(x):
                return 0
            return self.value_to_idx.get(x, 0)  # 0 for unknown

        return data.map(map_value).values.astype(np.int64)

    def save(self):
        # 需要直接保存到文件里
        return

    def save(self, ):
        pass

    def load(self, input_dict):
        self.value_to_idx = input_dict['value_to_idx']
