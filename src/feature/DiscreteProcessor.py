import numpy as np
import pandas as pd

from src.feature.base import BaseProcessor


class DiscreteProcessor(BaseProcessor):
    def __init__(self, handle_unknown=True, **kwargs):
        super().__init__(**kwargs)
        self.handle_unknown = handle_unknown
        self.value_to_idx = {}
        self.stats = {}

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
        return {'value_to_idx': self.value_to_idx, **self.stats}

    def load(self, input_dict):
        self.value_to_idx = input_dict['value_to_idx']