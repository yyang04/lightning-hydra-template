import numpy as np
import pandas as pd

from src.feature.base import BaseProcessor


class CustomBinProcessor(BaseProcessor):

    def __init__(self, bins, **kwargs):
        super().__init__(**kwargs)
        self.bins = np.asarray(bins).astype(float)
        self.stats = {}

    def fit(self, data: pd.Series):
        self.stats['null_count'] = data.isnull().sum().item()
        self.stats['null_ratio'] = self.stats['null_count'] / len(data)
        valid_data = data.dropna()
        if len(valid_data) > 0:
            indices = np.digitize(valid_data, self.bins, right=True) + 1
            unique, counts = np.unique(indices, return_counts=True)
            bin_counts = dict(zip(unique.tolist(), counts.tolist()))
            self.stats['bin_counts'] = bin_counts
            self.stats['vocab_size'] = len(unique)
        else:
            self.stats['bin_counts'] = {}
            self.stats['vocab_size'] = 0

    def transform(self, data: pd.Series) -> np.ndarray:
        def _map(x):
            if pd.isna(x):
                return 0
            idx = np.digitize([x], self.bins, right=True)[0] + 1
            return idx
        return data.apply(_map).values.astype(np.int64)

    def save(self) -> dict:
        return {'bins': self.bins.tolist(), **self.stats}

    def load(self, input_dict: dict):
        self.bins = np.asarray(input_dict['bins'])
        self.stats = {k: v for k, v in input_dict.items() if k != 'bins'}