import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from src.feature.base import BaseProcessor


class KMeansBinProcessor(BaseProcessor):

    def __init__(self, n_bins=5, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = int(n_bins)
        self.bins = None
        self.stats = {}

    def fit(self, data: pd.Series):
        self.stats['null_count'] = data.isnull().sum().item()
        self.stats['null_ratio'] = self.stats['null_count'] / len(data)
        valid_data = data.dropna()
        unique_vals = valid_data.unique()
        if len(unique_vals) <= self.n_bins:
            self.bins = np.sort(unique_vals).astype(float)
            self.stats['actual_n_bins'] = len(self.bins)
        else:
            X = valid_data.values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=self.n_bins, random_state=42, n_init=10)
            kmeans.fit(X)
            centers = np.sort(kmeans.cluster_centers_.flatten())
            boundaries = (centers[:-1] + centers[1:]) / 2
            self.bins = boundaries
            self.stats['actual_n_bins'] = len(self.bins)

        if len(valid_data) > 0:
            indices = np.digitize(valid_data, self.bins, right=True) + 1
            unique, counts = np.unique(indices, return_counts=True)
            self.stats['bin_counts'] = dict(zip(unique.tolist(), counts.tolist()))
        else:
            self.stats['bin_counts'] = {}

    def transform(self, data: pd.Series) -> np.ndarray:
        if self.bins is None or len(self.bins) == 0:
            return np.zeros(len(data), dtype=np.int64)

        def _map(x):
            if pd.isna(x):
                return 0
            idx = np.digitize([x], self.bins, right=True)[0] + 1
            return idx

        return data.apply(_map).values.astype(np.int64)

    def save(self) -> dict:
        return {
            'n_bins': self.n_bins,
            'bins': self.bins.tolist() if self.bins is not None else [],
            **self.stats
        }

    def load(self, input_dict: dict):
        self.n_bins = input_dict.get('n_bins', 5)
        self.bins = np.asarray(input_dict['bins'])
        self.stats = {k: v for k, v in input_dict.items() if k not in ['bins', 'n_bins']}
