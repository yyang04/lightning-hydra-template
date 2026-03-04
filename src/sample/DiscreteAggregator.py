from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.preprocess import DataPreprocessor
from src.sample.BaseAggregator import BaseAggregator


class DiscreteAggregator(BaseAggregator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save_samples(self, df: DataFrame, path):
        final_df = pd.DataFrame()
        for feature in self.features:
            if feature in df.columns:
                final_df[feature] = df[feature]
        final_df.astype(self.dtype)
        filepath = Path(path) / f"{self.group}.npy"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, final_df.values)

    def save(self, processors):
        field_dims = []
        field_names = []
        for feature in self.features:
            if feature in processors:
                processor = processors[feature]
                field_dims.append(processor.stats['vocab_size'])
                field_names.append(feature)

        return {
            "field_dims": field_dims,
            "field_names": field_names
        }

