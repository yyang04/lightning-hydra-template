import numpy as np
import pandas as pd

from src.feature.base import BaseProcessor


class ContinuousProcessor(BaseProcessor):

    def __init__(self, impute_strategy='mean', scale_strategy='standard', **kwargs):
        """
        :param impute_strategy: 缺失值填充策略 -> 'zero' (0值), 'mean' (均值), 'median' (中位数)
        :param scale_strategy: 数据缩放策略 -> 'minmax' (0-1归一化), 'standard' (正态分布/Z-score)
        """
        super().__init__(**kwargs)
        valid_impute = {'zero', 'mean', 'median'}
        valid_scale = {'minmax', 'standard', 'origin'}

        if impute_strategy not in valid_impute:
            raise ValueError(f"impute_strategy must be one of {valid_impute}")
        if scale_strategy not in valid_scale:
            raise ValueError(f"scale_strategy must be one of {valid_scale}")

        self.impute_strategy = impute_strategy
        self.scale_strategy = scale_strategy
        self.params = {}

    def fit(self, data: pd.Series):

        # 统计信息
        self.params['min'] = float(data.min())
        self.params['max'] = float(data.max())
        self.params['range'] = self.params['max'] - self.params['min']
        self.params['range'] = 1.0 if self.params['range'] == 0 else self.params['range']
        self.params['mean'] = float(data.mean())
        self.params['median'] = float(data.median())
        self.params['std'] = 1 if float(data.std()) == 0 else float(data.std())

        if self.impute_strategy == 'mean':
            fill_val = float(data.mean())
        elif self.impute_strategy == 'median':
            fill_val = float(data.median())
        else:
            fill_val = 0.0
        self.params['fill_value'] = float(fill_val)

    def transform(self, data: pd.Series) -> np.ndarray:

        filled_data = data.fillna(self.params['fill_value'])
        X = filled_data.values.astype(float)
        if self.scale_strategy == 'minmax':
            scaled_data = (X - self.params['min']) / self.params['range']
        elif self.scale_strategy == 'standard':
            scaled_data = (X - self.params['mean']) / self.params['std']
        else:
            scaled_data = X
        return scaled_data

    def save(self) -> dict:
        return {
            'impute_strategy': self.impute_strategy,
            'scale_strategy': self.scale_strategy,
            'params': self.params
        }

    def load(self, input_dict: dict):
        self.impute_strategy = input_dict['impute_strategy']
        self.scale_strategy = input_dict['scale_strategy']
        self.params = input_dict['params']