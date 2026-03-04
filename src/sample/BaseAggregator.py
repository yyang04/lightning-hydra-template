from abc import abstractmethod

import pandas as pd
from pandas import DataFrame


class BaseAggregator:

    @abstractmethod
    def __init__(self, features, dtype, group):
        self.features = features
        self.dtype = dtype
        self.group = group

    @abstractmethod
    def save_samples(self, df: DataFrame, path: str):
        raise NotImplementedError

    @abstractmethod
    def save(self, **kwargs):
        raise NotImplementedError

