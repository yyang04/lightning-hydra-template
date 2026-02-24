from abc import abstractmethod

import pandas as pd


class BaseProcessor:

    @abstractmethod
    def __init__(self, feature, source):
        self.feature = feature
        self.source = source

    @abstractmethod
    def fit(self, df: pd.Series):
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.Series):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self, input_dict):
        raise NotImplementedError


