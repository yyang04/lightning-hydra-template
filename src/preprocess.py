import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from typing import Dict
from pathlib import Path
import logging


from src.feature.BaseProcessor import BaseProcessor
from src.sample.BaseAggregator import BaseAggregator

logger = logging.getLogger(__name__)


class DataPreprocessor:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.processors: Dict[str, BaseProcessor] = {}
        self.aggregators: Dict[str, BaseAggregator] = {}

        self.df = None
        self.processed_df = pd.DataFrame()

        self._init_path()
        self._initialize_processors()
        self._initiate_aggregators()

    def _init_path(self):
        output_path = Path(self.cfg.output_path)
        sample_path = Path(self.cfg.sample_path)
        output_path.mkdir(parents=True, exist_ok=True)
        sample_path.mkdir(parents=True, exist_ok=True)
        self.meta_file = output_path / 'meta.yaml'
        self.data_file = output_path / 'data.parquet'
        self.sample_meta_file = sample_path / 'meta.yaml'
        self.sample_path = sample_path

    def load_data(self):
        if self.df is None:
            self.df = pd.read_parquet(self.cfg.input_file)

    def _initialize_processors(self):
        for feature in self.cfg.get('features', []):
            processor = instantiate(feature)
            self.processors[processor.feature] = processor

    def _initiate_aggregators(self):
        for sample in self.cfg.get('samples', []):
            aggregator = instantiate(sample)
            self.aggregators[aggregator.group] = aggregator

    def fit(self):
        self.load_data()
        dump_dict = {}
        for feature_name, processor in self.processors.items():
            if processor.source in self.df.columns:
                processor.fit(self.df[processor.source])
                dump_dict[feature_name] = processor.save()
        self.save_feature_meta(dump_dict)

    def transform(self):
        self.load_data()
        new_columns = {}
        for feature_name, processor in self.processors.items():
            if processor.source in self.df.columns:
                new_columns[feature_name] = processor.transform(self.df[processor.source])

        if new_columns:
            self.processed_df = pd.DataFrame(new_columns, index=self.df.index)

    def fit_transform(self):
        self.fit()
        self.transform()
        self.save_data()

    def load_transform(self):
        self.load_feature_meta()
        self.transform()

    def save_feature_meta(self, dump_dict):
        OmegaConf.save(dump_dict, self.meta_file)

    def load_feature_meta(self):
        dump_dict = OmegaConf.load(self.meta_file)
        for feature_name, processor in self.processors.items():
            if feature_name in dump_dict:
                processor.load(dump_dict[feature_name])

    def save_data(self):
        if not self.processed_df.empty:
            self.processed_df.to_parquet(self.data_file)

    def save_samples(self):
        dump_dict = {}
        for group, aggregator in self.aggregators.items():
            aggregator.save_samples(df=self.processed_df, path=self.cfg.sample_path)
            dump_dict[group] = aggregator.save(self.processors)
        self.save_sample_meta(dump_dict)

    def save_sample_meta(self, dump_dict):
        OmegaConf.save(dump_dict, self.sample_meta_file)


@hydra.main(version_base="1.3", config_path="../configs/preprocess", config_name="titanic.yaml")
def main(cfg: DictConfig):
    dataPreprocessor = DataPreprocessor(cfg)
    dataPreprocessor.fit_transform()
    dataPreprocessor.save_samples()
    return


if __name__ == "__main__":
    main()
