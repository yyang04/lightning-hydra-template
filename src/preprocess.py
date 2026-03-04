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
        self._initialize_processors()
        self.processed_df = pd.DataFrame()
        self.df = pd.read_parquet(self.cfg.data.processed_path)

    def _initialize_processors(self):
        if hasattr(self.cfg, 'features'):
            for feature in self.cfg.features:
                processor = instantiate(feature)
                self.processors[processor.feature] = processor

    def fit(self):
        # 这里是去做一些 processors 的预处理工作，为以后做准备
        for feature_name, processor in self.processors.items():
            if processor.source in self.df.columns:
                processor.fit(self.df[processor.source])

        dump_dict = {
            feature_name: processor.save()  # processor.save() 会把 processor 的内容变成一个字典，然后写入大字典里
            for feature_name, processor in self.processors.items()
        }

        self.save_feature_meta(dump_dict)  # 将这个大字典变成 omegaconf 后写入进文档

    def transform(self):
        # 对于每个特征都进行数据处理，然后写进大的 processed_df 里
        for feature_name, processor in self.processors.items():
            if processor.source in self.df.columns:
                self.processed_df[feature_name] = processor.transform(self.df[processor.source])

    def fit_transform(self):
        self.fit()
        self.transform()

    def load_transform(self):
        self.load_feature_meta()
        self.transform()

    def save_feature_meta(self, dump_dict):
        filepath = Path(self.cfg.output_path) / 'meta_data.yaml'
        filepath.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(dump_dict, filepath)

    def load_feature_meta(self):
        dump_dict = OmegaConf.load(self.cfg.dump_path + '/meta_data.yaml')
        for feature_name, processor in self.processors.items():
            if feature_name in dump_dict:
                processor.load(dump_dict[feature_name])

    def save_data(self):
        if not self.processed_df.empty:
            filepath = Path(self.cfg.output_path) / 'data.parquet'
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.processed_df.to_parquet(filepath)

    def _initiate_samples(self):
        if hasattr(self.cfg, 'samples'):
            for sample in self.cfg.samples:
                aggregator = instantiate(sample)
                self.aggregators[aggregator.group] = aggregator

    def save_samples(self):
        for aggregator in self.aggregators.values():
            aggregator.save_samples(df=self.processed_df, path=self.cfg.sample_path)

        dump_dict = {
            group: aggregator.save()
            for group, aggregator in self.aggregators.items()
        }
        self.save_sample_meta(dump_dict)

    def save_sample_meta(self, dump_dict):
        filepath = Path(self.cfg.sample_path) / 'meta_data.yaml'
        filepath.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(dump_dict, filepath)


@hydra.main(version_base="1.3", config_path="../configs/preprocess", config_name="titanic.yaml")
def main(cfg: DictConfig):
    dataPreprocessor = DataPreprocessor(cfg)
    dataPreprocessor.fit_transform()
    dataPreprocessor.transform()
    dataPreprocessor.save()
    return


if __name__ == "__main__":
    main()
