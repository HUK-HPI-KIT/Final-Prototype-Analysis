from pathlib import Path
from torch.utils.data import Dataset
import json
import logging

from utils import check_paths

cache_dir = Path("./cache")
cache_dataset_file_name = Path("dataset_english.json")

class ProductDataset(Dataset):
    def __init__(self, data_path, force_rebuild=False):
        super().__init__()
        self.data_path = Path(data_path)
        check_paths(self.data_path)

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dataset_path = self.cache_dir / cache_dataset_file_name

        if force_rebuild or not self.cache_dataset_path.exists():
            logging.info("Building dataset....")
            self.build_dataset()
        else:
            logging.info(f"loading cached dataset from {str(self.cache_dataset_path.resolve())}")
            self.load_cached_dataset()

    def build_dataset(self):
        with self.data_path.open() as dataset_file:
            dataset = json.load(dataset_file)

    def load_cached_dataset(self):
        with self.cache_dataset_path.open() as cached_dataset:
            self.dataset = json.load(cached_dataset)