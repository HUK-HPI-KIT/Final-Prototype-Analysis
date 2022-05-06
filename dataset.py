from pathlib import Path
from torch.utils.data import Dataset
import json
import logging
import pickle

from utils import check_paths

cache_dir = Path("./cache")
cache_dataset_file_name = Path("dataset_english.json")

class ProductDataset(Dataset):
    def __init__(self, tokenizer, data_path, force_rebuild=False):
        super().__init__()
        self.data_path = Path(data_path)
        check_paths(self.data_path)

        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dataset_path = self.cache_dir / cache_dataset_file_name

        if force_rebuild or not self.cache_dataset_path.exists():
            logging.info("Building dataset....")
            self.build_dataset(tokenizer)
        else:
            logging.info(f"loading cached dataset from {str(self.cache_dataset_path.resolve())}")
            self.load_cached_dataset()

    def build_dataset(self, tokenizer):
        with self.data_path.open() as dataset_file:
            dataset = json.load(dataset_file)
        self.dataset = {}
        self.dataset["sequences"] = []
        for recommendation, userAnswers in zip(dataset["userAnswers"], dataset["recommendations"]):
            model_inputs = tokenizer((list(dataset["products"].values()) + userAnswers), padding="max_length")
            model_inputs["labels"] = recommendation
            self.dataset["sequences"].append([model_inputs, recommendation])
        self.dataset["products"] = dataset["products"]
        self.save_dataset()
    
    def save_dataset(self):
        with self.cache_dataset_path.open("w") as cache_dataset:
            pickle.dump(self.dataset, cache_dataset)

    def load_cached_dataset(self):
        with self.cache_dataset_path.open() as cached_dataset:
            self.dataset = json.load(cached_dataset)
    
    def __getitem__(self, idx):
        return self.dataset["sequences"][idx]

    def __len__(self):
        return len(self.dataset["sequences"])
    
    def build_data_point(self, userAnswers):
        return self.tokenizer((list(self.dataset["products"].values()) + userAnswers), padding="max_length")

    def get_ranked_products(self, scores):
        product_list = list(self.dataset["products"].keys())
        ranked_product_scores = sorted(list(zip(product_list, scores)), lambda a, b: a[1] > b[1])
        return [pair[0] for pair in ranked_product_scores]
    