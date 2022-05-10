import torch
from logging.handlers import DEFAULT_SOAP_LOGGING_PORT
from site import USER_BASE
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
import json
import logging
import tarfile
import pickle

from deep_translator import GoogleTranslator
import requests

from utils import check_paths

cache_dir = Path("./cache")
cache_dataset_file_name = Path("dataset_english.json")

class ProductDataset(Dataset):
    def __init__(self, tokenizer, data_path, force_rebuild=False, translate=True):
        super().__init__()
        self.data_path = Path(data_path)
        check_paths(self.data_path)

        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dataset_path = self.cache_dir / cache_dataset_file_name
        
        self.translator = GoogleTranslator(source='auto', target='en')

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
        for key, value in dataset["products"].items():
            dataset["products"][key] = self.translator.translate(value)
        for userAnswers, recommendations in tqdm(zip(dataset["userAnswers"], dataset["recommendations"]), total=len(dataset["userAnswers"])):
            userAnswers = [self.translator.translate(answer) for answer in userAnswers]
            # print(userAnswers)
            for product_recommendation, product_description in zip(recommendations, dataset["products"].values()):
                model_inputs = tokenizer(product_description, ".".join(userAnswers), padding="max_length", truncation=True)
                model_inputs["labels"] = product_recommendation
                self.dataset["sequences"].append(model_inputs)
                
        self.dataset["products"] = dataset["products"]
        logging.info("Example decoded sequence:")
        print(self.tokenizer.decode(model_inputs["input_ids"]))
        logging.info("Successfully built dataset!")
        self.save_dataset()
    
    def save_dataset(self):
        logging.info(f"Saving dataset to {str(self.cache_dataset_path)}...")
        with self.cache_dataset_path.open("wb") as cache_dataset:
            pickle.dump(self.dataset, cache_dataset)
        logging.info("Successfully saved dataset!")

    def load_cached_dataset(self):
        logging.info(f"loading cached dataset from {str(self.cache_dataset_path)}....")
        with self.cache_dataset_path.open("rb") as cached_dataset:
            self.dataset = pickle.load(cached_dataset)
        logging.info("Successfully loaded cached dataset!")
    
    def __getitem__(self, idx):
        return self.dataset["sequences"][idx]

    def __len__(self):
        return len(self.dataset["sequences"])
    
    def build_data_point(self, userAnswers):
        userAnswers = [self.translator.translate(answer) for answer in userAnswers]
        batched_input = {"input_ids": [], "attention_mask": [], "labels": []}
        for product_description in self.dataset["products"]:
            model_inputs = self.tokenizer(product_description, ".".join(userAnswers), padding="max_length", truncation=True)
            # data_points.append(model_inputs)
            batched_input["input_ids"].append(model_inputs["input_ids"])
            batched_input["attention_mask"].append(model_inputs["attention_mask"])
            batched_input["labels"].append(0)
            
        batched_input["input_ids"] = torch.Tensor(batched_input["input_ids"]).to(int)
        batched_input["attention_mask"] = torch.Tensor(batched_input["attention_mask"]).to(int)
        batched_input["labels"] = torch.Tensor(batched_input["labels"]).to(int)
        return batched_input

    def get_ranked_products(self, scores):
        product_list = list(self.dataset["products"].keys())
        print(scores)
        ranked_product_scores = sorted(list(zip(product_list, scores)), key=lambda a: a[1], reverse=True)
        return [pair[0] + ", " + str(pair[1].item()) for pair in ranked_product_scores]
        # return [pair[0] for pair in ranked_product_scores]


class TabularDataset(Dataset):
    download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tic.tar.gz"
    
    def __init__(self, data_path, force_rebuild=False, translate=True):
        super().__init__()
        self.data_path = Path(data_path)

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dataset_path = self.cache_dir / cache_dataset_file_name
        if not self.data_path.exists():
            self.download_dataset()
            
        if force_rebuild or not self.cache_dataset_path.exists():
            logging.info("Building dataset....")
            self.build_dataset()
        else:
            logging.info(f"loading cached dataset from {str(self.cache_dataset_path.resolve())}")
            self.load_cached_dataset()
    
    def download_dataset(self):
        self.data_path = self.cache_dir / 'original_dataset' / 'file.tar.gz'
        logging.info(f"dataset does not exist under given path! Downloading to {str(self.data_path)}")
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_file = requests.get(self.download_url)
        with self.data_path.open("wb") as dataset_output_file:
            dataset_output_file.write(dataset_file.content)
        logging.info("Decompressing dataset ....")
        with tarfile.open(self.data_path) as tar_file:
            tar_file.extractall(self.data_path.parent)


    def build_dataset(self, tokenizer):
        with self.data_path.open() as dataset_file:
            dataset = json.load(dataset_file)
            
        self.save_dataset()
    
    def save_dataset(self):
        logging.info(f"Saving dataset to {str(self.cache_dataset_path)}...")
        with self.cache_dataset_path.open("wb") as cache_dataset:
            pickle.dump(self.dataset, cache_dataset)
        logging.info("Successfully saved dataset!")

    def load_cached_dataset(self):
        logging.info(f"loading cached dataset from {str(self.cache_dataset_path)}....")
        with self.cache_dataset_path.open("rb") as cached_dataset:
            self.dataset = pickle.load(cached_dataset)
        logging.info("Successfully loaded cached dataset!")
    
    def __getitem__(self, idx):
        return self.dataset["sequences"][idx]

    def __len__(self):
        return len(self.dataset["sequences"])
    
    def build_data_point(self, userAnswers):
        userAnswers = [self.translator.translate(answer) for answer in userAnswers]
        batched_input = {"input_ids": [], "attention_mask": [], "labels": []}
        for product_description in self.dataset["products"]:
            model_inputs = self.tokenizer(product_description, ".".join(userAnswers), padding="max_length", truncation=True)
            # data_points.append(model_inputs)
            batched_input["input_ids"].append(model_inputs["input_ids"])
            batched_input["attention_mask"].append(model_inputs["attention_mask"])
            batched_input["labels"].append(0)
            
        batched_input["input_ids"] = torch.Tensor(batched_input["input_ids"]).to(int)
        batched_input["attention_mask"] = torch.Tensor(batched_input["attention_mask"]).to(int)
        batched_input["labels"] = torch.Tensor(batched_input["labels"]).to(int)
        return batched_input

    def get_ranked_products(self, scores):
        product_list = list(self.dataset["products"].keys())
        print(scores)
        ranked_product_scores = sorted(list(zip(product_list, scores)), key=lambda a: a[1], reverse=True)
        return [pair[0] + ", " + str(pair[1].item()) for pair in ranked_product_scores]
        # return [pair[0] for pair in ranked_product_scores]
