import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
import json
import pandas as pd
import logging
import tarfile
import pickle

from deep_translator import GoogleTranslator
import requests
from model_rf import UNDEFINED_USER_INFORMATION

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
    
    def __init__(self, question_path, data_path=None, force_rebuild=False, translate=True):
        super().__init__()

        self.question_path = Path(question_path)
        check_paths(self.question_path)
        with self.question_path.open() as question_file:
            self.questions = json.load(question_file)
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dataset_path = self.cache_dir / cache_dataset_file_name
        if data_path is None:
            self.data_path = self.data_path = self.cache_dir / 'original_dataset' / 'ticdata2000.txt'
        else:
            self.data_path = Path(data_path)

        if self.data_path is not None and not self.data_path.exists():
            self.download_dataset()
            
        if force_rebuild or not self.cache_dataset_path.exists():
            logging.info("Building dataset....")
            self.build_dataset()
        else:
            logging.info(f"loading cached dataset from {str(self.cache_dataset_path.resolve())}")
            self.load_cached_dataset()
    
    def download_dataset(self):
        download_path = self.cache_dir / 'original_dataset' / 'file.tar.gz'
        logging.info(f"dataset {str(self.data_path)} does not exist under given path! Downloading to {str(self.data_path)}")
        download_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_file = requests.get(self.download_url)
        with download_path.open("wb") as dataset_output_file:
            dataset_output_file.write(dataset_file.content)
        logging.info("Decompressing dataset ....")
        with tarfile.open(download_path) as tar_file:
            tar_file.extractall(download_path.parent)
        self.data_path = self.cache_dir / 'original_dataset' / 'ticdata2000.txt'


    def build_dataset(self):
        with self.data_path.open() as dataset_file:
            dataset = pd.read_csv(dataset_file, sep="\t")
        dict_path = self.data_path.parent / "dictionary.txt"
        with dict_path.open("r", encoding="windows-1252") as dict_file:
            lines = dict_file.readlines()
            entries = [entry.split(" ")[:2] for entry in lines[3:89]]
            column_df = pd.DataFrame(entries)
        column_names = column_df.loc[:, 1]
        dataset.columns = column_names
        
        self.dataset = pd.DataFrame()
        self.dataset["households"] = dataset["MAANTHUI"]
        self.dataset["household_size"] = dataset["MGEMOMV"]
        self.dataset["age"] = dataset["MGEMLEEF"]
        self.dataset["living_situation"] = dataset[["MRELGE", "MRELSA", "MRELSA", "MFALLEEN"]].values.argmax(axis=1)
        self.dataset["children"] = dataset[["MFGEKIND", "MFWEKIND"]].values.argmax(axis=1)
        self.dataset["education"] = dataset[["MOPLHOOG", "MOPLMIDD", "MOPLLAAG"]].values.argmax(axis=1)
        self.dataset["job"] = dataset[["MBERHOOG", "MBERZELF", "MBERBOER", "MBERMIDD", "MBERARBG", "MBERARBO"]].values.argmax(axis=1)
        self.dataset["liquidity"] = dataset[["MSKA", "MSKB1", "MSKB2", "MSKC", "MSKD"]].values.argmax(axis=1)
        self.dataset["house_rented"] = dataset[["MHHUUR", "MHKOOP"]].values.argmax(axis=1)
        self.dataset["num_cars"] = dataset[["MAUT0", "MAUT1", "MAUT2"]].values.argmax(axis=1)
        self.dataset["health_insurance"] = dataset[["MZFONDS", "MZPART"]].values.argmax(axis=1)
        self.dataset["income"] = dataset[["MINKM30", "MINK3045", "MINK4575", "MINK7512", "MINK123M"]].values.argmax(axis=1)
        
        self.products = pd.DataFrame()
        self.products["car_insurance"] = (dataset["APERSAUT"] > 0).astype(int)
        self.products["private_accident_insurance"] = (dataset["APERSONG"] > 0).astype(int)
        self.products["disability_insurance"] = (dataset["AWAOREG"] > 0).astype(int)
        self.products["life_insurance"] = (dataset["ALEVEN"] > 0).astype(int)
        self.products["property_insurance"] = (dataset["AINBOED"] > 0).astype(int)
        
        
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
    
    def product_names(self):
        return [str(column) for column in self.products.columns]
    
    def __getitem__(self, idx):
        return self.dataset.iloc[idx]

    def __len__(self):
        return len(self.dataset)
    
    def create_user_data_array(self, user_data):
        user_data_array = np.zeros((len(self.dataset.columns),), dtype=int)
        for idx, column in enumerate(self.dataset.columns):
            column_name = str(column)
            if column_name in user_data:
                user_data_array[idx] = user_data[column_name]
            else:
                user_data_array[idx] = UNDEFINED_USER_INFORMATION
        return user_data_array

    def user_feature_names(self):
        return [str(column) for column in self.dataset.columns]

    def get_next_question(self, feature_vote):
        feature_to_ask = feature_vote.argmax()
        next_question = self.questions[self.user_feature_names()[feature_to_ask]]
        next_question["questionId"] = self.user_feature_names()[feature_to_ask]
        
        return next_question
    
    def get_explanations(self, significant_features, user_information):
        explanations = []
        for significant_feature in significant_features:
            if len(significant_feature) > 0:
                feature_name = self.questions[significant_feature]["name"]
                answer = self.questions[significant_feature]["answers"][user_information[self.user_feature_names().index(significant_feature)]]
                explanations.append(f"weil du auf die Frage nach {feature_name} mit '{answer}' geantwortet hast.")
            else:
                explanations.append("")
        return explanations