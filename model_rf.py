import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from pathlib import Path
import joblib
import logging

from dataset import TabularDataset

class ProductForest(RandomForestClassifier):
    name = ""
    
    def process_information(self, user_information):
        ...

class ProductRecommender():
    cache_dir = "cache/model"
    
    def __init__(self, dataset: TabularDataset, train=False):
        self.model_store_path = Path(self.cache_dir) / "rf_model.joblib"
        self.model_store_path.parent.mkdir(parents=True, exist_ok=True)
        
        if train or not self.model_store_path.exists():
            logging.info("Training random forest classifiers for insurance products...")
            self.estimators = {}
            for product in dataset.product_names():
                logging.info(f"Training for {product}...")
                self.estimators[product] = ProductForest(n_jobs=-1, max_features=1)
                self.estimators[product].fit(dataset, dataset.products[product])
                self.estimators[product].name = product
            logging.info(f"Storing model at {str(self.model_store_path)}..")
            joblib.dump(self.estimators, self.model_store_path)
            logging.info("Model successfully saved!")
        else:
            logging.info(f"Loading random forest classifiers from {str(self.model_store_path)}...")
            self.estimators = joblib.load(self.model_store_path)
            logging.info("Model successfully loaded!")
    
    def infer(self, user_information):
        recommendation = {}
        feature_vote = np.array((self.estimators[0].n_features,), dtype=int)
        for estimator in self.estimators:
            product_score, estimator_feature_vote = estimator.process_information(user_information)
            feature_vote += estimator_feature_vote
            recommendation[estimator.name] = product_score
        
        return recommendation, feature_vote