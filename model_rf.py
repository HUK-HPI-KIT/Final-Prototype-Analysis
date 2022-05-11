import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from pathlib import Path
import joblib
import logging


UNDEFINED_USER_INFORMATION = -1

class ProductForest(RandomForestClassifier):
    name = ""
    feature_names = []
    
    def compute_recommendation(self, user_information):
        # TODO: maybe already put this in tree traversal for speedup
        # TODO: implement rigid rules for certain products
        for idx, value in enumerate(user_information):
            feature_name = self.feature_names[idx]
            if feature_name == "num_cars" and self.name == "car_insurance" and value == 0:
                return 0.0
            # TODO: implement more rules. mooooooooooooooore ruuuuuuuuules.
        
        return float(self.predict_proba(np.expand_dims(user_information, axis=0))[0][1])
    
    def process_information(self, user_information):
        print(user_information)
        # TODO: maybe cache estimator states (where they stopped last time) for speedup
        feature_vote = np.zeros((self.n_features_in_,), dtype=int)
        for estimator in self.estimators_:
            tree = estimator.tree_
            
            def traverse(node, depth):
                if tree.feature[node] != _tree.TREE_UNDEFINED:
                    feature_idx = tree.feature[node]
                    if user_information[feature_idx] == UNDEFINED_USER_INFORMATION:
                        feature_vote[feature_idx] += 1
                        logging.debug(f"found undefined feature at depth {depth}")
                        return
                    threshold = tree.threshold[node]
                    if user_information[feature_idx] <= threshold:
                        traverse(tree.children_left[node], depth + 1)
                    else:
                        traverse(tree.children_right[node], depth + 1)

            traverse(0, 1)
        return feature_vote, self.compute_recommendation(user_information)

class ProductRecommender():
    cache_dir = "cache/model"
    
    def __init__(self, dataset, train=False):
        self.model_store_path = Path(self.cache_dir) / "rf_model.joblib"
        self.model_store_path.parent.mkdir(parents=True, exist_ok=True)
        
        if train or not self.model_store_path.exists():
            logging.info("Training random forest classifiers for insurance products...")
            self.estimators = []
            feature_names = [str(column) for column in dataset.dataset.columns]
            for product in dataset.product_names():
                logging.info(f"Training for {product}...")
                self.estimators.append(ProductForest(n_jobs=-1, max_features=1))
                self.estimators[-1].fit(dataset, dataset.products[product])
                self.estimators[-1].name = product
                self.estimators[-1].feature_names = feature_names
            self.n_features = len(feature_names)
            logging.info(f"Storing model at {str(self.model_store_path)}..")
            joblib.dump(self, self.model_store_path)
            logging.info("Model successfully saved!")
        else:
            logging.info(f"Loading random forest classifiers from {str(self.model_store_path)}...")
            self.estimators = joblib.load(self.model_store_path)
            logging.info("Model successfully loaded!")
    
    def infer(self, user_information):
        recommendation = {}
        feature_vote = np.zeros((self.n_features,), dtype=int)
        for estimator in self.estimators:
            estimator_feature_vote, product_score = estimator.process_information(user_information)
            feature_vote += estimator_feature_vote
            recommendation[estimator.name] = product_score
        
        return recommendation, feature_vote