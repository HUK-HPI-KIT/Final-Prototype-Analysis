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
    
    def compute_recommendation(self, user_information, recommendation_vote):
        # TODO: maybe already put this in tree traversal for speedup
        # TODO: implement rigid rules for certain products
        for idx, value in enumerate(user_information):
            feature_name = self.feature_names[idx]
            if feature_name == "num_cars" and self.name == "car_insurance" and value == 0:
                return 0.0
            # TODO: implement more rules. mooooooooooooooore ruuuuuuuuules.
        
        return float(recommendation_vote[1] / (recommendation_vote[0] + recommendation_vote[1]))
    
    def process_information(self, user_information):
        # TODO: maybe cache estimator states (where they stopped last time) for speedup
        feature_vote = np.zeros((self.n_features_in_,), dtype=int)
        recommendation_vote = np.array([1, 0], dtype=int)
        for estimator in self.estimators_:
            tree = estimator.tree_
            
            def traverse(node):
                if tree.feature[node] != _tree.TREE_UNDEFINED:  # is split node
                    feature_idx = tree.feature[node]
                    if user_information[feature_idx] == UNDEFINED_USER_INFORMATION:
                        feature_vote[feature_idx] += 1
                        # if we reach a split where we don't have enough information we 
                        # return the number of samples that did have the insurance and the number of samples that didn't
                        return np.array(tree.value[node], dtype=int).squeeze()
                    threshold = tree.threshold[node]
                    if user_information[feature_idx] <= threshold:
                        return traverse(tree.children_left[node])
                    else:
                        return traverse(tree.children_right[node])
                else:  # is leaf node
                    return np.array(tree.value[node], dtype=int).squeeze()
            recommendation_vote += traverse(0)
        most_important_feature = self.feature_names[np.argmax(self.feature_importances_)]
        return feature_vote, self.compute_recommendation(user_information, recommendation_vote), most_important_feature

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
            logging.info(f"Storing model at {str(self.model_store_path)}..")
            joblib.dump(self.estimators, self.model_store_path)
            logging.info("Model successfully saved!")
        else:
            logging.info(f"Loading random forest classifiers from {str(self.model_store_path)}...")
            self.estimators = joblib.load(self.model_store_path)
            logging.info("Model successfully loaded!")
    
    def infer(self, user_information):
        recommendation = {}
        feature_vote = np.zeros((len(self.estimators[0].feature_names),), dtype=int)
        significant_features = []
        for estimator in self.estimators:
            estimator_feature_vote, product_score, most_important_feature = estimator.process_information(user_information)
            feature_vote += estimator_feature_vote
            recommendation[estimator.name] = product_score
            significant_features.append(most_important_feature)
        
        return recommendation, feature_vote, significant_features
