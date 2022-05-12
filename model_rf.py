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
    leaf_vote_lookup = []
    
    def build_leaf_lookup(self):
        # for getting all leaf nodes of each subtree in order to determine
        # final recommendation if not enough user information is provided.
        # this way we don't have to go one random path and our recommendation is more robust
        for estimator in self.estimators_:
            tree = estimator.tree_
            self.leaf_vote_lookup.append(np.zeros((len(tree.node_count), 2), dtype=int))
            
            def traverse(node):
                if tree.feature[node] != _tree.TREE_UNDEFINED:
                    entry = traverse(tree.children_left[node])
                    entry += traverse(tree.children_right[node])
                else:
                    value = tree.value[node]
                    if value == 0:
                        entry = np.array([value, 0], dtype=int)
                    else:
                        entry = np.array([0, value], dtype=int)
                self.leaf_vote_lookup[-1][node] = entry
                return entry

            traverse(0)
    
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
        for idx, estimator in enumerate(self.estimators_):
            tree = estimator.tree_
            
            def traverse(node):
                if tree.feature[node] != _tree.TREE_UNDEFINED:  # is split node
                    feature_idx = tree.feature[node]
                    if user_information[feature_idx] == UNDEFINED_USER_INFORMATION:
                        feature_vote[feature_idx] += 1
                        # if we reach a split where we don't have enough information we 
                        # return the number of leafs below this node that vote for 0 and the number of leafs voting 1
                        return self.leaf_vote_lookup[idx][node]
                    threshold = tree.threshold[node]
                    if user_information[feature_idx] <= threshold:
                        traverse(tree.children_left[node])
                    else:
                        traverse(tree.children_right[node])
                else:  # is leaf node
                    return self.leaf_vote_lookup[idx][node]
                    
            recommendation_vote += traverse(0)
        return feature_vote, self.compute_recommendation(recommendation_vote)

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
        for estimator in self.estimators:
            estimator_feature_vote, product_score = estimator.process_information(user_information)
            feature_vote += estimator_feature_vote
            recommendation[estimator.name] = product_score
        
        return recommendation, feature_vote