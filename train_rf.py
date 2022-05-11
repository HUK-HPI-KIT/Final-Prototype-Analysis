import json
import numpy as np
import argparse
from typing import Union
import logging
import random
from pathlib import Path
from transformers import AutoTokenizer, Trainer, TrainingArguments
from dataset import ProductDataset, TabularDataset
from model_rf import ProductRecommender
from sklearn.model_selection import train_test_split

from utils import set_logging

def main(args: argparse.Namespace) -> None:
    set_logging(args.log_file, args.log_level, args.log_stdout)
    
    train_dataset = TabularDataset(args.question_path, args.data_path, args.force_rebuild)
    recommender = ProductRecommender(train_dataset, args.force_train)
    
    user_data = json.loads(args.user_data)
    user_data_array = train_dataset.create_user_data_array(user_data)
    recommendation, feature_request_vote = recommender.infer(user_data_array)
    next_question = train_dataset.get_next_question(feature_request_vote)
    inference_result = {
        "recommendation": recommendation,
        "fv": feature_request_vote.tolist(),
        "next_question": next_question,
    }
    print(json.dumps(inference_result, indent=4))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("HUK HPI KIT final prototype analysis tool")
    
    parser.add_argument("--log-level", type=str, default="info",
                     choices=["debug", "info", "warning", "error", "critical"],
                     help="log level for logging message output")
    parser.add_argument("--log-file", type=str, default="log.log",
                     help="output file path for logging. default to stdout")
    parser.add_argument("--log-stdout", action="store_true", default=False,
                     help="toggles force logging to stdout. if a log file is specified, logging will be "
                     "printed to both the log file and stdout")
    parser.add_argument("--data-path", type=str, required=False, default=None,
                     help="path to dataset json")
    parser.add_argument("--question-path", type=str, required=False, default=None,
                     help="path to question json")
    parser.add_argument("--user-data", type=str, required=True,
                     help="json encoded user data")
    parser.add_argument("--force-rebuild", action="store_true", default=False,
                     help="forces rebuild of dataset")
    parser.add_argument("--force-train", action="store_true", default=False,
                     help="forces training of classifier")
    args = parser.parse_args()
    main(args)