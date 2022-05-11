
import numpy as np
import argparse
from typing import Union
import logging
import random
from pathlib import Path
from transformers import AutoTokenizer, Trainer, TrainingArguments
from dataset import ProductDataset, TabularDataset
from model_lm import InsurwayRecommender
from sklearn.model_selection import train_test_split

from utils import set_logging

def main(args: argparse.Namespace) -> None:
    set_logging(args.log_file, args.log_level, args.log_stdout)
    
    train_dataset = TabularDataset(args.data_path, args.force_rebuild)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("HUK HPI KIT final prototype analysis tool")
    
    parser.add_argument("--log-level", type=str, default="info",
                     choices=["debug", "info", "warning", "error", "critical"],
                     help="log level for logging message output")
    parser.add_argument("--log-file", type=str, default=None,
                     help="output file path for logging. default to stdout")
    parser.add_argument("--log-stdout", action="store_true", default=True,
                     help="toggles force logging to stdout. if a log file is specified, logging will be "
                     "printed to both the log file and stdout")
    parser.add_argument("--data-path", type=str, required=True,
                     help="path to dataset json")
    parser.add_argument("--mode", type=str, required=True,
                     help="mode")
    parser.add_argument("--force-rebuild", action="store_true", default=False,
                     help="forces rebuild of dataset")

    args = parser.parse_args()
    main(args)