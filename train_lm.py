# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch


# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# # Let's chat for 5 lines
# for step in range(5):
#     # encode the new user input, add the eos_token and return a tensor in Pytorch
#     new_user_input_ids = tokenizer.encode(input(">> User: ") + tokenizer.eos_token, return_tensors='pt')

#     # append the new user input tokens to the chat history
#     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

#     # generated a response while limiting the total chat history to 1000 tokens, 
#     chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#     # pretty print last ouput tokens from bot
#     print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

import numpy as np
import argparse
from typing import Union
import logging
import random
from pathlib import Path
from transformers import AutoTokenizer, Trainer, TrainingArguments
from dataset import ProductDataset
from model import InsurwayRecommender
from sklearn.model_selection import train_test_split

from utils import set_logging

def main(args: argparse.Namespace) -> None:
    set_logging(args.log_file, args.log_level, args.log_stdout)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    train_dataset = ProductDataset(tokenizer, args.data_path, args.force_rebuild)
    if args.mode == "train":
        model = InsurwayRecommender.from_pretrained(
            "roberta-base" 
        )
        train_split, test_split = train_test_split(train_dataset, test_size=0.1)
        train_args = TrainingArguments("test-trainer", num_train_epochs=4, fp16=True, do_eval=True, per_device_train_batch_size=2)
        trainer = Trainer(model, train_args, train_dataset=train_split, eval_dataset=test_split, tokenizer=tokenizer)
        trainer.train()
        model.save_pretrained("trained_models")
    if args.mode == "inference":
        model = InsurwayRecommender.from_pretrained("trained_models")
        user_answers = []
        while True:
            user_input = input("User: ")
            print("got>", user_input, "<")
            if user_input == "end":
                break
            if user_input == "clear":
                user_answers = []
                print("cleared!")
                continue
            user_answers.append(user_input)
            model_input = train_dataset.build_data_point(user_answers)
            _, prediction = model(**model_input)
            print("Model:", "\n\t" + "\n\t".join(train_dataset.get_ranked_products(prediction)))
    

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