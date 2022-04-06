import torch
import numpy as np
import argparse
from typing import Union
import logging
import random
from pathlib import Path


def set_logging(log_file: Union[None, str], log_level: str, output_stdout: bool) -> None:
    """configures logging module.

    Args:
        log_file (str): path to log file. if omitted, logging will be forced to stdout.
        log_level (str): string name of log level (e.g. 'debug')
        output_stdout (bool): toggles stdout output. will be activated automatically if no log file was given.
            otherwise if activated, logging will be outputed both to stdout and log file.
    """
    logger = logging.getLogger()

    log_level_name = log_level.upper()
    log_level = getattr(logging, log_level_name)
    logger.setLevel(log_level)

    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s [%(filename)s : %(funcName)s() : l. %(lineno)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging_format)
        logger.addHandler(file_handler)
    else:
        output_stdout = True

    if output_stdout:
        stream = logging.StreamHandler()
        stream.setLevel(log_level)
        stream.setFormatter(logging_format)
        logger.addHandler(stream)

def main(args: argparse.Namespace) -> None:
    set_logging(args.log_file, args.log_level, args.log_stdout)
    logging.info("----------------------------------------------------------")
    logging.info("Python script started!")
    logging.info("loading conciousness....")
    logging.info("torch version:", torch.__version__)
    logging.info("numpy version:", np.__version__)

    msg = f"You need {random.randint(0, 20)} more insurances ... now!"

    logging.info(f"printing message >{msg}<...")
    print(msg)
    logging.info("done!")

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
    args = parser.parse_args()
    main(args)