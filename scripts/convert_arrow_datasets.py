"""
Script for converting arrow datasets to jsonl.gz files.
Created with the help of Claude 3.5 Sonnet.
"""

import argparse
import glob
import json
import os
import gzip
import re
from datasets import load_dataset
from multiprocessing import Pool, Manager
from functools import partial
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def convert_arrow_to_jsonl(arrow_file, output_dir, processed_files):
    """
    Convert a single Arrow file to compressed JSONL format (gzip).

    Args:
        arrow_file (str): Path to the Arrow file
        output_dir (str): Directory to save the compressed JSONL file
        processed_files (list): Shared list to track processed files
    """
    try:
        if arrow_file in processed_files:
            return

        processed_files.append(arrow_file)
        dataset = load_dataset("arrow", data_files={"train": arrow_file})["train"]

        # add an id column to the dataset
        new_column = [str(i) for i in range(len(dataset))]
        dataset = dataset.add_column("id", new_column)

        base_name = os.path.basename(arrow_file)
        output_file = os.path.join(output_dir, base_name.replace(".arrow", ".jsonl.gz"))

        logging.info(f"Converting {arrow_file} to {output_file}")
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            for item in dataset:
                json_str = json.dumps(item)
                f.write(json_str + "\n")

        logging.info(f"Successfully converted {arrow_file}")

    except Exception as e:
        logging.error(f"Error processing {arrow_file}: {str(e)}")
        if arrow_file in processed_files:
            processed_files.remove(arrow_file)
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass


def filter_arrow_files(files, pattern=None):
    """
    Filter Arrow files based on regex pattern.

    Args:
        files (list): List of file paths
        pattern (str): Regex pattern that must match in filename

    Returns:
        list: Filtered list of file paths
    """
    if not pattern:
        return files

    try:
        regex = re.compile(pattern)
        return [f for f in files if regex.search(f)]
    except re.error as e:
        logging.error(f"Invalid regex pattern: {e}")
        return []


def convert_directory(input_dir, output_dir, pattern=None, num_workers=4):
    """
    Convert filtered Arrow files to compressed JSONL format using parallel processing.

    Args:
        input_dir (str): Directory containing Arrow files
        output_dir (str): Directory to save compressed JSONL files
        pattern (str): Regex pattern that must match in filename
        num_workers (int): Number of parallel workers
    """
    os.makedirs(output_dir, exist_ok=True)

    arrow_files = glob.glob(os.path.join(input_dir, "*.arrow"))

    if not arrow_files:
        logging.warning(f"No Arrow files found in {input_dir}")
        return

    filtered_files = filter_arrow_files(arrow_files, pattern)

    if not filtered_files:
        logging.warning(f"No files matched the pattern: {pattern}")
        return

    logging.info(f"Found {len(filtered_files)} files matching the pattern")

    with Manager() as manager:
        processed_files = manager.list()
        convert_func = partial(convert_arrow_to_jsonl, output_dir=output_dir, processed_files=processed_files)

        with Pool(num_workers) as pool:
            pool.map(convert_func, filtered_files)

        logging.info(f"Completed processing {len(processed_files)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    # Convert files matching the pattern
    convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        num_workers=args.num_workers,
    )
