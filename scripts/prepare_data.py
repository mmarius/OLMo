"""
Script for preparing the Tulu V2 data for fine-tuning an OLMo model.
"""

import logging
import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from rich.progress import track

from olmo.tokenizer import Tokenizer
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)

#  DKYoon/SlimPajama-6B

DATASETS = {
    "fineweb": {"name": "HuggingFaceFW/fineweb", "split": "train"},
    "slimpajama-validation": {"name": "DKYoon/SlimPajama-6B", "split": "validation"},
    "slimpajama-train": {"name": "DKYoon/SlimPajama-6B", "split": "train"},
    "slimpajama-test": {"name": "DKYoon/SlimPajama-6B", "split": "test"},
}

DATASET_COLUMNS_TO_REMOVE = {
    "fineweb": ["id", "dump", "url", "date", "file_path", "language", "language_score", "token_count"],
    "slimpajama-validation": ["meta", "__index_level_0__"],
    "slimpajama-train": ["meta", "__index_level_0__"],
    "slimpajama-test": ["meta", "__index_level_0__"],
}


def load_dataset(dataset_name: str) -> ds.Dataset:
    hf_name, split = (
        DATASETS[dataset_name]["name"],
        DATASETS[dataset_name]["split"],
    )
    log.info(f"Loading dataset {hf_name} split {split}...")
    # dataset = ds.load_dataset(hf_name, split=split, data_files="sample/10BT/000_00000.parquet")
    dataset = ds.load_dataset(hf_name, split=split)

    return dataset


def main(opts) -> None:
    tokenizer: Tokenizer
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)

    # load the dataset
    dataset = load_dataset(opts.dataset_name)

    # # subsample the dataset for debugging purposes
    # dataset = dataset.select(range(100))

    log.info("Tokenizing dataset...")
    dataset = dataset.map(
        partial(preprocess_lm, tokenizer=tokenizer, max_seq_len=opts.seq_len, text_key="text"),
        batched=True,
        remove_columns=DATASET_COLUMNS_TO_REMOVE[opts.dataset_name],
        num_proc=opts.num_proc,  # type: ignore
    )

    log.info("Filtering dataset...")
    n = len(dataset)  # type: ignore
    dataset = dataset.filter(filter, batched=False, num_proc=opts.num_proc)  # type: ignore
    log.info(f"Filtered out {n - len(dataset):,d} examples")

    log.info("Counting tokens...")
    total_tokens = 0
    for ex in track(dataset):
        assert len(ex["input_ids"]) == opts.seq_len  # type: ignore
        total_tokens += len(ex["input_ids"])  # type: ignore
    log.info(f"Total tokens: {total_tokens:,d}")

    dataset_name = f"{DATASETS[opts.dataset_name]['name'].replace('/', '_')}_{DATASETS[opts.dataset_name]['split']}_{opts.seq_len}"
    output_dir = os.path.join(opts.output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Saving results to '{output_dir}'...")

    # create a memmap for the input ids and label mask
    input_ids_file = np.memmap(
        os.path.join(output_dir, "input_ids.npy"), dtype=np.uint32, mode="w+", shape=(total_tokens,)
    )
    label_mask_file = np.memmap(
        os.path.join(output_dir, "label_mask.npy"), dtype=np.bool_, mode="w+", shape=(total_tokens,)
    )
    offset = 0
    for ex in track(dataset):
        ex_len = len(ex["input_ids"])  # type: ignore
        input_ids_file[offset : offset + ex_len] = ex["input_ids"]  # type: ignore
        label_mask_file[offset : offset + ex_len] = ex["label_mask"]  # type: ignore
        offset += ex_len
    input_ids_file.flush()
    label_mask_file.flush()

    log.info("Done!")


def filter(example):
    return example["n_labels"] > 0


def preprocess_lm(examples, tokenizer: Tokenizer, max_seq_len: int, text_key: str = "text"):
    """Preprocesses text examples for language modeling by tokenizing and preparing labels.

    This function processes a batch of text examples by:
    1. Tokenizing each text example
    2. Adding EOS tokens at the start and end
    3. Creating label masks where True indicates tokens to predict
    4. Padding/truncating sequences to max_seq_len

    Args:
        examples: A list of dictionaries containing batched examples with text under the text_key
        tokenizer: The Tokenizer instance to use for encoding text
        max_seq_len: Maximum sequence length for truncation/padding
        text_key: The key in examples dict containing the text data (default: "text")

    Returns:
        dict: A dictionary containing:
            - input_ids: List of token ID sequences, each padded to max_seq_len
            - label_mask: List of boolean masks indicating which tokens should be predicted
            - n_labels: List of integers indicating number of tokens to predict per sequence

    Note:
        - EOS tokens are added at the start and end of each sequence
        - Label masks are False for EOS and padding tokens
        - Sequences longer than max_seq_len are truncated
        - Sequences shorter than max_seq_len are padded with pad_token_id
    """
    input_ids = []
    label_mask = []

    # concatenate all samples in the dataset
    for msg in examples[text_key]:
        # we sorround the text with eos tokens
        ids = tokenizer.encode(msg.strip() + tokenizer.eos_token, add_special_tokens=False)
        ids = [tokenizer.eos_token_id] + ids + [tokenizer.eos_token_id]
        labels = [False] + ([True] * len(ids))
        # mask out the eos token
        assert ids[-1] == tokenizer.eos_token_id
        labels[-1] = False

        # truncate the input ids and label mask
        ids = ids[:max_seq_len]
        labels = labels[:max_seq_len]

        # pad the input ids and label mask to the max sequence length
        ids = ids + [tokenizer.pad_token_id] * (max_seq_len - len(ids))
        labels = labels + [False] * (max_seq_len - len(labels))

        # collect the ids and labels for each sequence in the batch
        assert len(ids) == len(labels)
        input_ids.append(ids)
        label_mask.append(labels)

    assert len(input_ids) == len(label_mask)

    # Calculate n_labels for each sequence
    n_labels = [sum(mask) for mask in label_mask]

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare a huggingface dataset for use with an OLMo model")
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=DATASETS.keys(),
        help="""Name of the dataset to prepare.""",
    )
    parser.add_argument("--output_dir", type=str, help="""Directory to save the results to.""")
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="""Tokenizer path or identifier.""",
        default="allenai/eleuther-ai-gpt-neox-20b-pii-special",
    )
    parser.add_argument("-s", "--seq-len", type=int, help="""Max sequence length.""", default=2048)
    parser.add_argument("--eos", type=int, help="""EOS token ID.""", default=100257)  # using OLMo2 defaults here
    parser.add_argument("--pad", type=int, help="""PAD token ID.""", default=100277)  # using OLMo2 defaults here
    parser.add_argument("-j", "--num-proc", type=int, help="""Number of workers.""", default=8)
    return parser


if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    main(opts)
