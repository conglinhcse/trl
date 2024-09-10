import sys
sys.path.append("")

CUDA_LAUNCH_BLOCKING = '1'

import warnings
import os
import json
import random

import torch
from accelerate import PartialState
from datasets import load_dataset
import datasets
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from datasets import Dataset

from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config


tqdm.pandas()

def create_comparison_dataset(data_list, eos_token):

    ls_chosen, ls_rejected = [], []
    for entry in data_list:
        prompt = entry["prompt"]
        chosen = entry["chosen"]
        rejected = [entry["rejected"]] if type(entry["rejected"]) is str else entry["rejected"]
        for res in rejected:
            ls_chosen.append(f"{prompt}{chosen}{eos_token}".strip())
            ls_rejected.append(f"{prompt}{res}{eos_token}".strip())

    return Dataset.from_dict({
        "chosen": ls_chosen,
        "rejected": ls_rejected
    })


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    model.config.pad_token_id = tokenizer(tokenizer.pad_token)["input_ids"][0]

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    ################
    # Dataset
    ################
    
    data_path = "ctg/data/ranking_data"
    train_data, eval_data = [], []
    with open(os.path.join(data_path, "train.json"), "r", encoding="utf8") as fp:
        train_data = json.load(fp)
    with open(os.path.join(data_path, "val.json"), "r", encoding="utf8") as fp:
        eval_data = json.load(fp)
    with open(os.path.join(data_path, "train_extra.json"), "r", encoding="utf8") as fp:
        train_data.extend(json.load(fp))

    random.shuffle(train_data)
    random.shuffle(eval_data)

    train_dataset = create_comparison_dataset(train_data, tokenizer.eos_token)
    eval_dataset = create_comparison_dataset(eval_data, tokenizer.eos_token)

    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets
    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=config.dataset_num_proc,
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=config.dataset_num_proc,
        )

    
    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    trainer.push_to_hub()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)