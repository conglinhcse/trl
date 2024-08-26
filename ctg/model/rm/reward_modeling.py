import sys
sys.path.append("")

CUDA_LAUNCH_BLOCKING = '1'

import warnings

import torch
from accelerate import PartialState
from datasets import load_dataset
import datasets
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config


tqdm.pandas()


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

    train_dataset = load_dataset("Anthropic/hh-rlhf", split=[
        datasets.ReadInstruction('train', to=10, unit='%')])[0]
    eval_dataset = load_dataset("Anthropic/hh-rlhf", split=[
        datasets.ReadInstruction('test', to=10, unit='%')])[0]

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
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= config.max_length
            and len(x["input_ids_rejected"]) <= config.max_length,
            num_proc=config.dataset_num_proc,
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=config.dataset_num_proc,
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= config.max_length
            and len(x["input_ids_rejected"]) <= config.max_length,
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