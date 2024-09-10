import sys
sys.path.append("")

import logging
import multiprocessing
import os
import json
import random
from contextlib import nullcontext
from tqdm import tqdm

from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser
from trl.env_utils import strtobool

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

def preprocess_pairwise(data_points):
    data_samples = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    for sample in tqdm(data_points):
        prompt = sample["prompt"]
        chosen_sent = sample["chosen"].strip()
        ls_rejected = [sample["rejected"]] if type(sample["rejected"]) == str else sample["rejected"]
        for i in range(len(ls_rejected)):
            data_samples["prompt"].append(prompt)
            data_samples["chosen"].append(chosen_sent)
            data_samples["rejected"].append(ls_rejected[i])
    return data_samples

if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

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
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    # data_path = "/home/leycy016/conglinhle/resource/data/rocstories-data/rm"
    data_path = "ctg/data/ranking_data"
    train_data, eval_data = [], []
    with open(os.path.join(data_path, "train.json"), "r", encoding="utf8") as fp:
        train_data = json.load(fp)
    with open(os.path.join(data_path, "train_extra.json"), "r", encoding="utf8") as fp:
        train_data.extend(json.load(fp))
    with open(os.path.join(data_path, "val.json"), "r", encoding="utf8") as fp:
        eval_data = json.load(fp)

    train_dataset = preprocess_pairwise(data_points=train_data)
    eval_dataset = preprocess_pairwise(data_points=eval_data)

    from datasets import Dataset
    train_dataset = Dataset.from_dict({"prompt": train_dataset["prompt"], "chosen": train_dataset["chosen"], "rejected": train_dataset["rejected"]})
    eval_dataset = Dataset.from_dict({"prompt": eval_dataset["prompt"], "chosen": eval_dataset["chosen"], "rejected": eval_dataset["rejected"]})

    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
