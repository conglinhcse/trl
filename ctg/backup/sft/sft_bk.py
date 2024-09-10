import sys
sys.path.append("")

import logging
import os
from contextlib import nullcontext
import json
import random

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from trl.env_utils import strtobool

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

from sft_dataclass import ROCDataset

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model init kwargs & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    SPECIAL_TOKENS  = { "bos_token": "<|BOS|>", # beginning of story
                        "eos_token": "<|EOS|>", # end of story
                        "unk_token": "<|UNK|>",
                        "pad_token": "<|PAD|>",
                        "sep_token": "<|SEP|>"}
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    print("Special tokens added")

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # force_download=True,
    )
    training_args.model_init_kwargs = model_kwargs
    

    ################
    # Dataset
    ################
    data_path = "/home/leycy016/conglinhle/resource/data/rocstories-data/sft"
    train_data, eval_data = [], []
    with open(os.path.join(data_path, "train.json"), "r", encoding="utf8") as fp:
        train_data = json.load(fp)
    with open(os.path.join(data_path, "val.json"), "r", encoding="utf8") as fp:
        eval_data = json.load(fp)

    random.shuffle(train_data) 
    random.shuffle(eval_data)

    train_dataset = ROCDataset(data=train_data, tokenizer=tokenizer, num_lkw=2, num_gkw=0, device=device)
    eval_dataset = ROCDataset(data=eval_data, tokenizer=tokenizer, num_lkw=2, num_gkw=0, device=device)

    print(f'There are {len(train_dataset) :,} and {len(eval_dataset) :,} samples for training, validation.')

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )
    trainer.model.resize_token_embeddings(len(tokenizer))

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
