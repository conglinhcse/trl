import sys
sys.path.append("")

import logging
import os
from contextlib import nullcontext
import json
import random

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from trl.env_utils import strtobool
from datasets import Dataset

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
    DataCollatorForCompletionOnlyLM
)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prep_data(data_list):
    ls_instructions, ls_leading_contexts, ls_expected_sentences = [], [], []
    for entry in data_list:
        ls_leading_contexts.append(entry["leading_context"])
        ls_expected_sentences.append(entry["expected_sentence"])
        local_keywords = entry["local_keywords"]
        if len(local_keywords) >= 2:
            local_keywords = random.sample(local_keywords, 2)
        ls_instructions.append("Given a leading context of a story, write one single next sentence containing the following keywords: " + ", ".join(local_keywords) + ".")

    return Dataset.from_dict({
        "instruction": ls_instructions,
        "leading_context": ls_leading_contexts,
        "expected_sentence": ls_expected_sentences
    })


def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        leading_context = examples["leading_context"][i]
        expected_sentence = examples["expected_sentence"][i]
        text = f"### Instruction: {instruction}\n ### Leading Context: {leading_context}\n ### Next Sentence: {expected_sentence}{tokenizer.eos_token}"
        output_texts.append(text)

    return output_texts


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
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        force_download=True,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

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

    train_dataset = prep_data(data_list=train_data)
    eval_dataset = prep_data(data_list=eval_data)

    print(f'There are {len(train_dataset) :,} and {len(eval_dataset) :,} samples for training, validation.')

    response_template = " ### Next Sentence:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

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
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
