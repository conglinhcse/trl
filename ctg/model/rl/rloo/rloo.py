import sys, os
sys.path.append("")
import shutil
import json
import random

from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from datasets import Dataset

from trl import ModelConfig
from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

def prep_data(data_list):
    ls_instructions, ls_leading_contexts = [], []
    for entry in data_list:
        ls_leading_contexts.append(entry["leading_context"])
        local_keywords = entry["local_keywords"]
        if len(local_keywords) >= 2:
            local_keywords = random.sample(local_keywords, 2)
        ls_instructions.append("Given a leading context of a story, write one single next sentence containing the following keywords: " + ", ".join(local_keywords) + ".")

    return Dataset.from_dict({
        "instruction": ls_instructions,
        "leading_context": ls_leading_contexts
    })


if __name__ == "__main__":
    parser = HfArgumentParser((RLOOConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(config.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path
    )
    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path
    )
    ################
    # Dataset
    ################
    def formatting_prompts_func(dataset, tokenizer):
        def tokenize(element):
            prompt = f"### Instruction: {element["instruction"]}\n ### Leading Context: {element["leading_context"]}\n ### Next Sentence:"
            outputs = tokenizer(
                prompt,
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}
        
        return dataset.map(
            tokenize,
            batched=False,
            remove_columns=dataset.column_names,
            num_proc=config.dataset_num_proc,
        )
    
    # data_path = "/home/leycy016/conglinhle/resource/data/rocstories-data/sft"
    train_data, eval_data = [], []
    with open(os.path.join("/home/leycy016/conglinhle/resource/data/rocstories-data/rl", "train.json"), "r", encoding="utf8") as fp:
        train_data = json.load(fp)
    with open(os.path.join("/home/leycy016/conglinhle/resource/data/rocstories-data/sft", "val.json"), "r", encoding="utf8") as fp:
        eval_data = json.load(fp)

    random.shuffle(train_data) 
    random.shuffle(eval_data)

    train_dataset = prep_data(data_list=train_data)
    eval_dataset = prep_data(data_list=eval_data)

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = formatting_prompts_func(train_dataset, tokenizer)
        eval_dataset = formatting_prompts_func(eval_dataset, tokenizer)

    print(f'There are {len(train_dataset) :,} and {len(eval_dataset) :,} samples for training, validation.')

    ################
    # Training
    ################
    trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()
