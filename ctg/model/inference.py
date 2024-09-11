import sys
sys.path.append("")

import os
import json
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_CKPT_PATH = "ctg/ckpts/sft/sft_gpt2_03092024/checkpoint-5400"
# MODEL_OUTPUT_PATH = "/home/leycy016/conglinhle/STSEval/trl/sft/gpt2-sml"

# MODEL_CKPT_PATH = "ctg/ckpts/rl/rl_rloo/rloo_gpt2-med_10092024/checkpoint-395"
# MODEL_OUTPUT_PATH = "/home/leycy016/conglinhle/STSEval/trl/rloo"

############
# PEFT
############
MODEL_CKPT_PATH = "gpt2-xl" # Path or Hugging Face model ID of the base model
LORA_MODEL_PATH = "ctg/ckpts/sft/sft_gpt2-xl_11092024/checkpoint-100" # Path where your fine-tuned LoRA model is saved
MODEL_OUTPUT_PATH = "/home/leycy016/conglinhle/STSEval/trl/sft/gpt2-xl-peft"


def prep_data(data_list, tokenizer):
    dataset = []
    for entry in data_list:
        leading_context = entry["leading_context"]
        expected_sentence = entry["expected_sentence"]
        length_lc = entry["length_lc"]
        local_keywords = entry["local_keywords"]
        if len(local_keywords) >= 2:
            local_keywords = random.sample(local_keywords, 2)
        instruction = "Given a leading context of a story, write one single next sentence containing the following keywords: " + ", ".join(local_keywords) + "."
        prompt = f"### Instruction: {instruction}\n ### Leading Context: {leading_context}\n ### Next Sentence:"
        encodings_dict = tokenizer(prompt, return_tensors='pt')
        dataset.append({
            "encodings_dict": encodings_dict,
            "prompt": prompt,
            "keyword_sentence":instruction,
            "leading_context": leading_context,
            "expected_sentence": expected_sentence,
            "beginning_mark": encodings_dict.input_ids.size()[-1],
            "length_lc": length_lc
        })

    return dataset


def generate(is_peft=False):

    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_CKPT_PATH)
    if is_peft:
        # Load the LoRA fine-tuned model
        model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
    model.to(device)
    _ = model.eval()
    print(f'Num of params: {model.num_parameters()}')

    test_data = []
    with open(os.path.join("/home/leycy016/conglinhle/resource/data/rocstories-data/sft", "test.json"), "r", encoding="utf8") as fp:
        test_data = json.load(fp)
    test_dataset = prep_data(data_list=test_data, tokenizer=tokenizer)
    print(f'There are {len(test_dataset) :,} samples for testing.')

    ##########
    # Greedy-decoding sampling (default)
    ##########
    g_kwargs = {}


    ##########
    # Nucleus sampling
    ##########
    n_kwargs = {
        "do_sample": True,
        "top_k": 40,
        "top_p": 0.9,
        "temperature": 0.7,
        "repetition_penalty": 1.0
    }


    ##########
    # Beam-search sampling
    ##########
    b_kwargs = {
        "do_sample": False,
        "num_beams": 3,
        "repetition_penalty": 1.0
    }

    ##########
    # Recommended sampling
    ##########
    r_kwargs = {
        "min_length": -1,
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True,
        "tokenizer": tokenizer,
    }

    results = {}
    for test_id, test_sample in tqdm(enumerate(test_dataset), desc="Generating . . . . ."):
        with torch.no_grad():
            sample_outputs = model.generate(
                                            **test_sample["encodings_dict"].to(device),
                                            max_new_tokens=32,
                                            num_return_sequences=1,
                                            pad_token_id=tokenizer.eos_token_id,
                                            **r_kwargs
                                        )
            
            gen_sent = tokenizer.decode(sample_outputs[0][test_sample["beginning_mark"]:]).split(tokenizer.eos_token)[0].strip()
            gen_sent = gen_sent.split(". ")[0].strip() + "."
            # gen_sent = tokenizer.decode(sample_outputs[0])
            
            results[test_id] = dict(test_sample)
            results[test_id]['generated_sentence'] = gen_sent
            results[test_id].pop('encodings_dict')
            
            if test_id % 10 == 0 or test_id == len(test_dataset) - 1:
                with open(os.path.join(MODEL_OUTPUT_PATH, "results.json"), 'w', encoding='utf8') as fp:
                    fp.write(json.dumps(results, indent=4))


def generate_story():

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_CKPT_PATH)
    model.to(device)
    _ = model.eval()
    print(f'Num of params: {model.num_parameters()}')

    def make_prompt(ls_current_sents, ls_kw):
        return f"### Instruction: Given a leading context of a story, write one single next sentence containing the following keywords: {' ,'.join(ls_kw)}.\n ### Leading Context: {' '.join(ls_current_sents)}\n ### Next Sentence: "

    original_story = [
        ["A man was walking his dog down the street.", "The dog seemed to be having trouble walking on the leash.", "As time went on the man walked his dog everyda, y.", "Over time the man didn't have to use a leash, the dog followed.", "Now the man is walking with his dog and a new puppy on a leash."],
        ["Joe needed more papers for his printer. So Joe drove to the store to get more papers. Joe found lots of paper for a good deal. Joe took the paper home. Joe put the paper in the printer."]
    ]

    kws = [
        [
            ["dog", "leash"],
            ["walked", "time"],
            ["followed", "use"],
            ["puppy", "leash"]
        ],
        [
            ["papers", "drove"],
            ["deal", "found"],
            ["home", "took"],
            ["joe", "paper"]
        ]
    ]

    ##########
    # Beam-search sampling
    ##########
    b_kwargs = {
        "do_sample": False,
        "num_beams": 3,
        "repetition_penalty": 1.0
    }

    for story_id in tqdm(range(len(original_story)), desc="Generating . . . . ."):
        generated_story = [original_story[story_id][0]]
        for sentence_id in range(4):
            prompt = make_prompt(generated_story, kws[story_id][sentence_id])
            sample_outputs = model.generate(
                                            tokenizer(prompt, return_tensors="pt")["input_ids"].to(device),
                                            max_new_tokens=32,
                                            num_return_sequences=1,
                                            pad_token_id=tokenizer.eos_token_id,
                                            **b_kwargs
                                        )
            
            gen_sent = tokenizer.decode(sample_outputs[0]).split("### Next Sentence:")[1].split(tokenizer.eos_token)[0].strip()
            generated_story.append(gen_sent)

        print(generated_story)


if __name__=="__main__":
    generate(is_peft=True)
    # generate_story()
    pass