import sys
sys.path.append("")

import os
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed
import random
import torch
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CKPT_PATH = "ctg/ckpts/sft/sft_gpt2_03092024/checkpoint-5400"

def generate():

    set_seed(2024)

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_CKPT_PATH)
    model.to(device)
    _ = model.eval()
    print(f'Num of params: {model.num_parameters()}')

    #############
    # Making data
    #############
    data_path = "/home/leycy016/conglinhle/resource/data/rocstories-data/rl"
    type_data = "train"
    dataset = []
    with open(os.path.join(data_path, f"{type_data}.json"), "r", encoding="utf8") as fp:
        dataset = json.load(fp)[:50000]

    data_samples = []
    for sample in tqdm(dataset):

        # ### Data from rm
        # kw_str = sample["prompt"].split("<|BOS|>")[0].strip().split(":")[-1].strip()
        # instruction = f"Given a leading context of a story, write one single next sentence containing the following keywords: {kw_str}"
        # leading_context = sample["leading_context"]
        # prompt = f"### Instruction: {instruction}\n ### Leading Context: {leading_context}\n ### Next Sentence: "
        # chosen = sample["feedback"][0][0]

        # ### Data from rl
        local_keywords  = sample["local_keywords"]
        if len(local_keywords) > 2:
            local_keywords = random.sample(local_keywords, 2)
        instruction = f"Given a leading context of a story, write one single next sentence containing the following keywords: " + ", ".join(local_keywords) + "."
        leading_context = sample["leading_context"]
        prompt = f"### Instruction: {instruction}\n ### Leading Context: {leading_context}\n ### Next Sentence: "
        chosen = sample["expected_sentence"]


        data_samples.append({
            "prompt": prompt,
            "leading_context": leading_context,
            "kw_str": instruction,
            "chosen": chosen
        })

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
        "early_stopping": True,
        "repetition_penalty": 1.0
    }

    ##########
    # Recommended sampling
    ##########
    r_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
    }

    results = []
    for id, data_sample in tqdm(enumerate(data_samples), desc="Generating . . . . ."):
        sample_outputs = model.generate(
                                        **tokenizer(data_sample["prompt"], return_tensors='pt').to(device),
                                        max_new_tokens=32,
                                        num_return_sequences=2,
                                        pad_token_id=tokenizer.eos_token_id,
                                        **r_kwargs
                                    )
        rejected = [gen_sent.split("### Next Sentence: ")[-1].strip() for gen_sent in tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)]
        new_data_sample = dict(data_sample)
        new_data_sample["rejected"] = rejected
        results.append(new_data_sample)
        
        if id % 200 == 0 or id == len(data_samples) - 1 or id == len(data_samples):
            with open(os.path.join("ctg/data/ranking_data_2", f"{type_data}_extra.json"), 'w', encoding='utf8') as fp:
                fp.write(json.dumps(results, indent=4))


if __name__=="__main__":
    # generate()
    pass