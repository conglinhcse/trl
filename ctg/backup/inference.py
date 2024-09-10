import sys
sys.path.append("")

import os
import json
import logging

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
import random
import torch
import numpy as np
from tqdm import tqdm

from sft.sft_dataclass import ROCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CKPT_PATH = "ctg/ckpts/rl/rl_dpo/checkpoint-1900"
MODEL_OUTPUT_PATH = "/home/leycy016/conglinhle/STSEval/trl/dpo"

# MODEL_CKPT_PATH = "ctg/ckpts/sft/sft_ctg/checkpoint-2600"
# MODEL_OUTPUT_PATH = "/home/leycy016/conglinhle/STSEval/trl/sft"

# MODEL_CKPT_PATH = "ctg/ckpts/rl/rl_dpo_degen/checkpoint-300"
# MODEL_OUTPUT_PATH = "/home/leycy016/conglinhle/STSEval/trl/dpo_degen"

def generate():

    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

    # load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_CKPT_PATH)
    # model = GPT2LMHeadModel.from_pretrained(MODEL_CKPT_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_CKPT_PATH)
    model.to(device)
    _ = model.eval()
    print(f'Num of params: {model.num_parameters()}')

    test_data = []
    with open(os.path.join("/home/leycy016/conglinhle/resource/data/rocstories-data/sft", "test.json"), "r", encoding="utf8") as fp:
        test_data = json.load(fp)
    random.shuffle(test_data)
    test_dataset = ROCDataset(data=test_data, tokenizer=tokenizer, device=device, num_lkw=2, num_gkw=0, generate_mode=True)
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
        "early_stopping": True,
        "repetition_penalty": 1.0
    }

    results = {}
    for test_id, test_sample in tqdm(enumerate(test_dataset), desc="Generating . . . . ."):
        # Beam-search text generation:
        sample_outputs = model.generate(
                                        test_sample["input_ids"].to(device),
                                        max_new_tokens=50,
                                        num_return_sequences=1,
                                        pad_token_id=tokenizer.eos_token_id,
                                        **b_kwargs
                                    )
        
        gen_sent = tokenizer.decode(sample_outputs[0][test_sample["beginning_mark"]:], skip_special_tokens=True).strip()
        
        results[test_id] = dict(test_sample)
        results[test_id]['generated_sentence'] = gen_sent
        results[test_id].pop('input_ids')
        
        if test_id % 10 == 0 or test_id == len(test_dataset) - 1:
            with open(os.path.join(MODEL_OUTPUT_PATH, "results-14.json"), 'w', encoding='utf8') as fp:
                fp.write(json.dumps(results, indent=4))


if __name__=="__main__":
    # generate()
    pass