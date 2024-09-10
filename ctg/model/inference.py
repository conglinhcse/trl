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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CKPT_PATH = "/home/leycy016/conglinhle/resource/ckpts/sft/simctg/gpt2_medium/22082024/training_step_9900_train_mle_loss_1.69_train_cl_loss_0.003_dev_ppl_5.524"
MODEL_OUTPUT_PATH = "/home/leycy016/conglinhle/STSEval/trl/ppo"

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
        prompt = f"### Instruction: {instruction}\n ### Leading Context: {leading_context}\n ### Next Sentence: "
        encodings_dict = tokenizer(prompt, return_tensors='pt').to(device)
        dataset.append({
            "input_ids": encodings_dict.input_ids,
            "prompt": prompt,
            "keyword_sentence":instruction,
            "leading_context": leading_context,
            "expected_sentence": expected_sentence,
            "beginning_mark": encodings_dict.input_ids.size()[-1],
            "length_lc": length_lc
        })

    return dataset

def generate():

    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_CKPT_PATH)
    model.to(device)
    _ = model.eval()
    print(f'Num of params: {model.num_parameters()}')

    test_data = []
    with open(os.path.join("/home/leycy016/conglinhle/resource/data/rocstories-data/sft", "test.json"), "r", encoding="utf8") as fp:
        test_data = json.load(fp)
    random.shuffle(test_data)
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
        "stop_strings": tokenizer.eos_token,
    }

    results = {}
    for test_id, test_sample in tqdm(enumerate(test_dataset), desc="Generating . . . . ."):
        sample_outputs = model.generate(
                                        test_sample["input_ids"].to(device),
                                        max_new_tokens=32,
                                        num_return_sequences=1,
                                        pad_token_id=tokenizer.eos_token_id,
                                        **b_kwargs
                                    )
        
        # gen_sent = tokenizer.decode(sample_outputs[0][test_sample["beginning_mark"]:], skip_special_tokens=True).strip()
        gen_sent = tokenizer.decode(sample_outputs[0][test_sample["beginning_mark"]:]).split(tokenizer.eos_token)[0].strip()

        # import pdb
        # pdb.set_trace()

        #### Adding
        # gen_sent = gen_sent.split(". ")[0].strip() + "."
        
        results[test_id] = dict(test_sample)
        results[test_id]['generated_sentence'] = gen_sent
        results[test_id].pop('input_ids')
        
        if test_id % 10 == 0 or test_id == len(test_dataset) - 1:
            with open(os.path.join(MODEL_OUTPUT_PATH, "results.json"), 'w', encoding='utf8') as fp:
                fp.write(json.dumps(results, indent=4))


if __name__=="__main__":
    generate()