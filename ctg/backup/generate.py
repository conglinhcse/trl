import sys
sys.path.append("")

import os
import json

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
from transformers import pipeline, set_seed
import random
import torch
import numpy as np
from tqdm import tqdm

from sft.sft_dataclass import ROCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CKPT_PATH = "ctg/ckpts/sft/sft_ctg/checkpoint-2600"

def generate():

    set_seed(2024)

    # load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_CKPT_PATH)
    # model = GPT2LMHeadModel.from_pretrained(MODEL_CKPT_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_CKPT_PATH)
    model.to(device)
    _ = model.eval()
    print(f'Num of params: {model.num_parameters()}')

    #############
    # Making data
    #############
    # data_path = "/home/leycy016/conglinhle/resource/data/rocstories-data/rm"
    data_path = "/home/leycy016/conglinhle/resource/data/rocstories-data/rl"
    type_data = "train"
    dataset = []
    with open(os.path.join(data_path, f"{type_data}.json"), "r", encoding="utf8") as fp:
        dataset = json.load(fp)

    data_samples = []
    for sample in tqdm(dataset):
        # kw_str = sample["prompt"].split("<|BOS|>")[0].strip()
        # leading_context = sample["leading_context"]
        # prompt = f"TASK: {kw_str}\nSTORY: {leading_context}\nSENTENCE: "
        # chosen = sample["feedback"][0][0]

        local_keywords  = sample["local_keywords"]
        if len(local_keywords) > 2:
            local_keywords = random.sample(local_keywords, 2)
        kw_str = "Write the next sentence of the story containing the following keywords: " + ", ".join(local_keywords) + "."
        leading_context = sample["leading_context"]
        prompt = f"TASK: {kw_str}\nSTORY: {leading_context}\nSENTENCE: "
        chosen = sample["expected_sentence"]
        data_samples.append({
            "prompt": prompt,
            "leading_context": leading_context,
            "kw_str": kw_str,
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
    "min_length": -1, # don't ignore the EOS token (see above)
    "top_k": 0.0, # no top-k sampling
    "top_p": 1.0, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "max_new_tokens": 32, # specify how many tokens you want to generate at most
    }

    results = []
    for id, data_sample in tqdm(enumerate(data_samples), desc="Generating . . . . ."):
        # Nucleaus-search text generation:
        sample_outputs = model.generate(
                                        **tokenizer(data_sample["prompt"], return_tensors='pt').to(device),
                                        max_length=256,
                                        num_return_sequences=2,
                                        pad_token_id=tokenizer.eos_token_id,
                                        **r_kwargs
                                    )
        rejected = [gen_sent.split("\nSENTENCE:")[-1].strip() for gen_sent in tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)]
        new_data_sample = dict(data_sample)
        new_data_sample["rejected"] = rejected
        results.append(new_data_sample)
        
    with open(os.path.join("ctg/data/ranking_data", f"{type_data}_extra.json"), 'w', encoding='utf8') as fp:
        fp.write(json.dumps(results, indent=4))

def generate_examples(prompt_list, model_name='gpt2', max_length=50, num_return_sequences=2, seed=42):
    generator = pipeline('text-generation', model=model_name, device=0)
    set_seed(seed)
    examples = []
    for prompt in prompt_list:
        result = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        example = {'prompt': prompt}
        for i, res in enumerate(result):
            answer = res['generated_text'].lstrip().removeprefix(prompt).strip()
            example[f'answer{i + 1}'] = answer
        examples.append(example)
        print(json.dumps(example, indent=2))
    return examples

if __name__=="__main__":
    # generate()
    pass