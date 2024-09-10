import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from torch.utils.data import Dataset

class ROCDataset(Dataset):

    def __init__(self, data, tokenizer, num_lkw=1, num_gkw=1, device='cpu', generate_mode=False):

        ls_leading_contexts, ls_expected_sentences, ls_local_keywords, ls_global_keywords, ls_length_lcs = [], [], [], [], []
        for entry in data:
            ls_leading_contexts.append(entry["leading_context"])
            ls_expected_sentences.append(entry["expected_sentence"])
            ls_local_keywords .append(entry["local_keywords"])
            ls_global_keywords.append(entry["global_keywords"])
            ls_length_lcs.append(entry["length_lc"])

        self.tokenizer = tokenizer
        self.ls_leading_contexts = ls_leading_contexts
        self.ls_expected_sentences = ls_expected_sentences
        self.ls_local_keywords  = ls_local_keywords 
        self.ls_global_keywords = ls_global_keywords
        self.ls_length_lcs = ls_length_lcs
        self.num_lkw = num_lkw
        self.num_gkw = num_gkw
        self.device = device
        self.generate_mode = generate_mode
    #---------------------------------------------#

    @staticmethod
    def select_keywords(local_keywords, global_keywords, num_lkw=1, num_gkw=1):

        selected_local_keywords = []
        if len(local_keywords) >= num_lkw:
            selected_local_keywords = random.sample(local_keywords, num_lkw)
        elif num_lkw > 0:
            selected_local_keywords = local_keywords
        assert len(selected_local_keywords) <= num_lkw


        selected_global_keywords = []
        if len(global_keywords) >= num_gkw:
            selected_global_keywords = random.sample(global_keywords, num_gkw)
        elif num_gkw > 0:
            selected_global_keywords = global_keywords
        assert len(selected_global_keywords) <= num_gkw
            
        return selected_local_keywords + selected_global_keywords
    #---------------------------------------------#

    def __len__(self):
        return len(self.ls_leading_contexts)
    #---------------------------------------------#

    def __getitem__(self, i):

        selected_keywords = self.select_keywords(self.ls_local_keywords[i].copy(), self.ls_global_keywords[i].copy(), self.num_lkw, self.num_gkw)
        
        if self.generate_mode:
            
            prompt =  "Write the next sentence of the story containing the following keywords: " + ", ".join(selected_keywords) + "." + \
                    " " + self.tokenizer.bos_token + " " + self.ls_leading_contexts[i] + \
                    " " + self.tokenizer.sep_token

            
            encodings_dict = self.tokenizer(prompt, return_tensors='pt').to(self.device)

            return {
                "input_ids": encodings_dict.input_ids,
                "prompt": prompt,
                "keyword_sentence": "Write the next sentence of the story containing the following keywords: " + ", ".join(selected_keywords) + ".",
                "leading_context": self.ls_leading_contexts[i],
                "expected_sentence": self.ls_expected_sentences[i],
                "beginning_mark": encodings_dict.input_ids.size()[-1],
                "length_lc": self.ls_length_lcs[i]
            }
        
        else:

            prompt =  "Write the next sentence of the story containing the following keywords: " + ", ".join(selected_keywords) + "." + \
                    " " + self.tokenizer.bos_token + " " + self.ls_leading_contexts[i] + \
                    " " + self.tokenizer.sep_token + " " + self.ls_expected_sentences[i] +\
                    " " + self.tokenizer.eos_token

            encodings_dict = self.tokenizer(prompt,
                                    truncation=True,
                                    max_length=256,
                                    padding="max_length",
                                    return_tensors="pt",
                                )

            input_ids = encodings_dict['input_ids']
            attention_mask = encodings_dict['attention_mask']
            labels = input_ids.clone()

            return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
            }