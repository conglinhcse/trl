import os
import json
import datetime
import pytz

import torch
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm

class PairwiseDataset(Dataset):
    def __init__(self, points, tokenizer, max_length):
        self.data_points = []
        for point in tqdm(points):
            for pair in point:
                chosen, rejected = pair["chosen"], pair["rejected"]
                chosen_encodings_dict = tokenizer(
                    chosen,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                rejected_encodings_dict = tokenizer(
                    rejected,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                if not torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])).item():
                    continue
                self.data_points.append({
                    "chosen_input_ids": chosen_encodings_dict["input_ids"],
                    "chosen_attention_mask": chosen_encodings_dict["attention_mask"],
                    "rejected_input_ids": rejected_encodings_dict["input_ids"],
                    "rejected_attention_mask": rejected_encodings_dict["attention_mask"]
                })

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        return self.data_points[idx]