import os
import json

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
import torch
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

def first_true_indices(bools: torch.Tensor, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.

    Args:
        bools (`torch.Tensor`):
            An N-dimensional boolean tensor.
        dtype (`torch.dtype`, optional):
            The desired data type of the output tensor. Defaults to `torch.long`.

    Returns:
        `torch.Tensor`:
            An (N-1)-dimensional tensor of integers indicating the position of the first True
            in each row. If no True value is found in a row, returns the length of the row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values

def get_reward(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the reward logits and the rewards for a given model and query responses.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the reward logits.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for the reward model.
            - `final_rewards` (`torch.Tensor`):
                The final rewards for each query response.
            - `sequence_lengths` (`torch.Tensor`):
                The lengths of the sequences in the query responses.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,
    )
    reward_logits = model.score(output.hidden_states[-1])
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length

    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1),
        sequence_lengths,
    )

def create_comparison_dataset(data_list, eos_token):

    ls_chosen, ls_rejected, ls_context = [], [], []
    for entry in data_list:
        prompt = entry["prompt"].strip()
        chosen = entry["chosen"]
        rejected = [entry["rejected"]] if type(entry["rejected"]) is str else entry["rejected"]
        for res in rejected:
            ls_chosen.append(f"{prompt} {chosen}{eos_token}".strip())
            ls_rejected.append(f"{prompt} {res}{eos_token}".strip())
            ls_context.append(prompt)

    return ls_chosen, ls_rejected, ls_context


if __name__=="__main__":
    reward_model_path = "ctg/ckpts/rm/rm_gpt2-med_06092024/checkpoint-450"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "### Instruction: Given a leading context of a story, write one single next sentence containing the following keywords: parking, lot.\n ### Leading Context: Ken went to a big amusement park. When he went outside he couldn't find his car.\n ### Next Sentence:"
    chosen = "I felt like I was catching up on lost time somehow."
    rejected = "It eventually felt like I lost all the time."

    data_path = "ctg/data/ranking_data"
    eval_data = []
    with open(os.path.join(data_path, "val.json"), "r", encoding="utf8") as fp:
        eval_data = json.load(fp)
    ls_chosen, ls_rejected, ls_context = create_comparison_dataset(eval_data, tokenizer.eos_token)
    print(f'There are {len(ls_chosen) :,} samples for validation.')

    rw_scores = []
    for idx in range(len(ls_chosen)):
        query_responses_ids = tokenizer([ls_chosen[idx], ls_rejected[idx]], max_length=256, padding=True, return_tensors="pt").input_ids
        context_length = len(tokenizer(ls_context[idx]).input_ids)
        _, rw_score, _ = get_reward(reward_model, query_responses_ids, tokenizer.pad_token_id, context_length)
        rw_scores.append(rw_score.reshape(1,2))
    rw_scores = torch.cat(rw_scores)
    acc = (rw_scores[:,0] > rw_scores[:,1]).sum()
    print("Total accuracy: ", acc)


    