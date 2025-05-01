#!/usr/bin/env python3
# Copyright    2025  VinBigdata Corp.        (authors: anhnmt2)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import re
import torch
from collections import Counter
from typing import Callable, Dict, Optional

# import len-based reward function
from reward_len_cosine import CosineScaledSparseReward
from reward_len_min import reward_len_min
from reward_len_l1 import reward_len_l1

# import proposed reward function
from reward_intrinsic import DynamicRewardCalculator
import pyarrow as pa # Import pyarrow
import lancedb
import os
from reward_group_intrinsic import DiversityRewardCalculator

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    # pattern = r"<reasoning>[\s\S]*</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def len_cosine_reward_func(completions, answer, tokenizer, **kwargs) -> list[float]:
    reward_model = CosineScaledSparseReward(
        min_value_wrong=-1.0,
        max_value_wrong=0.0,
        min_value_correct=0.0,
        max_value_correct=1.0,
        max_len=512,  # maximum length e.g. generate_kwargs["max_new_tokens"]
        exceed_length=0.0,  # big penalty if too long (Optional)
        repetition_max_penalty=0.0,  # penalty for repeating ngrams
        repetition_ngram_size=20,  # n-gram size to detect repetition
    )

    rewards = reward_model.reward(
        completions,
        answer,
        tokenizer,
    )

    return rewards

def len_min_reward_func(prompts, completions, answer, tokenizer, **kwargs) -> list[float]:
    rewards = reward_len_min(prompts, completions, answer, tokenizer)

    return rewards

def len_l1_reward_func(completions, answer, tokenizer, num_tokens, **kwargs) -> list[float]:
    rewards = reward_len_l1(completions, answer, tokenizer, num_tokens)

    return rewards

def majority_voting_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]

    # Extract answers from each output
    extracted_responses = [extract_xml_answer(r) for r in responses]

    # Find the majority answer
    counts = Counter(extracted_responses)
    majority_response, _ = counts.most_common(1)[0]

    # Assign rewards: 1 if matches majority, else 0
    return [1.0 if ans == majority_response else 0.0 for ans in extracted_responses]

def intrinsic_reward_func(completions, answer, **kwargs) -> list[float]:
   # --- Setup LanceDB ---
    db_path = "./lancedb"
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)

    # --- Create/open vector table for successful examples ---
    # Note: Using a fixed dimension based on the embedding model specified in the class
    embedding_dims = 1536 # Should match the default in the class
    vector_field = pa.list_(pa.float32(), list_size=embedding_dims)
    schema = pa.schema([
        pa.field("vector", vector_field),
        pa.field("response", pa.string()), # Storing the full response text
        pa.field("timestamp", pa.timestamp('ms')) # To track age for removal (optional in this simplified version)
    ])
    # Use mode="overwrite" for simplicity in example, creates table if it doesn't exist
    success_table = db.create_table("success_simplified", schema=schema, exist_ok=True)

    reward_model = DynamicRewardCalculator(
        success_table=success_table,
        diversity_scale=3.0,
    )
    rewards = reward_model.calculate_rewards(completions, answer)

    return rewards

def reward_group_intrinsic_func(completions, answer, **kwargs) -> list[float]:
    diversity_reward_calculator = DiversityRewardCalculator(
        base_correctness_reward=1.0,
        diversity_scale=3.0 # Adjust this to control diversity incentive strength
    )
    rewards = diversity_reward_calculator.calculate_rewards(completions, answer)
    return rewards

def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "correctness": correctness_reward_func,
        "int": int_reward_func,
        "strict_format": strict_format_reward_func,
        "soft_format": soft_format_reward_func,
        "xml": xmlcount_reward_func,
        "len_cosine": len_cosine_reward_func,
        "len_min": len_min_reward_func,
        "len_l1": len_l1_reward_func,
        "majority_voting": majority_voting_reward_func,
        "intrinsic": intrinsic_reward_func,
        "group_intrinsic": reward_group_intrinsic_func
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs