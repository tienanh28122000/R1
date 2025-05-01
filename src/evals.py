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

"""Evaluation functions for GRPO training."""

from lighteval.tasks.requests import Doc
from lighteval.metrics.dynamic_metrics import multilingual_extractive_match_metric
from typing import Callable, Dict, Optional

def gsm8k(line, task_name: str = None):
    # Has special analysis in metric for number decomposition
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {line['answer']}"],
        gold_index=0,
    )

def extractive_match(prompts, completions, answer, **kwargs) -> list[float]:
    metric = multilingual_extractive_match_metric()
    scores = []
    responses = [completion[0]['content'] for completion in completions]
    questions = [prompt[-1]['content'] for prompt in prompts]
    # extracted_responses = [extract_xml_answer(r) for r in responses]
    for idx, (ques, pred, ans) in enumerate(zip(questions, responses, answer)):
        line = {
            "question": ques,
            "answer": ans
        }
        formatted_doc = gsm8k(line)
        golds = [ans]
        predictions = [pred]
        score = metric.sample_level_fn(golds, predictions, formatted_doc)
        scores.append(score)
    return scores

def get_eval_funcs(script_args) -> list[Callable]:
    EVAL_FUNCS_REGISTRY = {
        "extractive_match": extractive_match,
    }
    eval_funcs = [EVAL_FUNCS_REGISTRY[func] for func in script_args.eval_funcs]

    return eval_funcs