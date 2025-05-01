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

# This script is adopted a lot from https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py and https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

import logging
import os
import sys

import datasets
import transformers
from datasets import load_dataset, Dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.trainer_utils import get_last_checkpoint

from rewards import get_reward_funcs
from evals import get_eval_funcs
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config, GRPOConfig, ScriptArguments, get_kbit_device_map, get_quantization_config
from dataclasses import dataclass, field
from typing import Optional
import torch

logger = logging.getLogger(__name__)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["correctness", "int", "strict_format", "soft_format", "xml", "len_cosine", "len_min", "len_l1", "majority_voting", "intrinsic", "group_intrinsic"],
        metadata={
            "help": "List of reward functions."
        },
    )
    eval_funcs: list[str] = field(
        default_factory=lambda: ["extractive_match"],
        metadata={
            "help": "List of eval functions."
        },
    )
    max_items_training: Optional[int] = field(default=None, metadata={"help": "Max samples when training, use for debugging"})
    max_items_eval: Optional[int] = field(default=None, metadata={"help": "Max samples when evaluation, use for debugging"})
    num_tokens: Optional[int] = field(default=None, metadata={"help": "Desired tokens of completions, use for l1 reward function"})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )

def get_tokenizer(model_args: ModelConfig, training_args: GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side='left',
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_model(model_args: ModelConfig, training_args: GRPOConfig) -> AutoModelForCausalLM:
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    ).to("cuda")
    model.config.use_cache = False
    return model

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split="train", max_items: int = None, num_tokens: int = None, system_prompt: str = None) -> Dataset:
    if system_prompt is None:
        system_prompt = "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"

    # Load the dataset split
    data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore

    # Optionally limit the number of examples
    if max_items is not None:
        data = data.select(range(max_items))

    # Function to format each example
    def format_example(x):
        # Initialize prompt with system and user messages
        system_content = system_prompt

        # Modify system prompt if num_tokens is specified
        if num_tokens is not None:
            token_instruction = (
                f"Think for a maximum of {abs(num_tokens)} tokens."
                if num_tokens < 0
                else f"Think for {num_tokens} tokens."
            )
            system_content = f"{system_content}. {token_instruction}"

        prompt = [
            {'role': 'system', 'content': system_content},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ]

        # Build result dictionary
        result = {
            'prompt': prompt,
            'answer': extract_hash_answer(x['answer'])
        }

        # Optionally add num_tokens field
        if num_tokens is not None:
            result['num_tokens'] = num_tokens

        return result

    # Apply formatting to each example
    data = data.map(format_example)  # type: ignore
    return data  # type: ignore

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Load the dataset, currently support openai/gsm8k dataset only
    assert script_args.dataset_name == "openai/gsm8k", print("Currently support gsm8k dataset only!")
    dataset_train = get_gsm8k_questions(max_items=script_args.max_items_training, num_tokens=script_args.num_tokens, system_prompt=script_args.system_prompt)
    dataset_eval = get_gsm8k_questions(split="test", max_items=script_args.max_items_eval, num_tokens=script_args.num_tokens, system_prompt=script_args.system_prompt)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Get eval functions from the registry
    eval_funcs = get_eval_funcs(script_args)

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        eval_funcs=eval_funcs,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset_train)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(output_dir)

    ##########
    # Evaluate
    ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(dataset_eval)
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)