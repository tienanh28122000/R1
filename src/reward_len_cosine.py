import math
from typing import Callable, List, Optional
import torch
from transformers import AutoTokenizer

# We add this to the gen len before checking if it is above max len to ensure there are no
# off-by-one mistakes.
MAX_LEN_MARGIN = 16

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def get_repetition_penalty(ngram_size: int, max_penalty: float, generation: str) -> float:
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if max_penalty == 0:
        return 0

    ngrams = set()
    total = 0
    for ng in zipngram(generation, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    return scaling * max_penalty

class CosineScaledSparseReward:
    def __init__(
            self,
            min_value_wrong: float,
            max_value_wrong: float,
            min_value_correct: float,
            max_value_correct: float,
            max_len: int,
            exceed_length: Optional[float],
            repetition_max_penalty: float,
            repetition_ngram_size: int):
        self.min_value_wrong = min_value_wrong
        self.min_value_correct = min_value_correct
        self.max_value_wrong = max_value_wrong
        self.max_value_correct = max_value_correct
        self.max_len = max_len
        self.exceed_length = exceed_length if exceed_length is not None else 0
        self.repetition_max_penalty = repetition_max_penalty
        self.repetition_ngram_size = repetition_ngram_size

        if min_value_wrong > 0 or max_value_wrong > 0:
            raise ValueError("Wrong values should not be positive")

        # print(
        #     "Initialized math rule cosine scaled reward with"
        #     f" min_value_wrong: {min_value_wrong}, max_value_wrong: {max_value_wrong},"
        #     f" min_value_correct: {min_value_correct}, max_value_correct: {max_value_correct},"
        #     f" max_len: {max_len}, exceed_length: {exceed_length}, MAX_LEN_MARGIN: {MAX_LEN_MARGIN},"
        #     f" repetition_max_penalty: {repetition_max_penalty}, repetition_ngram_size: {repetition_ngram_size}")

    def reward(
            self,
            completions: List[str],
            answer: List[str],
            tokenizer) -> List[float]:
        """Calculate correct/wrong rewards based solution length using a cosine schedule.

        The general idea is:
        - Shorter correct solutions should be rewarded over longer ones.
        - Longer wrong solutions should be rewarded over shorter ones.
        - Shorter solutions should be more risk averse (wrong penalized more than correct rewarded).
        """

        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        scores = [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
        gen_lengths = [len(tokenizer.encode(seq, add_special_tokens=False)) for seq in responses]

        rewards = []
        for gen, gen_len, score in zip(responses, gen_lengths, scores):
            if gen_len + MAX_LEN_MARGIN >= self.max_len:
                print(f"Exceed length penalty applied -- gen_len: {gen_len}, max_len: {self.max_len}")
                rewards.append(self.exceed_length)
                continue

            if score == 1:
                min_value = self.min_value_correct
                max_value = self.max_value_correct
                rep_penalty = 0
            else:
                # Yes, they are swapped. This is required for the cosine formula below
                # to work with negative numbers.
                min_value = self.max_value_wrong
                max_value = self.min_value_wrong

                rep_penalty = get_repetition_penalty(
                    ngram_size=self.repetition_ngram_size,
                    max_penalty=self.repetition_max_penalty,
                    generation=gen)

            progress = gen_len / self.max_len
            cosine = math.cos(progress * math.pi)
            r = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            r += rep_penalty

            rewards.append(r)

        return rewards

def main():
    model_path = "/data1/speech/anhnmt2/RL/pretrained_llms/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # or any model you like

    # ===== 1. Prepare example inputs =====
    completions = [
        "<reasoning>\n Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. \n</reasoning>\n<answer>\n10\n</answer>",
        "<reasoning>\n Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. \n</reasoning>\n<answer>\n12\n</answer>",
        "<reasoning>\n Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. \n</reasoning>\n<answer>\n14\n</answer>",
        "<reasoning>\n Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. \n</reasoning>\n<answer>\n16\n</answer>",
        "<reasoning>\n Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. \n</reasoning>\n<answer>\n12\n</answer>"
    ]
    completions = [[{'role': 'assistant', 'content': item}] for item in completions]
    answers = [
        "12",
        "12",
        "12",
        "12",
        "12"
    ]

    # ===== 2. Initialize CosineScaledSparseReward =====
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

    # ===== 3. Run reward calculation =====
    rewards = reward_model.reward(
        completions,
        answers,
        tokenizer,
    )

    # ===== 4. Print results =====
    print(rewards)

if __name__ == "__main__":
    main()


