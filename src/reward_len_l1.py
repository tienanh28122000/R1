from transformers import AutoTokenizer
import math
import numpy as np

# update dataloader based on https://github.com/cmu-l3/l1/blob/main/scripts/data/deepscaler_dataset.py

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def get_delta_score_max(num_tokens: int, used_tokens: int):
    alpha = 1/500 # coefficient for the positive part
    beta = alpha # coefficient for the negative part

    delta = used_tokens - abs(num_tokens)
    sc = 0
    if delta < 0:
        sc = beta * delta * -1
    else:
        sc = alpha * delta * -1
    # Clip sc to [-1, 1]
    sc = max(-1, min(1, sc))
    # Add delta_constant = 1/2 and normalize to [0,1]
    return (sc + 1)/2

def get_delta_score_exact(num_tokens: int, used_tokens: int):
    # z_score = abs(used_tokens - num_tokens) / (num_tokens/2)
    alpha = 1/3000
    z_score = abs(used_tokens - num_tokens) * alpha
    
    delta_score = 1 - z_score
    return delta_score

def reward_len_l1(completions, answer, tokenizer, num_tokens):
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    scores = [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    gen_lengths = [len(tokenizer.encode(seq, add_special_tokens=False)) for seq in responses]

    for gen_len, score, num_token in zip(gen_lengths, scores, num_tokens):
        if num_token < 0:
            # L1-Max
            delta_score = get_delta_score_max(num_token, float(gen_len))
            reward = max(0, delta_score) * score
            rewards.append(reward)
        else:
            # L1-Exact
            delta_score = get_delta_score_exact(num_token, float(gen_len))
            reward = delta_score if score else delta_score-1
            rewards.append(reward)

    assert len(rewards) == len(answer)
    return rewards

if __name__ == "__main__":
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
    num_tokens = [512 for _ in range(len(answers))]

    output = reward_len_l1(completions, answers, tokenizer, num_tokens)
    print(output)