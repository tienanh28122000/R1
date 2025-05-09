'''
This file is the server for the reward function. It listens for incoming connections from the client and sends the reward to the OpenRLHF environment.
'''

import numpy as np
from transformers import AutoTokenizer

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# reward_type can be linear or sigmoid
def reward_len_min(prompts, completions, answer, tokenizer, alpha=1):
    rewards = []

    question_list = [q[-1]['content'] for q in prompts]
    assert len(list(set(question_list))) == 1 # All element in a batch should have the same question

    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    lens = [len(tokenizer.encode(seq, add_special_tokens=False)) for seq in responses]
    scores = [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    for response, response_len, accuracy in zip(responses, lens, scores):

        if accuracy > 0:
            # 1e-7 is to avoid division by zero
            relative_length = (response_len - np.mean(lens)) / (np.std(lens) + 1e-7) # Reward only when answer is correct.
            reward = accuracy * (1 - alpha * (sigmoid(relative_length)))
        else:
            reward = 0.0

        rewards.append(float(reward)) # score can be something else as well, not just correctness
                                            
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
    prompt = [
        {'role': 'system', 'content': 'abc'},
        {'role': 'user', 'content': 'Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?'}
    ]
    prompts = [prompt for _ in range(len(answers))]

    output = reward_len_min(prompts, completions, answers, tokenizer)
    print(output)