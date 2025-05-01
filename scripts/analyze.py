from datasets import load_dataset
import os
import re

def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text)
    if match:
        return match.group(1)
    else:
        return None

details_path = "/data1/speech/anhnmt2/RL/scripts/evals/Qwen2.5-1.5B-Instruct/details/Qwen2.5-1.5B-Instruct/2025-04-29T20-25-16.098059/details_lighteval|gsm8k_test|0_2025-04-29T20-25-16.098059.parquet"

# Load the details
details = load_dataset("parquet", data_files=details_path, split="train")

all_answers = []
all_predictions = []
for detail in details:
    # print(detail)
    # print(detail.keys())
    print(detail["full_prompt"])
    # print(detail["gold"])
    print(detail["predictions"])
    # print(detail["specifics"])

    gt = detail["specifics"]["extracted_golds"][0]
    try:
        preds = detail["specifics"]["extracted_predictions"][0]
    except:
        preds = extract_answer(detail["predictions"][0])
        print(detail["predictions"])
        print(f'filtered preds: {preds}')

    all_answers.append(gt)
    all_predictions.append(preds)
    print("------------")

# for idx, (pred, ans) in enumerate(zip(all_predictions, all_answers)):
#     if pred != ans:
#         print(f'pred: {pred}')
#         print(f'gt: {ans}')
#         print("-----------------")

# Calculate accuracy
accuracy = sum(p == a for p, a in zip(all_predictions, all_answers)) / len(all_answers)
print(f"Accuracy: {accuracy:.4f}")