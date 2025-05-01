from lighteval.tasks.requests import Doc
from lighteval.metrics.dynamic_metrics import multilingual_extractive_match_metric

def gsm8k(line, task_name: str = None):
    # Has special analysis in metric for number decomposition
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {line['answer']}"],
        gold_index=0,
    )

# Step 0: Import necessary functions
metric = multilingual_extractive_match_metric()

# Step 1: Prepare your line (one sample)
line = {
    "question": "What is 2 plus 3?",
    "answer": "5"
}

# Step 2: Create the Doc using gsm8k
formatted_doc = gsm8k(line)

# Step 3: Prepare golds and predictions
# Note: 'golds' is from 'line["answer"]', 'predictions' is model output

golds = [line["answer"]]
predictions = ["2"]  # pretend this is model prediction

# golds = [line["answer"], line["answer"], line["answer"]]
# predictions = ["2", "2", "2"]  # pretend this is model prediction

# Step 4: Call the sample_level_fn
score = metric.sample_level_fn(golds, predictions, formatted_doc)

# Step 5: Print the score
print("Score:", score)