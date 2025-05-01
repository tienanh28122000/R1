# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase, SampleLevelMetric
import re
import numpy as np
from aenum import extend_enum

def custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in predictions] 
    return [1.0 if match else 0.0 for match in matches]

my_custom_metric = SampleLevelMetric(
    metric_name="format",
    sample_level_fn=custom_metric,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

extend_enum(Metrics, "format", my_custom_metric)

gsm8k_test = LightevalTaskConfig(
    name="gsm8k_test",
    suite=["lighteval"],
    prompt_function=prompt.gsm8k,
    hf_repo="gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=512,
    # metric=[Metrics.expr_gold_metric, my_custom_metric],
    metric=[Metrics.expr_gold_metric],
    stop_sequence=["Question="],
    trust_dataset=True,
    version=0,
)

TASKS_TABLE = [gsm8k_test]