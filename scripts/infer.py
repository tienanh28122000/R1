from vllm import LLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def print_outputs(outputs):
        print("\nGenerated Outputs:\n" + "-" * 80)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}\n")
            print(f"Generated text: {generated_text!r}")
            print("-" * 80)

model = "Qwen2.5-0.5B-Instruct"
llm = LLM(model=model,
        max_model_len=4096)

question = "Question: The girls are trying to raise money for a carnival. Kim raises $320 more than Alexandra, who raises $430, and Maryam raises $400 more than Sarah, who raises $300. How much money, in dollars, did they all raise in total? Answer:"

conversation = [{
    "role": "user",
    "content": question
}]

outputs = llm.chat(conversation)
print_outputs(outputs)