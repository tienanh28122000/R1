import numpy as np
import pandas as pd
from openai import OpenAI
import lancedb
import pyarrow as pa # Import pyarrow
import os
import time # To get timestamp for element aging

class DynamicRewardCalculator:
    """
    Calculates rewards for language model completions, incorporating a
    diversity incentive for correct reasoning steps based on their similarity
    to the centroid of the top-k nearest successful examples from a LanceDB table.
    """
    def __init__(self,
                 success_table,
                 base_correctness_reward: float = 1.0,
                 diversity_scale: float = 0.5, # Scaling factor for the diversity reward
                 embedding_model: str = "text-embedding-3-small",
                 embedding_dims: int = 1536,
                 top_k_similarity_centroid: int = 10, # Number of nearest neighbors to use for centroid calculation
                 nprobes: int = 10):
        """
        Initializes the diversity reward calculator.

        Args:
            success_table: A LanceDB table object for storing successful completion examples.
                           This table should have a 'vector' column for embeddings,
                           and potentially other metadata columns like 'response' and 'timestamp'.
            base_correctness_reward: The base reward assigned to any correct completion.
            diversity_scale: A scaling factor for the diversity bonus. Higher values
                             provide a stronger incentive for diversity relative to
                             past successful examples.
            embedding_model: The name of the embedding model to use for semantic similarity.
                             Defaults to an OpenAI model.
            embedding_dims: The dimensionality of the embeddings produced by the model.
            top_k_similarity_centroid: The number of top nearest neighbors in the
                                       success_table to use when calculating the centroid
                                       for diversity comparison.
            nprobes: The number of probes to use for ANN search in LanceDB.
        """
        self.success_table = success_table
        self.base_correctness_reward = base_correctness_reward
        self.diversity_scale = diversity_scale
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims
        self.top_k_similarity_centroid = top_k_similarity_centroid
        self.nprobes = nprobes

        # --- Embedding Client ---
        # Using OpenAI based on the provided example's structure
        # WARNING: Assumes OPENAI_API_KEY is set in environment variables.
        self.client = OpenAI()
        # Basic check
        self.client.models.list()


    def get_embedding(self, text: str) -> np.ndarray | None:
        """
        Generates an embedding for the given text using the configured model.
        Returns a numpy array or None if the input text is empty or results
        in a problematic embedding (though explicit API error handling is removed).
        """

        # Direct call without extensive error handling as requested for simplicity
        response = self.client.embeddings.create(input=text, model=self.embedding_model)
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        # Optional: Add a check if needed, though OpenAI API usually returns correct dims
        assert embedding.shape[0] == self.embedding_dims
        return embedding


    def extract_reasoning(self, text: str) -> str:
        """
        Extracts the reasoning part from the completion text.
        """
        # Split by the <reasoning> tag and then by the </reasoning> tag
        answer = text.split("</reasoning>")[0]
        answer = answer.split("<reasoning>")[-1]
        return answer.strip()


    def extract_answer(self, text: str) -> str:
        """
        Extracts the answer part from the completion text (content between <answer> tags).
        """
        # Split by the <answer> tag and then by the </answer> tag
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()


    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates the cosine similarity between two vectors.
        Returns a float between -1 and 1. Handles near-zero vectors to avoid division by zero.
        """
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        # Handle case where one or both vectors are zero or near-zero
        if norm_a < 1e-9 or norm_b < 1e-9:
             return 0.0 # Treat near-zero vectors as having 0.0 similarity

        dot_product = np.dot(vec1, vec2)
        return dot_product / (norm_a * norm_b)


    def calculate_rewards(self, completions, correct_answers) -> list[float]:
        """
        Calculates rewards for a batch of completions for a single question.
        Assigns a base reward for correctness and an additional diversity bonus
        for correct completions based on the semantic similarity of their reasoning
        to the centroid of the top-k nearest successful examples from the success_table.

        Args:
            completions: A list of completion objects, where each object is expected
                         to be a list containing a dictionary like
                         [{'role': 'assistant', 'content': 'completion text'}].
            correct_answers: A list of ground truth correct answers corresponding
                             to each completion in the batch.

        Returns:
            A list of float rewards, corresponding to the input completions.
            Incorrect completions receive a reward of 0. Correct completions
            receive base_correctness_reward + diversity_reward.
        """
        rewards = [0.0] * len(completions) # Initialize rewards list with 0.0
        successful_entries_to_add = [] # Collect successful entries to add later

        for i, (completion_item, correct_answer) in enumerate(zip(completions, correct_answers)):
            completion_text = completion_item[0]['content']
            extracted_answer = self.extract_answer(completion_text)

            is_correct = (extracted_answer == correct_answer)

            if is_correct:
                reasoning_text = self.extract_reasoning(completion_text)
                reasoning_embedding = self.get_embedding(reasoning_text)
                # Base reward for correctness
                current_reward = self.base_correctness_reward
                diversity_reward = 0.0

                # --- Calculate Diversity Reward ---
                # Query the success table for top-k similar examples
                search_vector = reasoning_embedding.tolist()
                # Use a filter to potentially exclude the exact same response if it exists,
                # though with embeddings and reasoning variation, exact matches might be rare.
                # For simplicity here, we just get top-k nearest.
                top_k_success = self.success_table.search(search_vector)\
                                        .limit(self.top_k_similarity_centroid)\
                                        .metric("cosine")\
                                        .nprobes(self.nprobes)\
                                        .to_list()

                if top_k_success:
                    # Extract embeddings from the search results
                    successful_embeddings = np.array([np.array(row['vector'], dtype=np.float32) for row in top_k_success])

                    # Calculate the centroid of these top-k successful embeddings
                    centroid_successful_embeddings = np.mean(successful_embeddings, axis=0)

                    # Calculate similarity to the centroid of historical successful examples
                    similarity_to_historical_centroid = self.cosine_similarity(reasoning_embedding, centroid_successful_embeddings)

                    # Diversity is higher when similarity is lower. Clamp similarity for calculation [0, 1].
                    clamped_similarity = max(0.0, min(1.0, similarity_to_historical_centroid))

                    # Calculate diversity bonus: C * (1 - clamped_similarity)
                    diversity_reward = self.diversity_scale * (1.0 - clamped_similarity)

                    # Add diversity bonus to the base reward
                    current_reward += diversity_reward
                    # print(f"Correct & Diverse: Base={self.base_correctness_reward:.2f}, Similarity to Centroid={similarity_to_historical_centroid:.4f}, Diversity Bonus={diversity_reward:.2f}, Total={current_reward:.2f}")


                rewards[i] = current_reward

                # Prepare to add this successful example to the table
                successful_entries_to_add.append({
                    "vector": reasoning_embedding.tolist(),
                    "response": completion_text, # Store full response or just reasoning? Storing full response for now.
                    "timestamp": int(time.time() * 1000)
                })

        # Add all successful entries to the LanceDB table in one batch
        if successful_entries_to_add:
            # Using try/except for add as LanceDB operations can sometimes fail
            # in multi-process environments, though simplified example removes full handling.
            self.success_table.add(successful_entries_to_add)
            # print(f"Added {len(successful_entries_to_add)} successful entries to table.")


        return [float(r) for r in rewards]

# --- Example Usage ---
if __name__ == "__main__":
    # Instantiate the calculator
    # *** WARNING: Script will crash if OPENAI_API_KEY is missing or invalid,
    # or if DB operations fail, or embeddings fail, or XML is invalid. ***

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

    # Clear table for a clean run if exist_ok=True
    try:
        success_table.delete("1=1")
        print("Cleared success table for example run.")
    except Exception as e:
        print(f"Warning: Could not clear success table: {e}")


    reward_calculator = DynamicRewardCalculator(
        success_table=success_table,
        base_correctness_reward=1.0,
        diversity_scale=3.0,
        embedding_model="text-embedding-3-small", # Ensure this matches schema dims
        embedding_dims=embedding_dims,
        top_k_similarity_centroid=10, # Use top 3 for centroid in this example
        nprobes=10
    )

    print(f"Initial Success Table Size: {len(reward_calculator.success_table)}")

    # --- First Batch ---
    # All correct, but reasoning varies. Expecting diversity bonus.
    print("\n--- Processing First Batch ---")
    completions1 = [
        {'role': 'assistant', 'content': "<reasoning>The multiplication is 5 times 2 equals 10.</reasoning><answer>10</answer>"},
        {'role': 'assistant', 'content': "<reasoning>Based on standard arithmetic, 5 * 2 = 10.</reasoning><answer>10</answer>"},
        {'role': 'assistant', 'content': "<reasoning>A quick calculation shows 5 multiplied by 2 is 10.</reasoning><answer>10</answer>"},
        {'role': 'assistant', 'content': "<reasoning>Simply 5 + 5 = 10.</reasoning><answer>10</answer>"}, # Different reasoning style
        {'role': 'assistant', 'content': "<reasoning>Incorrect answer.</reasoning><answer>12</answer>"}, # Incorrect
    ]
    # Reshape to match the expected input format [[{'role': 'assistant', 'content': '...'}]...]
    completions1_formatted = [[item] for item in completions1]
    answers1 = ["10"] * len(completions1) # All answers are 10 for this question

    rewards1 = reward_calculator.calculate_rewards(completions1_formatted, answers1)
    print("\nBatch 1 Results:")
    for i, reward in enumerate(rewards1):
        print(f"Completion {i+1}: Reward = {reward:.4f}")

    print(f"Success Table Size after Batch 1: {len(reward_calculator.success_table)}")
    # print("Success Table Content after Batch 1:")
    # try:
    #     print(reward_calculator.success_table.to_pandas())
    # except Exception as e:
    #     print(f"Could not display success table content: {e}")

    print("-" * 30)

    # --- Second Batch ---
    # Some correct, some incorrect. Expecting correct ones to get base + diversity.
    # Diversity bonus will be based on similarity to the centroid of examples from Batch 1.
    print("\n--- Processing Second Batch ---")
    completions2 = [
         {'role': 'assistant', 'content': "<reasoning>As calculated before, 5 times 2 is 10.</reasoning><answer>10</answer>"}, # Similar to Batch 1 reasonings
         {'role': 'assistant', 'content': "<reasoning>Let's recheck: 5 multiplied by 2 is indeed 10.</reasoning><answer>10</answer>"}, # Similar to Batch 1 reasonings
         {'role': 'assistant', 'content': "<reasoning>Using grouping: (1+1+1+1+1) * 2 = 10.</reasoning><answer>10</answer>"}, # Potentially more diverse
         {'role': 'assistant', 'content': "<reasoning>This is not the answer.</reasoning><answer>7</answer>"}, # Incorrect
    ]
    completions2_formatted = [[item] for item in completions2]
    answers2 = ["10"] * len(completions2) # All answers are 10 for this question

    rewards2 = reward_calculator.calculate_rewards(completions2_formatted, answers2)
    print("\nBatch 2 Results:")
    for i, reward in enumerate(rewards2):
        print(f"Completion {i+1}: Reward = {reward:.4f}")

    print(f"Success Table Size after Batch 2: {len(reward_calculator.success_table)}")
    # print("Success Table Content after Batch 2:")
    # try:
    #     print(reward_calculator.success_table.to_pandas())
    # except Exception as e:
    #      print(f"Could not display success table content: {e}")

    print("-" * 30)

    # --- Third Batch (demonstrating low diversity reward for highly similar reasoning) ---
    print("\n--- Processing Third Batch (Low Diversity Expectation) ---")
    completions3 = [
         {'role': 'assistant', 'content': "<reasoning>Simply 5 times 2 is 10.</reasoning><answer>10</answer>"}, # Very similar to Batch 1/2
         {'role': 'assistant', 'content': "<reasoning>Calculation result is ten.</reasoning><answer>10</answer>"}, # Also similar
         {'role': 'assistant', 'content': "<reasoning>Wrong again.</reasoning><answer>15</answer>"}, # Incorrect
    ]
    completions3_formatted = [[item] for item in completions3]
    answers3 = ["10"] * len(completions3)

    rewards3 = reward_calculator.calculate_rewards(completions3_formatted, answers3)
    print("\nBatch 3 Results:")
    for i, reward in enumerate(rewards3):
        print(f"Completion {i+1}: Reward = {reward:.4f}")

    print(f"Success Table Size after Batch 3: {len(reward_calculator.success_table)}")
    print("-" * 30)

    # Clean up LanceDB directory after example
    # import shutil
    # try:
    #     shutil.rmtree(db_path)
    #     print(f"Cleaned up LanceDB directory: {db_path}")
    # except Exception as e:
    #     print(f"Warning: Could not remove LanceDB directory: {e}")