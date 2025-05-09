import numpy as np
from openai import OpenAI
# Depending on your actual embedding model choice, you might need to import
# from sentence_transformers import SentenceTransformer

class DiversityRewardCalculator:
    """
    Calculates rewards for language model completions, incorporating a
    diversity incentive for correct reasoning steps within a batch for a single question.

    This version uses the similarity to the centroid of other correct reasonings
    in the batch as the measure for diversity.
    """
    def __init__(self,
                 base_correctness_reward: float = 1.0,
                 diversity_scale: float = 0.5, # Scaling factor for the diversity reward (corresponds to C)
                 embedding_model: str = "text-embedding-3-small", # Using OpenAI based on example structure
                 embedding_dims: int = 1536):
        """
        Initializes the diversity reward calculator.

        Args:
            base_correctness_reward: The base reward assigned to any correct completion.
            diversity_scale: A scaling factor for the diversity bonus. Higher values
                             provide a stronger incentive for diversity.
            embedding_model: The name of the embedding model to use for semantic similarity.
                             Defaults to an OpenAI model as in the example.
            embedding_dims: The dimensionality of the embeddings produced by the model.
        """
        self.base_correctness_reward = base_correctness_reward
        self.diversity_scale = diversity_scale
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims

        # --- Embedding Client ---
        # Using OpenAI based on the provided example's structure
        # WARNING: Assumes OPENAI_API_KEY is set in environment variables.
        self.client = OpenAI()

        # If using SentenceTransformer:
        # self.embedding_model_st = SentenceTransformer(embedding_model)


    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates an embedding for the given text using the configured model.
        Returns a numpy array.
        *** WARNING: Removed API error handling. Crashes possible if API fails or text is problematic. ***
        Also handles empty text explicitly to prevent potential issues with the embedding model.
        """
        if not text or not text.strip():
            # Return a zero vector or handle as appropriate for empty content
            # Returning a vector of small non-zero values avoids division by zero in cosine_similarity norm calculation
            return np.full(self.embedding_dims, 1e-9, dtype=np.float32)


        response = self.client.embeddings.create(input=text, model=self.embedding_model)
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding

        # If using SentenceTransformer:
        # return self.embedding_model_st.encode(text, convert_to_numpy=True)


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
        Returns a float between -1 and 1.
        Handles near-zero vectors explicitly to avoid division by zero.
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
        to the centroid of *other* correct reasonings in the batch.

        Args:
            completions: A list of completion objects, where each object is expected
                         to be a list containing a dictionary like
                         [{'role': 'assistant', 'content': 'completion text'}].
                         This structure matches the example provided.
            correct_answer: The ground truth correct answer for the question.

        Returns:
            A list of float rewards, corresponding to the input completions.
            Incorrect completions receive a reward of 0. Correct completions
            receive base_correctness_reward + diversity_reward.
        """
        rewards = [0.0] * len(completions) # Initialize rewards list with 0.0 for all completions

        # List to store information about correct completions that have valid reasoning
        # Stores tuples of (original_index, reasoning_text, reasoning_embedding)
        correct_reasonings_info = []

        # First pass: Identify correct completions and extract their reasoning and embeddings
        for i, (completion_item, correct_answer) in enumerate(zip(completions, correct_answers)):
            # Extract the completion text safely
            completion_text = completion_item[0]['content']
            extracted_answer = self.extract_answer(completion_text)

            is_correct = (extracted_answer == correct_answer)

            if is_correct:
                reasoning_text = self.extract_reasoning(completion_text)
                reasoning_embedding = self.get_embedding(reasoning_text)
                # Check if embedding is valid (not a zero vector from error handling in get_embedding)
                if np.linalg.norm(reasoning_embedding) > 1e-9: # Use a small tolerance
                    correct_reasonings_info.append((i, reasoning_text, reasoning_embedding))
                else:
                    # Correct but embedding invalid/zero, assign base reward only
                    rewards[i] = self.base_correctness_reward


        # Calculate diversity reward based on similarity to centroid if there's more than one valid reasoning
        if len(correct_reasonings_info) > 1:
            num_correct_reasonings = len(correct_reasonings_info)

            # Collect all valid embeddings from correct reasonings in this batch
            all_embeddings = np.array([emb for _, _, emb in correct_reasonings_info]) # Shape (Nc, embedding_dims)

            # Calculate diversity reward for each correct reasoning
            for i in range(num_correct_reasonings):
                current_original_index, _, current_embedding = correct_reasonings_info[i]

                # Calculate the sum of embeddings of *other* reasonings
                # Create a boolean mask to exclude the current embedding
                mask = np.ones(num_correct_reasonings, dtype=bool)
                mask[i] = False
                other_embeddings = all_embeddings[mask]

                # Calculate the centroid (mean) of other embeddings
                # We know num_correct_reasonings > 1, so len(other_embeddings) > 0
                centroid_other_embeddings = np.mean(other_embeddings, axis=0) # Shape (embedding_dims,)

                # Calculate similarity to the centroid of others
                # This will be a float between -1 and 1.
                similarity_to_centroid = self.cosine_similarity(current_embedding, centroid_other_embeddings)

                # For diversity reward, we typically treat similarity as non-negative or use 1 - similarity.
                # Clamping similarity to [0, 1] before calculating dissimilarity (1 - similarity)
                # aligns with the interpretation in the original proposal's table.
                clamped_similarity_for_diversity = max(0.0, min(1.0, similarity_to_centroid))

                # Calculate diversity reward using the function: C * (1 - clamped_similarity)
                # C is self.diversity_scale. Dissimilarity = 1 - similarity.
                # Higher dissimilarity (lower similarity to centroid) gives higher reward.
                diversity_reward = self.diversity_scale * (1.0 - clamped_similarity_for_diversity)

                # Final reward for this correct completion is base reward plus diversity bonus
                # We only assign the reward here if it had valid reasoning used in diversity calculation.
                # Items with no reasoning already got the base reward in the first pass.
                rewards[current_original_index] = self.base_correctness_reward + diversity_reward

        elif len(correct_reasonings_info) == 1:
            # Only one correct reasoning with embedding, no diversity to reward/penalize among others using centroid.
            # The base reward is already assigned in the first pass for this case.
            # Explicitly setting here for clarity that diversity bonus is 0.
            original_index, _, _ = correct_reasonings_info[0]
            rewards[original_index] = self.base_correctness_reward # Diversity bonus is 0

        # Incorrect completions already have 0.0 reward from initialization.
        # Correct completions with no valid reasoning already have base reward from initialization.

        return [float(r) for r in rewards]

# --- Example Usage ---
if __name__ == "__main__":
    # Instantiate the calculator
    # Make sure OPENAI_API_KEY environment variable is set.
    # *** WARNING: Script will crash if OPENAI_API_KEY is missing or invalid,
    # or if embedding calls fail. ***
    diversity_reward_calculator = DiversityRewardCalculator(
        base_correctness_reward=1.0,
        diversity_scale=3.0 # Adjust this to control diversity incentive strength
    )


    # --- Example Batch 1 ---
    print("\n--- Processing Batch 1 ---")
    # Question: What is 5 * 2? Answer: 10
    # Expect rewards to reflect diversity relative to the centroid of others in the batch.
    # The first few similar reasonings might get lower diversity bonus, the slightly different one more.
    completions1 = [
        [{'role': 'assistant', 'content': "<reasoning>The calculation is 5*2=10.</reasoning><answer>10</answer>"}], # Correct, standard reasoning 1
        [{'role': 'assistant', 'content': "<reasoning>Let's try adding, 5+2=7.</reasoning><answer>7</answer>"}],     # Incorrect
        [{'role': 'assistant', 'content': "<reasoning>Multiplying five by two gives ten.</reasoning><answer>10</answer>"}], # Correct, standard reasoning 2 (similar to 1)
        [{'role': 'assistant', 'content': "<reasoning>Thinking step-by-step, I multiply 5 and 2 to get 10.</reasoning><answer>10</answer>"}], # Correct, standard reasoning 3 (similar to 1 and 2)
        [{'role': 'assistant', 'content': "<reasoning>I know 5 times 2 is 10 from memory, it's a basic multiplication fact.</reasoning><answer>10</answer>"}], # Correct, slightly different reasoning?
        [{'role': 'assistant', 'content': "The result is <answer>25</answer>"}], # Incorrect (missing reasoning tag)
        [{'role': 'assistant', 'content': "<reasoning>If you have 5 pairs of items, you have 10 items in total.</reasoning><answer>10</answer>"}], # Correct, different framing
    ]
    correct_answer1 = ["10" for _ in range(len(completions1))]

    rewards1 = diversity_reward_calculator.calculate_rewards(completions1, correct_answer1)
    print("\nBatch 1 Results:")
    print("\nRewards:", rewards1)

    # --- Example Batch 3 (Only one correct answer with reasoning) ---
    print("\n--- Processing Batch 3 ---")
    # Question: What is 1 + 1? Answer: 2
    # Expected: The single correct answer to get only the base reward (diversity bonus is 0).
    completions3 = [
        [{'role': 'assistant', 'content': "<reasoning>Adding one and one gives two.</reasoning><answer>2</answer>"}], # Correct
        [{'role': 'assistant', 'content': "<reasoning>Let's try 3.</reasoning><answer>3</answer>"}], # Incorrect
    ]
    correct_answer3 = ["2" for _ in range(len(completions3))]

    rewards3 = diversity_reward_calculator.calculate_rewards(completions3, correct_answer3)
    print("\nBatch 3 Results:")
    print("\nRewards:", rewards3)

    # --- Example Batch 4 (No correct answers) ---
    print("\n--- Processing Batch 4 ---")
    # Question: What is 1 + 1? Answer: 2
    # Expected: All rewards to be 0.
    completions4 = [
        [{'role': 'assistant', 'content': "<reasoning>Adding one and one gives three.</reasoning><answer>3</answer>"}], # Incorrect
        [{'role': 'assistant', 'content': "<reasoning>Let's try 4.</reasoning><answer>4</answer>"}], # Incorrect
    ]
    correct_answer4 = ["2" for _ in range(len(completions4))]

    rewards4 = diversity_reward_calculator.calculate_rewards(completions4, correct_answer4)
    print("\nBatch 4 Results:")
    print("\nRewards:", rewards4)

    # --- Example Batch 5 (Correct answers, some with no reasoning) ---
    print("\n--- Processing Batch 5 ---")
    # Question: What is 1 + 1? Answer: 2
    # Expected: Correct answers with reasoning to potentially get diversity bonus,
    # correct answers without reasoning to get only base reward.
    completions5 = [
        [{'role': 'assistant', 'content': "<answer>2</answer>"}], # Correct, no reasoning (base reward only)
        [{'role': 'assistant', 'content': "<reasoning>The result of 1+1 is two.</reasoning><answer>2</answer>"}], # Correct, reasoning 1
        [{'role': 'assistant', 'content': "<reasoning>Adding one and one is 2.</reasoning><answer>2</answer>"}], # Correct, reasoning 2 (similar to 1)
        [{'role': 'assistant', 'content': "<reasoning>Nope, it's 3.</reasoning><answer>3</answer>"}], # Incorrect
        [{'role': 'assistant', 'content': "<reasoning>Count on your fingers: 1, then add 1 more makes 2.</reasoning><answer>2</answer>"}], # Correct, different reasoning
    ]
    correct_answer5 = ["2" for _ in range(len(completions5))]

    rewards5 = diversity_reward_calculator.calculate_rewards(completions5, correct_answer5)
    print("\nBatch 5 Results:")
    print("\nRewards:", rewards5)

    # --- Example Batch 6 (All correct, very similar reasoning) ---
    print("\n--- Processing Batch 6 ---")
    # Question: What is 2 + 2? Answer: 4
    # Expected: Correct answers to get base reward + a small diversity bonus (due to high similarity to centroid).
    completions6 = [
        [{'role': 'assistant', 'content': "<reasoning>Two plus two equals four.</reasoning><answer>4</answer>"}],
        [{'role': 'assistant', 'content': "<reasoning>The sum of two and two is four.</reasoning><answer>4</answer>"}],
        [{'role': 'assistant', 'content': "<reasoning>Adding 2 and 2 results in 4.</reasoning><answer>4</answer>"}],
    ]
    correct_answer6 = ["4" for _ in range(len(completions6))]

    rewards6 = diversity_reward_calculator.calculate_rewards(completions6, correct_answer6)
    print("\nBatch 6 Results:")
    print("\nRewards:", rewards6)
