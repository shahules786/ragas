from datasets import Dataset
from collections import defaultdict
from typing import Any
from sklearn.metrics import f1_score
import numpy as np

from ragas import evaluate, EvaluationDataset


def stratified_sample_no_duplicates(dataset: Dataset, batch_size: int, target_column: str = 'target', seed=42) -> Dataset:
    """
    Function to sample a batch of data with as equal distribution of target values as possible, avoiding duplicates.
    
    Parameters:
        dataset (Dataset): The Hugging Face dataset to sample from.
        batch_size (int): The number of samples in the output batch.
        target_column (str): The column name containing the target values.
        seed (int): Random seed for reproducibility.
        
    Returns:
        Dataset: A batch of data with as close to equal distribution of target values as possible without duplicates.
    """
    np.random.seed(seed)
    # Get all unique target values
    target_values = dataset.unique(target_column)
    num_classes = len(target_values)
    
    # Calculate initial number of samples per class
    samples_per_class = batch_size // num_classes
    remaining_slots = batch_size % num_classes

    # Group indices by target values
    target_indices = defaultdict(list)
    for idx, example in enumerate(dataset):
        target_indices[example[target_column]].append(idx)
    
    # Track used indices and sampled indices
    used_indices = set()
    sampled_indices = []
    
    # Sampling loop to avoid duplicates
    for target in target_values:
        available_indices = [idx for idx in target_indices[target] if idx not in used_indices]
        num_to_sample = min(samples_per_class, len(available_indices))
        
        if num_to_sample > 0:
            sampled = np.random.choice(available_indices, num_to_sample, replace=False).tolist()
            sampled_indices.extend(sampled)
            used_indices.update(sampled)
    
    # Handle remaining slots by resampling from classes with remaining samples
    for target in target_values:
        if remaining_slots <= 0:
            break
        available_indices = [idx for idx in target_indices[target] if idx not in used_indices]
        if available_indices:
            sampled = np.random.choice(available_indices, min(len(available_indices), remaining_slots), replace=False).tolist()
            sampled_indices.extend(sampled)
            used_indices.update(sampled)
            remaining_slots -= len(sampled)

    # Shuffle sampled indices to ensure randomness
    np.random.shuffle(sampled_indices)
    return dataset.select(sampled_indices)



def serialize_for_json(data: Any) -> Any:
    """
    Convert custom objects into a JSON-serializable format.

    Parameters:
    data (Any): The data to be converted, which may contain custom objects.

    Returns:
    Any: A JSON-serializable version of the input data.
    """
    if isinstance(data, dict):
        return {key: serialize_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialize_for_json(item) for item in data]
    elif hasattr(data, "__dict__"):
        # Convert custom objects by serializing their attributes (assumes they use __dict__)
        return serialize_for_json(data.__dict__)
    else:
        return data  # Return the data as-is if it is already JSON-serializable



def score_prompts(critic, dataset, candidate_prompts):
    result_scores = []
    for prompt in candidate_prompts:
        y_true = dataset['target']
        eval_dataset = [data['input'] for data in dataset]
        eval_dataset = EvaluationDataset.from_list(eval_dataset)
        critic.single_turn_prompt.prompt = prompt
        result = evaluate(dataset=eval_dataset,metrics=[critic])
        traces = result.traces
        df = result.to_pandas()
        y_pred = df["answer_correctness"].values.tolist()
        incorrect_indices = [i for i in range(len(y_pred)) if y_pred[i]!=y_true[i]]
        print("Number of incorrect predictions", len(incorrect_indices))
        print("Incorrect indices", incorrect_indices)
        feedback_samples = []
        for idx in incorrect_indices:
            if dataset[idx]['qdrant'] in ['TP','TN']:
                dic = {
                    "input":dataset[idx]["input"],
                    "incorrect_output":serialize_for_json(traces[idx]['answer_correctness']),
                    "expected_output":dataset[idx]["output"]
                }
                dic['incorrect_output'] = dic['incorrect_output']['0_single_turn_aspect_critic_prompt_with_reference']['output'][0]
                feedback_samples.append(dic)
        fscore = f1_score(y_true,y_pred)
        result_scores.append({"prompt":prompt,"score":fscore,"feedback":feedback_samples})
    return result_scores
        