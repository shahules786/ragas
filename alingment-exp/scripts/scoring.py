
import numpy as np
from datasets import Dataset
import os
import typing as t
import json

from ragas import EvaluationDataset, evaluate
from sklearn.metrics import f1_score
from ragas.metrics._aspect_critic import AspectCriticWithReference
from .mutation import reverse_engineer_instruction_from_correct_examples, get_feedback_for_prompt, generate_prompt_from_feedback, do_cross_over
from .utils import score_prompts, stratified_sample_no_duplicates


INITIAL_PROMPT = "Given the user_input, reference and response. Is the response correct compared with the reference"
INITIAL_PROMPT = "Evaluate the response based on its alignment with the reference text and determine if it is correct. Consider the accuracy of the information, the relevance of the details provided, and any factual inaccuracies. Ensure that specific dates or timeframes mentioned in the reference are present in the response, as their absence can lead to misalignment. Emphasize the importance of verifying that all critical details, such as dates, are included to ensure full alignment with the reference. Provide a reason for your evaluation and assign a verdict: 1 if the response is correct and aligns well with the reference, and 0 if there are significant inaccuracies, missing critical details like dates, or misalignments."
TOP_PROMPTS = 1

def subset_and_embed_training_samples(dataset, embedding_model, num_samples, seed=42):
    
    np.random.seed(seed)
    num_samples = min(num_samples, len(dataset))
    indices = np.random.randint(0,len(dataset),size=num_samples)
    dataset = dataset.select(indices)
    embedding_vector = [data["input"]["user_input"] for data in dataset]
    
    vectors = embedding_model.embed_documents(embedding_vector)
    return dataset, np.array(vectors)


async def run_evaluation(train_data, test_data, llm, embedding_model, num_train_samples, prompt_optimisation: bool, dynamic_retrieval:bool, search_type: t.Literal["similarity","random"], batch_size:int=12, seed:int=42):
    
    
    train_dataset = Dataset.from_json(train_data)
    test_dataset = Dataset.from_json(test_data)

    train_dataset, train_vectors = subset_and_embed_training_samples(train_dataset, embedding_model, num_train_samples, seed)

    if os.path.exists("train_vectors.npy"):
        os.remove("train_vectors.npy")

    #temp save
    np.save("train_vectors.npy", train_vectors)
    with open("train_vectors.json", "w") as f:
        json.dump(train_dataset.to_list(), f)

    metric_kwargs = {
    "llm":llm,
    "embedding_model":embedding_model,
    "dynamic_retrieval": dynamic_retrieval,
    "search_type": search_type,
    }

    critic = AspectCriticWithReference(name="answer_correctness",
                          definition=INITIAL_PROMPT, **metric_kwargs)

    if prompt_optimisation:
        print("Optimising prompt. Setting dynamic retrieval and examples to False to avoid inference")
        critic.dynamic_retrieval = False
        critic.single_turn_prompt.examples = []
        batch_data = stratified_sample_no_duplicates(train_dataset,batch_size,target_column='qdrant',seed=seed)


        scored_initial_prompt = score_prompts(critic, batch_data, [INITIAL_PROMPT])
        intial_prompt_score = scored_initial_prompt[0]['score']
        print("Initial prompt score is %s", intial_prompt_score)
        
        #global optimisation
        candidate_prompts = await reverse_engineer_instruction_from_correct_examples(train_dataset,llm, num_instructions=5, num_samples=3,seed=seed)
        for i, prompt in enumerate(candidate_prompts):
            candidate_prompts[i] = await do_cross_over(prompt, INITIAL_PROMPT, llm)
        print("Candidate prompts are %s", candidate_prompts)
        
        
        scored_prompts = score_prompts(critic, batch_data, candidate_prompts)
        scored_prompts = sorted(scored_prompts, key=lambda x: x['score'], reverse=True)
        best_score = scored_prompts[0]['score']
        
        print("Scored prompts %s", [(prompt['prompt'], prompt["score"]) for prompt in scored_prompts])

        
        if best_score < intial_prompt_score:
            print("Initial prompt was not improved")
            next_round_prompts = scored_initial_prompt
        else:
            next_round_prompts = [prompt for prompt in scored_prompts if prompt['score'] >= intial_prompt_score]
            #filter prompts that goes into next round
            next_round_prompts = next_round_prompts[:TOP_PROMPTS]
            print("Found prompts better than the initial prompt")
            print("Next round prompts are %s", [(prompt['prompt'], prompt["score"]) for prompt in next_round_prompts])



        #local optimisation
        improved_prompt_scores = []
        for prompt in next_round_prompts:
            print("Number of feedbacks for prompt %s", prompt["feedback"])
            feedbacks = await get_feedback_for_prompt(prompt, llm)
            improvement_prompt = await generate_prompt_from_feedback(prompt["prompt"], feedbacks,llm)
            print("proposed prompt is %s", improvement_prompt)
            improvement_score = score_prompts(critic, batch_data, [improvement_prompt])
            if improvement_score[0]['score'] >= prompt['score']:
                improved_prompt_scores.append(improvement_score)
                percent_improvement = (improvement_score[0]['score'] - prompt['score'])/prompt['score']
                print("Prompt improved by %s", percent_improvement*100)
            else:
                print("Prompt was not improved, score is %s", improvement_score[0]['score'])

        if improved_prompt_scores:
            best_improved_prompt = sorted(improved_prompt_scores, key=lambda x: x['score'], reverse=True)[0]
            critic.single_turn_prompt.instruction = best_improved_prompt['prompt']
            print("Best improved prompt is %s", best_improved_prompt['prompt'])

        critic.dynamic_retrieval = dynamic_retrieval


    #score the settings
    y_true = test_dataset["target"]
    eval_dataset = EvaluationDataset.from_hf_dataset(test_dataset)
    result = evaluate(dataset=eval_dataset[:],metrics=[critic])
    df = result.to_pandas()
    y_pred = df["answer_correctness"].values.tolist()
    score = f1_score(y_true, y_pred)

    result_dict = {
        "score":score,
        "prompt":critic.single_turn_prompt.instruction,
        "metric":critic.name,
        "initial_prompt":INITIAL_PROMPT,
        "final_prompt":critic.single_turn_prompt.instruction,
    }
    result_dict.update(metric_kwargs)
    result_dict["llm"] = embedding_model.embeddings.model
    result_dict["embedding_model"] = dynamic_retrieval
    if not result_dict["dynamic_retrieval"]:
        result_dict.update({"search_type":None,"embedding_model":None})
        
        
    #delete the index
    os.remove("train_vectors.npy")
    os.remove("train_vectors.json")
    
    return result_dict    
        
    
    
    
    
    
    
    
    