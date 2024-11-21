import numpy as np
from tqdm import tqdm

from .sampling import (CrossOverPrompt, FeedbackExample, FeedbackMutationInput,
                       FeedbackMutationPrompt,
                       FeedbackMutationPromptGeneration,
                       FeedbackMutationPromptInput, ParentPrompts,
                       PromptFromCorrectExample, SingleExample)


def format_dict(dic):
    string = ""
    for key, val in dic.items():
        string += f"\n{key}:\n\t{val}\n"
    return string


async def reverse_engineer_instruction_from_correct_examples(data,llm,num_instructions=5, num_samples=3,seed=42):
    """
    reverse engineer prompts using correct example
    """
    generated_prompts = []
    
    np.random.seed(seed=seed)
    correct_examples = [sample for sample in data if sample['qdrant'] in ['TN','TP']]

    prompt = PromptFromCorrectExample()
    for i in tqdm(range(num_instructions)):        
        examples = np.random.choice(correct_examples,size=num_samples)
        curr_examples = []
        for example in examples:
            curr_examples.append((format_dict(example["input"]), example['output']))
        prompt_input = SingleExample(examples=curr_examples)
        response = await prompt.generate(data=prompt_input,llm=llm)
        generated_prompts.append(response.instruction)

    return generated_prompts


async def get_feedback_for_prompt(prompt,llm):
    mutation_prompt = FeedbackMutationPrompt()
    original_prompt = prompt['prompt']
    examples = []
    for item in prompt['feedback']:
        input_ = item['input']
        input_ = format_dict(input_)
        prompt_input = FeedbackExample(input=input_,output=item["incorrect_output"],expected_output=item["expected_output"])
        examples.append(prompt_input)
    prompt_input = FeedbackMutationInput(prompt=original_prompt,examples=examples)
    response = await mutation_prompt.generate(data=prompt_input,llm=llm)
    return response.feedbacks
        
        
async def generate_prompt_from_feedback(prompt,feedbacks,llm):
    improvement_prompt = FeedbackMutationPromptGeneration()
    prompt_input = FeedbackMutationPromptInput(prompt=prompt,feedbacks=feedbacks)
    response = await improvement_prompt.generate(data=prompt_input,llm=llm)
    return response.instruction


async def do_cross_over(prompt_x, prompt_y, llm):
    
    cross_over_prompt = CrossOverPrompt()
    prompt_input = ParentPrompts(parent_1=prompt_x, parent_2=prompt_y)
    response = await cross_over_prompt.generate(data=prompt_input, llm=llm)
    return response.prompt