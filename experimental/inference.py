# Use a pipeline as a high-level helper
from transformers import pipeline
from datasets import load_dataset


pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.3")

def infer(prompt, examples, **kwargs):
    
        questions, contexts = examples["question"], examples["context"]
        inputs = ["\n".join(prompt,question,context) for question,context in zip(questions,contexts)]
        outputs = pipe(inputs, **kwargs)    
    
        return {"answer_vinuca":outputs}

    
 
if __name__ == "__main__":
    
    prompt =  "USER: Answer the following question using the given question"
    dataset = load_dataset("explodinggradients/wiki-eval")
    generation_args = {
        "do_sample":True,
        "max_new_tokens":512
    }
    batch_size = 1
    dataset = dataset.map(lambda example: infer(prompt, example, **generation_args),
                batch_size = batch_size)
    
    dataset.to_json("ragas_hawk.json")