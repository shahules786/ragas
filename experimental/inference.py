# Use a pipeline as a high-level helper
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import torch

pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.3", device="cuda:0",
                torch_dtype = torch.float16)

def infer(prompt, examples, **kwargs):
    
        question, context = examples["question"], examples["context"]
        # inputs = "\n".join([prompt,question,context])
        outputs = pipe(question, **kwargs)    
    
        return outputs[0]["generated_text"]

    
 
if __name__ == "__main__":
    
    prompt =  "USER: Answer the following question using the given question"
    dataset = load_dataset("explodinggradients/wiki-eval")
    generation_args = {
        "do_sample":True,
    }
    batch_size = 1
    # dataset = dataset.map(lambda example: infer(prompt, example, **generation_args),
    #             batch_size = batch_size)
    outputs = []
    for item in tqdm(dataset["train"]):
        output = infer(prompt, item, **generation_args)
        outputs.append(output)
        
    dataset = dataset["train"].add_column("answer_v2", outputs)
    dataset.to_json("ragas_hawk.json")