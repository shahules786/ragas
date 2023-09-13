# Use a pipeline as a high-level helper
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline

pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.3", device="cuda:0",
                torch_dtype = torch.float16)

def infer(prompt, examples, **kwargs):
    
        question, context = examples["question"], examples["context"]
        inputs = "\n".join([prompt,question])
        outputs = pipe(inputs+"\nASSISTANT:", **kwargs)    
    
        return outputs[0]["generated_text"]

    
 
if __name__ == "__main__":
    
    prompt =  "USER: Given a question provide an incomplete or partial answer"
    dataset = load_dataset("explodinggradients/wiki-eval")
    generation_args = {
        "do_sample":True,
        "max_new_tokens":512
    }
    batch_size = 1
    # dataset = dataset.map(lambda example: infer(prompt, example, **generation_args),
    #             batch_size = batch_size)
    outputs = []
    for item in tqdm(dataset["train"]):
        output = infer(prompt, item, **generation_args)
        outputs.append(output)
        
    dataset = dataset["train"].add_column("answer_vinuca", outputs)
    dataset.to_json("ragas_hawk.json")