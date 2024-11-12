from ragas.prompt import PydanticPrompt
from pydantic import BaseModel
from ragas.metrics._aspect_critic import AspectCriticInputWithReference, AspectCriticOutputWithReference
import typing as t


class SingleExample(BaseModel):
    inputs: t.List[t.Dict[str, t.Any]]
    outputs: t.List[t.Dict[str, t.Any]]
    
    
class CorrectIncorrectExample(BaseModel):
    correct_examples: SingleExample
    incorrect_examples: SingleExample
    
class ReversedPrompt(BaseModel):
    instruction: str
    
class PromptFromCorrectExample(PydanticPrompt[SingleExample, ReversedPrompt]):
    name: str = "reverse_engineer"
    instruction: str = (
        "Given a set of inputs and expected outputs from LLM, reverse engineer the instruction."
    )
    input_model = SingleExample
    output_model = ReversedPrompt
    
    
class PromptFromCorrectIncorrectExamples(PydanticPrompt[CorrectIncorrectExample, ReversedPrompt]):
    name: str = "reverse_engineer"
    instruction: str = (
        "Given a set of inputs and expected output first, then followed by inputs with incorrect outputs from LLM, reverse engineer the instruction."
    )
    input_model = CorrectIncorrectExample
    output_model = ReversedPrompt
    