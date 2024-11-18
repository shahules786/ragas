
from ragas.prompt import PydanticPrompt
from pydantic import BaseModel
from ragas.metrics._aspect_critic import AspectCriticInputWithReference, AspectCriticOutputWithReference
import typing as t


## Reverse Engineer prompts from examples

class SingleExample(BaseModel):
    examples: t.List[t.Tuple[str, t.Any]]
    
    
class Prompt(BaseModel):
    instruction: str
    
class PromptFromCorrectExample(PydanticPrompt[SingleExample, Prompt]):
    name: str = "reverse_engineer"
    instruction: str = (
        "Given a set of (input containing (user_input, response, reference), expected output) pairs that were manually annotated, guess and generate the instruction given to the annotator."
    )
    input_model = SingleExample
    output_model = Prompt
    
    
class ParentPrompts(BaseModel):
    parent_1: str
    parent_2: str
    

class OffspringPrompt(BaseModel):
    prompt: str
    
    
class CrossOverPrompt(PydanticPrompt[ParentPrompts, OffspringPrompt]):
    name: str = "crossover"
    instruction: str = (
        "You are a mutator who is familiar with the concept of cross-over in genetic algorithm, namely",
        "combining the genetic information of two parents to generate new offspring. Given two parent",
        "prompts, you will perform a cross-over to generate an offspring prompt that covers the same",
        "semantic meaning as both parents.",
    )
    input_model = ParentPrompts
    output_model = OffspringPrompt
    examples = [
        (
            ParentPrompts(
                parent_1="Now you are a categorizer, your mission is to ascertain the sentiment of the provided text, either favorable or unfavorable.",
                parent_2="Assign a sentiment label to the given sentence from [’negative’, ’positive’] and return only the label without any other text.",
            ),
            OffspringPrompt(
                prompt="Your mission is to ascertain the sentiment of the provided text and assign a sentiment label from [’negative’, ’positive’].",
            ),
        )
    ]
    
## Feedback mutation with reference

class FeedbackExample(BaseModel):
    input: str
    output: t.Dict[str, t.Any]
    expected_output: t.Dict[str, t.Any]
    
class FeedbackMutationInput(BaseModel):
    prompt: str
    examples: t.List[FeedbackExample]
    
class FeedbackMutationOutput(BaseModel):
    feedbacks: t.List[str]
    
class FeedbackMutationPrompt(PydanticPrompt[FeedbackMutationInput, FeedbackMutationOutput]):
    name: str = "feedback_mutation"
    instruction: str = (
        "You're an expert reviewer. Given a prompt and a set of (input  containing (user_input, response, reference), output, expected_output) examples, give maximum 3 feedbacks on how the prompt can be improved to correct the mistakes in incorrect outputs and reach expected output."
        "Do not provide the feedback to add examples with the prompt."
    )
    input_model = FeedbackMutationInput
    output_model = FeedbackMutationOutput


class FeedbackMutationPromptInput(BaseModel):
    prompt: str
    feedbacks: t.List[str]

class FeedbackMutationPromptGeneration(PydanticPrompt[FeedbackMutationPromptInput, Prompt]):
    name: str = "feedback_mutation_generation"
    instruction: str = (
        "You are a mutator. Given a prompt and a set of feedbacks on how the prompt can be improved generate a new prompt that incorporates the feedback."
    )
    input_model = FeedbackMutationPromptInput
    output_model = Prompt
    
    
    
class SingleInputOutput(BaseModel):
    input: str
    output: t.Dict[str, t.Any]
    
class ErrorAnalysisInput(BaseModel):
    prompt: str
    examples: t.List[SingleInputOutput]
    
class ErrorAnalysisOutput(BaseModel):
    feedbacks: t.List[str]
    
class ErrorAnalysisPrompt(PydanticPrompt[ErrorAnalysisInput, ErrorAnalysisOutput]):
    name: str = "error_analysis"
    instruction: str = (
        "You're an expert reviewer. Given a prompt and a set of (input  containing (user_input, response, reference), incorrect output) examples, analyze the patterns in the incorrect outputs and provide feedback on how the prompt can be improved to correct the mistakes in incorrect outputs."
        "Do not provide the feedback to add examples with the prompt."
    )
    input_model = ErrorAnalysisInput
    output_model = ErrorAnalysisOutput