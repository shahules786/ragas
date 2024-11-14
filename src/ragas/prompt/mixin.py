from __future__ import annotations

import inspect
import logging
import os
from re import search
import typing as t
import numpy as np
import json

from openai import BaseModel

from ragas.embeddings.base import BaseRagasEmbeddings

from .base import _check_if_language_is_supported
from .pydantic_prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM


logger = logging.getLogger(__name__)


class PromptMixin:
    """
    Mixin class for classes that have prompts.
    eg: [BaseSynthesizer][ragas.testset.synthesizers.base.BaseSynthesizer], [MetricWithLLM][ragas.metrics.base.MetricWithLLM]
    """

    def get_prompts(self) -> t.Dict[str, PydanticPrompt]:
        """
        Returns a dictionary of prompts for the class.
        """
        prompts = {}
        for name, value in inspect.getmembers(self):
            if isinstance(value, PydanticPrompt):
                prompts.update({name: value})
        return prompts

    def set_prompts(self, **prompts):
        """
        Sets the prompts for the class.

        Raises
        ------
        ValueError
            If the prompt is not an instance of `PydanticPrompt`.
        """
        available_prompts = self.get_prompts()
        for key, value in prompts.items():
            if key not in available_prompts:
                raise ValueError(
                    f"Prompt with name '{key}' does not exist. Use get_prompts() to see available prompts."
                )
            if not isinstance(value, PydanticPrompt):
                raise ValueError(
                    f"Prompt with name '{key}' must be an instance of 'ragas.prompt.PydanticPrompt'"
                )
            setattr(self, key, value)

    async def adapt_prompts(
        self, language: str, llm: BaseRagasLLM, adapt_instruction: bool = False
    ) -> t.Dict[str, PydanticPrompt]:
        """
        Adapts the prompts in the class to the given language and using the given LLM.

        Notes
        -----
        Make sure you use the best available LLM for adapting the prompts and then save and load the prompts using
        [save_prompts][ragas.prompt.mixin.PromptMixin.save_prompts] and [load_prompts][ragas.prompt.mixin.PromptMixin.load_prompts]
        methods.
        """
        prompts = self.get_prompts()
        adapted_prompts = {}
        for name, prompt in prompts.items():
            adapted_prompt = await prompt.adapt(language, llm, adapt_instruction)
            adapted_prompts[name] = adapted_prompt

        return adapted_prompts

    def save_prompts(self, path: str):
        """
        Saves the prompts to a directory in the format of {name}_{language}.json
        """
        # check if path is valid
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        prompts = self.get_prompts()
        for prompt_name, prompt in prompts.items():
            # hash_hex = f"0x{hash(prompt) & 0xFFFFFFFFFFFFFFFF:016x}"
            prompt_file_name = os.path.join(
                path, f"{prompt_name}_{prompt.language}.json"
            )
            prompt.save(prompt_file_name)

    def load_prompts(self, path: str, language: t.Optional[str] = None):
        """
        Loads the prompts from a path. File should be in the format of {name}_{language}.json
        """
        # check if path is valid
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        # check if language is supported, defaults to english
        if language is None:
            language = "english"
            logger.info(
                "Language not specified, loading prompts for default language: %s",
                language,
            )
        _check_if_language_is_supported(language)

        loaded_prompts = {}
        for prompt_name, prompt in self.get_prompts().items():
            prompt_file_name = os.path.join(path, f"{prompt_name}_{language}.json")
            loaded_prompt = prompt.__class__.load(prompt_file_name)
            loaded_prompts[prompt_name] = loaded_prompt
        return loaded_prompts
    
    async def retrieve_few_shot_examples(self, prompt_input: BaseModel, embedding_model: BaseRagasEmbeddings, index_path: str, top_k: int = 5, search="similarity") -> t.Any:
        
        metric_prompts = self.get_prompts()
        
        prompt_name = None
        input_model, output_model = None, None
        for key, val in metric_prompts.items():
            if val.input_model == type(prompt_input):
                prompt_name = key
                input_model = val.input_model
                output_model = val.output_model
                break
            
        if prompt_name is None:
            raise ValueError(f"Prompt with input model {prompt_input} not found")
        
        pos_examples, neg_examples = [], []

        train_inputs = json.load(open(index_path.replace("npy","json")))
        if search == "similarity":
            train_vectors = np.load(index_path)
            test_vector = await embedding_model.aembed_query(prompt_input.user_input)
            test_vector = np.array(test_vector)
            similarities = np.dot(train_vectors, test_vector) / (
                np.linalg.norm(train_vectors, axis=1) * np.linalg.norm(test_vector)
            )
            most_similar_indices = np.argsort(similarities)[::-1]
            most_similar_indices = most_similar_indices[:top_k]
            
            for idx in most_similar_indices:
                if len(pos_examples+neg_examples) < top_k:    
                    example = train_inputs[idx]
                    input_example = input_model(**example["input"])
                    output_example = output_model(**example["output"][0])
                    if example["qdrant"] in ["TN", "TP"]:    
                        pos_examples.append((input_example, output_example))
                    # elif example["qdrant"] in ["FN", "FP"]:
                    #     neg_examples.append((input_example, output_example))
                    else:
                        pass
        elif search == "random":
            np.random.seed(seed=42)
            examples = np.random.choice(train_inputs, top_k, replace=False)
            for example in examples:
                input_example = input_model(**example["input"])
                output_example = output_model(**example["output"][0])
                if example["qdrant"] in ["TN", "TP"]:    
                    pos_examples.append((input_example, output_example))
                # elif example["qdrant"] in ["FN","FP"]:
                #     neg_examples.append((input_example, output_example))
                else:
                    pass
        else:
            raise ValueError(f"Search method {search} not supported")
            
                    
                          
        return pos_examples, neg_examples
