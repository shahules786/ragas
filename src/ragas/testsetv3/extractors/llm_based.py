import typing as t
from dataclasses import dataclass

import numpy as np
import tiktoken

from ragas.llms.base import BaseRagasLLM, llm_factory
from ragas.llms.json_load import json_loader
from ragas.llms.prompt import Prompt
from ragas.testsetv3.extractors.base import Extractor
from ragas.testsetv3.extractors.prompts import (
    headline_extractor_prompt,
    keyphrase_extractor_prompt,
    summary_extactor_prompt,
    title_extractor_prompt,
)
from ragas.testsetv3.utils import MODEL_MAX_LENGTHS, merge_dicts


@dataclass
class LLMbasedExtractor(Extractor):
    prompt: Prompt
    llm: t.Optional[BaseRagasLLM] = None

    async def _generate_output(self, p_value, is_asycn=True):
        assert self.llm is not None, "LLM is not initialized"

        output = await self.llm.generate(prompt=p_value, is_async=is_asycn)
        output = output.generations[0][0].text.strip()
        if self.prompt.output_type == "json":
            return await json_loader.safe_load(output, self.llm)

        return {self.prompt.name: output}

    async def extract(self, text, is_asycn=True):
        if self.llm is None:
            self.llm = llm_factory()

        # TODO: handle different models
        model_name = "gpt-2"
        model_max_length = MODEL_MAX_LENGTHS.get(model_name, 8000)
        model_input_length = model_max_length - (model_max_length // 4)

        enc = tiktoken.encoding_for_model(model_name)
        p_value = self.prompt.format(text=text)
        tokens = enc.encode(p_value.to_string())
        prompt_length = len(tokens)
        ratio = prompt_length / model_input_length

        # TODO modify to suit abstractive tasks as well
        if ratio > 1:
            max_tokens_per_run = int(np.ceil(prompt_length / np.ceil(ratio)))
            inputs = [
                enc.decode(tokens[i : i + max_tokens_per_run])
                for i in range(0, len(tokens), max_tokens_per_run)
            ]
            inputs = [self.prompt.format(text=inp) for inp in inputs]
            outputs = [await self._generate_output(inp, is_asycn) for inp in inputs]
            output = merge_dicts(*outputs)

        else:
            output = await self._generate_output(p_value, is_asycn)

        return output

    def merge_extractors(self, *extractors):
        if isinstance(self, LLMbasedExtractor):
            extractors = (self,) + extractors

        final_extractors: t.List[t.List[LLMbasedExtractor]] = []
        added_indices = []

        extractors = list(extractors)
        for idx, extractor in enumerate(extractors):
            if idx not in added_indices:
                final_extractors.append([extractor])
                added_indices.append(idx)
                other_extractors = [
                    ext for i, ext in enumerate(extractors) if i not in added_indices
                ]
                input_keys = extractor.prompt.input_keys
                filtered_extractors = [
                    ext
                    for ext in other_extractors
                    if ext.prompt.input_keys == input_keys
                    and len(ext.prompt.examples) == len(extractor.prompt.examples)
                ]
                for ext in filtered_extractors:
                    input_values = [
                        ext.prompt.examples[i][key]
                        for i in range(len(ext.prompt.examples))
                        for key in input_keys
                    ]
                    if all(
                        extractor.prompt.examples[i][key] == input_values[i]
                        for i in range(len(ext.prompt.examples))
                        for key in input_keys
                    ):
                        final_extractors[-1].append(ext)
                        added_indices.append(extractors.index(ext))

        extractors_to_return = []
        for extractors in final_extractors:
            instruction = "\n".join(
                [
                    f"{i}:{extractor.prompt.instruction}"
                    for i, extractor in enumerate(extractors)
                ]
            )

            examples = []
            for idx, example in enumerate(extractors[0].prompt.examples):
                example = {key: example[key] for key in extractors[0].prompt.input_keys}
                output = {
                    extractor.prompt.output_key: extractor.prompt.examples[idx][
                        extractor.prompt.output_key
                    ]
                    for extractor in extractors
                }
                example.update({"output": output})
                examples.append(example)

            prompt = Prompt(
                name="merged_extractor",
                instruction=instruction,
                examples=examples,
                input_keys=extractors[0].prompt.input_keys,
                output_key="output",
                output_type="json",
            )
            extractors_to_return.append(LLMbasedExtractor(prompt=prompt))

        return extractors_to_return


summary_extractor = LLMbasedExtractor(prompt=summary_extactor_prompt)
headline_extractor = LLMbasedExtractor(prompt=headline_extractor_prompt)
keyphrase_extractor = LLMbasedExtractor(prompt=keyphrase_extractor_prompt)
title_extractor = LLMbasedExtractor(prompt=title_extractor_prompt)
