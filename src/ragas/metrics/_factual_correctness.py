from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ragas.metrics.base import (
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    get_segmenter,
)
from ragas.prompt import PydanticPrompt
from ragas.metrics.examples import ClaimDecompositionInput, ClaimDecompositionOutput, NLIInput, NLIOutput, nli_examples, new_claim_decomposition_examples

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.dataset_schema import SingleTurnSample


logger = logging.getLogger(__name__)


class ClaimDecompositionPrompt(
    PydanticPrompt[ClaimDecompositionInput, ClaimDecompositionOutput]
):
    instruction = """
    Decompose and break down each of the input sentences into one or more standalone factual statements. 
    Focus on the factual content of each sentence, and exclude any metaphors, analogies, or comparisons 
    that do not contribute directly to the factual claims. Ensure the output consists of concrete, verifiable statements.
    """
    input_model = ClaimDecompositionInput
    output_model = ClaimDecompositionOutput
    examples = new_claim_decomposition_examples
    

class NLIStatementPrompt(PydanticPrompt[NLIInput, NLIOutput]):
    instruction = "Given a context and a statement, determine if the statement can be inferred from context. Follow the examples to understand the level of inference required."
    input_model = NLIInput
    output_model = NLIOutput
    examples = nli_examples
    
    

@dataclass
class FactualCorrectness(MetricWithLLM, SingleTurnMetric):
    name: str = "factual_correctness"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
    )
    mode: t.Literal["precision", "recall", "f1"] = "f1"
    with_examples: bool = False
    claim_decomposition_prompt: PydanticPrompt = ClaimDecompositionPrompt()
    nli_prompt: PydanticPrompt = NLIStatementPrompt()

    def __post_init__(self):
        if not self.with_examples:
            self.claim_decomposition_prompt.examples = []
            self.nli_prompt.examples = []

        self.segmenter = get_segmenter(language="english")

    async def decompose_claims(
        self, response: str, callbacks: Callbacks
    ) -> t.List[str]:
        assert self.llm is not None, "LLM must be set"
        sentences = self.segmenter.segment(response)
        assert isinstance(sentences, list), "Segmenter must return a list of sentences"
        prompt_input = ClaimDecompositionInput(response=response, sentences=sentences)
        result = await self.claim_decomposition_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        claims_list = [
            claim for claim_list in result.decomposed_claims for claim in claim_list
        ]
        return claims_list

    async def verify_claims(
        self, premise: str, hypothesis_list: t.List[str], callbacks: Callbacks
    ) -> NDArray[np.bool_]:
        assert self.llm is not None, "LLM must be set"

        prompt_input = NLIInput(context=premise, statement=hypothesis_list)
        response = await self.nli_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        verdicts = [bool(output.verdict) for output in response.output]
        return np.array(verdicts)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        reference = sample.reference
        response = sample.response
        assert self.llm is not None, "LLM must be set"
        assert reference is not None, "Reference is not set"
        assert response is not None, "Response is not set"

        response_claims = await self.decompose_claims(response, callbacks)
        reference_claims = await self.decompose_claims(reference, callbacks)

        reference_response = await self.verify_claims(
            premise=reference, hypothesis_list=response_claims, callbacks=callbacks
        )
        response_reference = await self.verify_claims(
            premise=response, hypothesis_list=reference_claims, callbacks=callbacks
        )
        true_positives = sum(reference_response)
        false_positives = sum(~reference_response)
        false_negatives = sum(~response_reference)

        if self.mode == "precision":
            score = true_positives / (true_positives + false_positives + 1e-8)
        elif self.mode == "recall":
            score = true_positives / (true_positives + false_negatives + 1e-8)
        else:
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            score = 2 * (precision * recall) / (precision + recall + 1e-8)

        return np.round(score, 2)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
