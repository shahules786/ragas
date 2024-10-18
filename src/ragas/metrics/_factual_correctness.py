from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ragas.metrics.base import (
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    get_segmenter,
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.dataset_schema import SingleTurnSample


logger = logging.getLogger(__name__)


class ClaimDecompositionInput(BaseModel):
    response: str = Field(..., title="Response")
    sentences: t.List[str] = Field(..., title="Sentences from response")


class ClaimDecompositionOutput(BaseModel):
    decomposed_claims: t.List[t.List[str]] = Field(..., title="Decomposed Claims")


class ClaimDecompositionPrompt(
    PydanticPrompt[ClaimDecompositionInput, ClaimDecompositionOutput]
):
    instruction = """
    Decompose and break down each of the input sentences into one or more standalone statements.
    """
    input_model = ClaimDecompositionInput
    output_model = ClaimDecompositionOutput
    exampes = [
        # Example 1
        (
            ClaimDecompositionInput(
                response="John went to the store and bought some milk.",
                sentences=["John went to the store and bought some milk."],
            ),
            ClaimDecompositionOutput(
                decomposed_claims=[
                    ["John went to the store.", "John bought some milk."]
                ]
            ),
        ),
        # Example 2
        (
            ClaimDecompositionInput(
                response="Alice loves painting, and she has a gallery exhibition next week.",
                sentences=[
                    "Alice loves painting, and she has a gallery exhibition next week."
                ],
            ),
            ClaimDecompositionOutput(
                decomposed_claims=[
                    ["Alice loves painting.", "She has a gallery exhibition next week."]
                ]
            ),
        ),
        # Example 3
        (
            ClaimDecompositionInput(
                response="The weather was terrible, so the football match was postponed.",
                sentences=[
                    "The weather was terrible, so the football match was postponed."
                ],
            ),
            ClaimDecompositionOutput(
                decomposed_claims=[
                    ["The weather was terrible.", "The football match was postponed."]
                ]
            ),
        ),
    ]


class SingleNLIOutput(BaseModel):
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")

class NLIOutput(BaseModel):
    output: t.List[SingleNLIOutput] = Field(..., description="The output of the NLI model")

class NLIInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statement: t.List[str] = Field(..., description="The statement to judge")


class NLIStatementPrompt(PydanticPrompt[NLIInput, NLIOutput]):
    instruction = "Given a context and a statement, determine if the statement can be inferred from context."
    input_model = NLIInput
    output_model = NLIOutput
    examples = [
        (
            NLIInput(
                context="The Eiffel Tower is located in Paris, France. It was built in 1889.",
                statement=[
                    "The Eiffel Tower is in Berlin.",
                    "The Eiffel Tower was constructed in the 19th century."
                ]
            ),
            NLIOutput(
                output=[
                    SingleNLIOutput(
                        reason="The context states that the Eiffel Tower is in Paris, not Berlin.",
                        verdict=0
                    ),
                    SingleNLIOutput(
                        reason="The context mentions it was built in 1889, which is in the 19th century.",
                        verdict=1
                    )
                ]
            )
        ),
        # Second Example
        (
            NLIInput(
                context="Python is a high-level programming language known for its readability and support for multiple programming paradigms.",
                statement=[
                    "Python is a low-level language.",
                    "Python supports object-oriented programming."
                ]
            ),
            NLIOutput(
                output=[
                    SingleNLIOutput(
                        reason="The context states that Python is a high-level language, not a low-level one.",
                        verdict=0
                    ),
                    SingleNLIOutput(
                        reason="Since Python supports multiple programming paradigms and OOP is one of them, this can be inferred.",
                        verdict=1
                    )
                ]
            )
        ),
        # Third Example
        (
            NLIInput(
                context="The Great Wall of China was built to protect Chinese states against invasions and raids. It is one of the most impressive architectural feats in history.",
                statement=[
                    "The Great Wall of China was constructed for trade purposes.",
                    "The Great Wall is considered an impressive architectural achievement."
                ]
            ),
            NLIOutput(
                output=[
                    SingleNLIOutput(
                        reason="The context mentions it was built to protect against invasions, not for trade.",
                        verdict=0
                    ),
                    SingleNLIOutput(
                        reason="The context states it is one of the most impressive architectural feats in history.",
                        verdict=1
                    )
                ]
            )
        )
    ]


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
