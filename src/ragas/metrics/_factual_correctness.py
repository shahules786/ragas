from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from enum import Enum

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


# Define an enum for decomposition types
class DecompositionType(Enum):
    LOW_ATOMICITY_LOW_COVERAGE = "low_atomicity_low_coverage"
    LOW_ATOMICITY_HIGH_COVERAGE = "low_atomicity_high_coverage"
    HIGH_ATOMICITY_LOW_COVERAGE = "high_atomicity_low_coverage"
    HIGH_ATOMICITY_HIGH_COVERAGE = "high_atomicity_high_coverage"


# Example input data
example1_input = ClaimDecompositionInput(
    response="Charles Babbage was a French mathematician, philosopher, and food critic.",
    sentences=[
        "Charles Babbage was a French mathematician, philosopher, and food critic."
    ],
)

# Define the examples using the new structure
claim_decomposition_examples = {
    DecompositionType.LOW_ATOMICITY_LOW_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                decomposed_claims=[
                    ["Charles Babbage was a mathematician and philosopher."]
                ]
            ),
        )
    ],
    DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                decomposed_claims=[
                    [
                        "Charles Babbage was a French mathematician, philosopher, and food critic."
                    ]
                ]
            ),
        )
    ],
    DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                decomposed_claims=[
                    ["Charles Babbage was a mathematician."],
                    ["Charles Babbage was a philosopher."],
                ]
            ),
        )
    ],
    DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE: [
        (
            example1_input,
            ClaimDecompositionOutput(
                decomposed_claims=[
                    ["Charles Babbage was a mathematician."],
                    ["Charles Babbage was a philosopher."],
                    ["Charles Babbage was a food critic."],
                    ["Charles Babbage was French."],
                ]
            ),
        )
    ],
}

# Example input data with two sentences
example2_input = ClaimDecompositionInput(
    response="Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics.",
    sentences=[
        "Albert Einstein was a German theoretical physicist.",
        "He developed the theory of relativity and also contributed to the development of quantum mechanics.",
    ],
)

# Adding examples to the dictionary with different decomposition types
claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_LOW_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Albert Einstein was a German physicist."],
                [
                    "Albert Einstein developed relativity and contributed to quantum mechanics."
                ],
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.LOW_ATOMICITY_HIGH_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Albert Einstein was a German theoretical physicist."],
                [
                    "Albert Einstein developed the theory of relativity and also contributed to the development of quantum mechanics."
                ],
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_LOW_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Albert Einstein was a German theoretical physicist."],
                ["Albert Einstein developed the theory of relativity."],
            ]
        ),
    )
)

claim_decomposition_examples[DecompositionType.HIGH_ATOMICITY_HIGH_COVERAGE].append(
    (
        example2_input,
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Albert Einstein was a German theoretical physicist."],
                [
                    "Albert Einstein developed the theory of relativity.",
                    "Albert Einstein contributed to the development of quantum mechanics.",
                ],
            ]
        ),
    )
)


class ClaimDecompositionPrompt(
    PydanticPrompt[ClaimDecompositionInput, ClaimDecompositionOutput]
):
    instruction = """
    Decompose and break down each of the input sentences into one or more standalone statements. Each statement should be a standalone claim that can be independently verified.
    Follow the level of atomicity and coverage as shown in the examples.
    """
    input_model = ClaimDecompositionInput
    output_model = ClaimDecompositionOutput


class NLIOutput(BaseModel):
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class NLIInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statement: str = Field(..., description="The statement to judge")


class NLIStatementPrompt(PydanticPrompt[NLIInput, NLIOutput]):
    instruction = "Given a context and a statement, determine if the statement can be inferred from context. Follow the level of inference as shown in the examples."
    input_model = NLIInput
    output_model = NLIOutput
    examples = []


class InferenceType(Enum):
    LEVEL_1 = "1"
    LEVEL_2 = "2"
    LEVEL_3 = "3"


example1_input = NLIInput(
    context="Maria put on her running shoes and went outside.",
    statement="Maria is going for a run",
)

nli_examples = {
    InferenceType.LEVEL_1: [
        (
            example1_input,
            NLIOutput(
                reason="The context mentions that Maria put on her running shoes and went outside but does not explicitly state that she is going for a run.",
                verdict=0,
            ),
        ),
    ],
    InferenceType.LEVEL_2: [
        (
            example1_input,
            NLIOutput(
                reason="It is reasonable to infer that Maria is going for a run because she put on running shoes and went outside, actions commonly associated with preparing to run.",
                verdict=1,
            ),
        ),
    ],
    InferenceType.LEVEL_3: [
        (
            example1_input,
            NLIOutput(
                reason="Based on common practices and world knowledge, putting on running shoes and going outside strongly suggests that Maria is going for a run, as running shoes are specifically designed for that activity.",
                verdict=1,
            ),
        ),
    ],
}


example3_input = NLIInput(
    context="Sarah is wearing a ring on her left hand's fourth finger.",
    statement="Sarah is married.",
)


# For Level 1: Strict Literal Entailment (No Leap of Faith)
nli_examples[InferenceType.LEVEL_1].append(
    (
        example3_input,
        NLIOutput(
            reason="The context states that Sarah is wearing a ring on her left hand's fourth finger but does not explicitly mention that she is married.",
            verdict=0,
        ),
    )
)

# For Level 2: Moderate Inference (Some Leap of Faith)
nli_examples[InferenceType.LEVEL_2].append(
    (
        example3_input,
        NLIOutput(
            reason="While Sarah wearing a ring may imply a relationship, it is not a direct inference that she is married without additional context.",
            verdict=0,
        ),
    )
)

# For Level 3: Deep Inference and World Knowledge (Significant Leap of Faith)
nli_examples[InferenceType.LEVEL_3].append(
    (
        example3_input,
        NLIOutput(
            reason="Based on cultural norms and world knowledge, wearing a ring on the fourth finger of the left hand signifies marriage, so it's strongly suggested that Sarah is married.",
            verdict=1,
        ),
    )
)


@dataclass
class FactualCorrectness(MetricWithLLM, SingleTurnMetric):
    name: str = "factual_correctness"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
    )
    mode: t.Literal["precision", "recall", "f1"] = "f1"
    atomicity: t.Literal["low", "high"] = "low"
    coverage: t.Literal["low", "high"] = "high"
    inference_level: t.Literal["1", "2", "3"] = "3"
    claim_decomposition_prompt: PydanticPrompt = ClaimDecompositionPrompt()
    nli_prompt: PydanticPrompt = NLIStatementPrompt()

    def __post_init__(self):
        value = f"{self.atomicity}_atomicity_{self.coverage}_coverage"
        for item in DecompositionType:
            if item.value == value:
                self.claim_decomposition_prompt.examples.extend(
                    claim_decomposition_examples[item]
                )
        if not self.claim_decomposition_prompt.examples:
            logger.warning(
                f"No examples found for the atomicity and coverage level: {value}"
            )

        for item in InferenceType:
            if item.value == self.inference_level:
                self.nli_prompt.examples.extend(nli_examples[item])

        if not self.nli_prompt.examples:
            logger.warning(
                f"No examples found for the inference level: {self.inference_level}"
            )
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

        verdicts = []
        for hypothesis in hypothesis_list:
            prompt_input = NLIInput(context=premise, statement=hypothesis)
            response = await self.nli_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )
            verdicts.append(bool(response.verdict))
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
