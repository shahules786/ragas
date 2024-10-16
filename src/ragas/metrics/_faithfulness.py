from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    get_segmenter,
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


class HasSegmentMethod(t.Protocol):
    def segment(self, text) -> t.Any:
        ...


logger = logging.getLogger(__name__)


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
    LEVEL_1 = "LEVEL_1"
    LEVEL_2 = "LEVEL_2"
    LEVEL_3 = "LEVEL_3"


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
class Faithfulness(MetricWithLLM, SingleTurnMetric):
    name: str = "faithfulness"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "retrieved_contexts",
            }
        }
    )
    nli_statements_message: PydanticPrompt = field(default_factory=NLIStatementPrompt)
    statement_prompt: PydanticPrompt = field(default_factory=LongFormAnswerPrompt)
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    max_retries: int = 1
    _reproducibility: int = 1

    @property
    def reproducibility(self):
        return self._reproducibility

    @reproducibility.setter
    def reproducibility(self, value):
        if value < 1:
            logger.warning("reproducibility cannot be less than 1, setting to 1")
            value = 1
        elif value % 2 == 0:
            logger.warning(
                "reproducibility level cannot be set to even number, setting to odd"
            )
            value += 1
        self._reproducibility = value

    def __post_init__(self):
        if self.sentence_segmenter is None:
            language = self.nli_statements_message.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    async def _create_verdicts(
        self, row: t.Dict, statements: t.List[str], callbacks: Callbacks
    ) -> NLIStatementOutput:
        assert self.llm is not None, "llm must be set to compute score"

        contexts_str: str = "\n".join(row["retrieved_contexts"])
        verdicts = await self.nli_statements_message.generate(
            data=NLIStatementInput(context=contexts_str, statements=statements),
            llm=self.llm,
            callbacks=callbacks,
        )

        return verdicts

    async def _create_statements(
        self, row: t.Dict, callbacks: Callbacks
    ) -> SentencesSimplified:
        assert self.llm is not None, "llm is not set"
        assert self.sentence_segmenter is not None, "sentence_segmenter is not set"

        text, question = row["response"], row["user_input"]
        sentences = self.sentence_segmenter.segment(text)
        sentences_with_index = {
            i: sentence
            for i, sentence in enumerate(sentences)
            if sentence.strip().endswith(".")
        }

        statements_simplified = await self.statement_prompt.generate(
            llm=self.llm,
            data=FaithfulnessStatements(
                question=question, answer=text, sentences=sentences_with_index
            ),
            callbacks=callbacks,
        )
        return statements_simplified

    def _compute_score(self, answers: NLIStatementOutput):
        # check the verdicts and compute the score
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.statements
        )
        num_statements = len(answers.statements)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self: t.Self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements_simplified = await self._create_statements(row, callbacks)
        if statements_simplified is None:
            return np.nan

        # unwrap the statements
        statements = []
        for component in statements_simplified.sentences:
            statements.extend(component.simpler_statements)

        verdicts = await self._create_verdicts(row, statements, callbacks)
        return self._compute_score(verdicts)


@dataclass
class FaithfulnesswithHHEM(Faithfulness):
    name: str = "faithfulness_with_hhem"  # type: ignore
    device: str = "cpu"
    batch_size: int = 10

    def __post_init__(self):
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "Huggingface transformers must be installed to use this feature, try `pip install transformers`"
            )
        self.nli_classifier = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        )
        self.nli_classifier.to(self.device)
        super().__post_init__()

    def _create_pairs(
        self, row: t.Dict, statements: t.List[str]
    ) -> t.List[t.Tuple[str, str]]:
        """
        create pairs of (question, answer) from the row
        """
        premise = "\n".join(row["retrieved_contexts"])
        pairs = [(premise, statement) for statement in statements]
        return pairs

    def _create_batch(
        self, pairs: t.List[t.Tuple[str, str]]
    ) -> t.Generator[t.List[t.Tuple[str, str]], None, None]:
        length_of_pairs = len(pairs)
        for ndx in range(0, length_of_pairs, self.batch_size):
            yield pairs[ndx : min(ndx + self.batch_size, length_of_pairs)]

    async def _ascore(self: t.Self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements_simplified = await self._create_statements(row, callbacks)
        if statements_simplified is None:
            return np.nan

        statements = []
        for components in statements_simplified.sentences:
            statements.extend(components.simpler_statements)
        assert isinstance(statements, t.List), "statements must be a list"

        scores = []
        pairs = self._create_pairs(row, statements)
        for input_pairs in self._create_batch(pairs):  # to avoid OOM
            batch_scores = (
                self.nli_classifier.predict(input_pairs).cpu().detach().round()
            )
            scores += batch_scores
        return sum(scores) / len(scores)


faithfulness = Faithfulness()
