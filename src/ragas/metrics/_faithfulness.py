from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field

from ragas.llms.base import llm_factory
from ragas.llms.ensembler import ensembler
from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import INFERENCE_LEVELS
from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.llms.base import BaseRagasLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue

logger = logging.getLogger(__name__)

class Statements(BaseModel):
    sentence_index: int = Field(..., description="Index of the sentence from the statement list")
    simpler_statements: t.List[str] = Field(..., description="the simpler statements")
    

class StatementsAnswers(BaseModel):
    __root__: t.List[Statements]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_statements_output_instructions = get_json_format_instructions(StatementsAnswers)
_statements_output_parser = RagasoutputParser(pydantic_object=StatementsAnswers)


LONG_FORM_ANSWER_PROMPT = Prompt(
    name="long_form_answer",
    output_format_instruction=_statements_output_instructions,
    instruction="Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON.",
    examples=[
        {
        "question": "Who was Albert Einstein and what is he best known for?",
        "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
        "sentences":"""
        0:He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. 
        1:He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
        """,
        "analysis":StatementsAnswers.parse_obj([
            {
      "sentence_index": 0,
      "simpler_statements": [
        "Albert Einstein was a German-born theoretical physicist.",
        "Albert Einstein is recognized as one of the greatest and most influential physicists of all time."
      ]
    },
    {
      "sentence_index": 1,
      "simpler_statements": [
        "Albert Einstein was best known for developing the theory of relativity.",
        "Albert Einstein also made important contributions to the development of the theory of quantum mechanics."
      ]
    }
            
        ]).dicts()

            
        }
    ],
    input_keys=["question","answer","sentences"],
    output_key="analysis")


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    inference_level: int = Field(..., description="the level of inference")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")



class StatementFaithfulnessAnswers(BaseModel):
    __root__: t.List[StatementFaithfulnessAnswer]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_faithfulness_output_instructions = get_json_format_instructions(
    StatementFaithfulnessAnswers
)
_faithfulness_output_parser = RagasoutputParser(
    pydantic_object=StatementFaithfulnessAnswers
)

NLI_STATEMENTS_MESSAGE = Prompt(
    name="nli_statements",
    instruction="Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.",
    output_format_instruction=_faithfulness_output_instructions,
    examples=[
        {
            "context": """John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
            "statements": [
                "John is majoring in Biology.",
                "John is taking a course on Artificial Intelligence.",
                "John is a dedicated student.",
                "John has a part-time job.",
            ],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "John is majoring in Biology.",
                        "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        "inference_level": 1,
                        "verdict": 0,

                    },
                    {
                        "statement": "John is taking a course on Artificial Intelligence.",
                        "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        "inference_level": 1,
                        "verdict": 0,

                    },
                    {
                        "statement": "John is a dedicated student.",
                        "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        "inference_level": 2,
                        "verdict": 1,

                    },
                    {
                        "statement": "John has a part-time job.",
                        "reason": "There is no information given in the context about John having a part-time job.",
                        "inference_level": 1,
                        "verdict": 0,

                    },
                ]
            ).dicts(),
        },
        {
            "context": """Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.""",
            "statements": ["Albert Einstein was a genius."],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "Albert Einstein was a genius.",
                        "reason": "The context and statement are unrelated",
                        "inference_level": 1,
                        "verdict": 0,

                    }
                ]
            ).dicts(),
        },
    ],
    input_keys=["context", "statements"],
    output_key="answer",
    output_type="json",
)  # noqa: E501


NLI_STATEMENTS_MESSAGE.instruction += INFERENCE_LEVELS

from ragas.metrics._context_relevancy import seg as sentence_segmenter
@dataclass
class TexttoStatements:
    
    statement_prompt: Prompt = field(
        default_factory=lambda: LONG_FORM_ANSWER_PROMPT
    )
    sentence_segmenter: t.Callable = sentence_segmenter
    
    llm: t.Optional[BaseRagasLLM] = None
    
    
    def __post_init__(self):
        
        if self.llm is None:
            self.llm = llm_factory()
                
    def _create_statement_prompt(self, question: str, text: str) -> PromptValue:
        
        sentences = sentence_segmenter.segment(text)
        sentences = [sentence for sentence in sentences if sentence.strip().endswith(".")]
        sentences = '\n'.join([f'{i}:{x}' for i, x in enumerate(sentences)])
        prompt_value = self.statement_prompt.format(
            question=question, answer=text, sentences=sentences
            
        )
        return prompt_value
    
    async def get_statements(self, question: str, text: str, **kwargs) -> t.List[str]:
        
        prompt_value = self._create_statement_prompt(question, text)
        answer_result = await self.llm.generate(
                prompt_value, callbacks=kwargs.get("callbacks", []), is_async=kwargs.get("is_async", True), n=kwargs.get("reproducibility", 1)
        )        
        statements = [await _statements_output_parser.aparse(
            result.text, prompt_value, self.llm, kwargs.get("max_retries", 1)
        ) for result in answer_result.generations[0]]
        
        statements = [statement.dicts() for statement in statements if statement is not None]
        is_equal_len = all(len(statements[0]) == len(item)for item in statements)
        if not is_equal_len:
            sentence_index = []
            for statement_list in statements:
                sentence_index.append([item['sentence_index'] for item in statement_list])
            common_indices = np.intersect1d(list(np.intersect1d(sentence_index[0], sentence_index[1])),sentence_index[2])
            statements = [[item for item in statement_list if item['sentence_index'] in common_indices] for statement_list in statements]
            
        output_statements = []
        if len(statements) > 1:
            for i in range(len(statements[0])):
                sentence_statements = [item[i]['simpler_statements'] for item in statements]
                sentence_statements = await ensembler.from_list_of_strings(sentence_statements)
                output_statements.extend(sentence_statements)
        else:
            output_statements = [item['simpler_statements'] for item in statements[0]]
            
        return output_statements
    
            

@dataclass
class Faithfulness(MetricWithLLM):
    name: str = "faithfulness"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qac  # type: ignore
    nli_statements_message: Prompt = field(
        default_factory=lambda: NLI_STATEMENTS_MESSAGE
    )
    text_to_statements: t.Optional[TexttoStatements] = None
    max_retries: int = 1
    reproducibility: int = 1
    
    def __post_init__(self):
        
        self.text_to_statements = TexttoStatements(llm=self.llm)

    def _create_nli_prompt(self, row: t.Dict, statements: t.List[str]) -> PromptValue:
        assert self.llm is not None, "llm must be set to compute score"

        contexts = row["contexts"]
        # check if the statements are support in the contexts
        contexts_str: str = "\n".join(contexts)
        statements_str: str = json.dumps(statements)
        prompt_value = self.nli_statements_message.format(
            context=contexts_str, statements=statements_str
        )
        return prompt_value

    def _compute_score(self, answers: StatementFaithfulnessAnswers):
        # check the verdicts and compute the score
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.__root__
        )
        num_statements = len(answers.__root__)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    async def _ascore(
        self: t.Self, row: t.Dict, callbacks: Callbacks, is_async: bool
    ) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"
        
        question, answer = row["question"], row["answer"]
        statements = await text_to_statements.get_statements(llm=self.llm ,question=question,text=answer,reproducibility=self.reproducibility,max_retries=self.max_retries,callbacks=callbacks,is_async=is_async)
      
        p_value = self._create_nli_prompt(row, statements)
        nli_result = await self.llm.generate(
            p_value,
            callbacks=callbacks,
            is_async=is_async,
            n=self.reproducibility,
        )

        if self.reproducibility > 1:
            nli_result_text = [
                nli_result.generations[0][i].text for i in range(self.reproducibility)
            ]
            faithfulness_list = [
                await _faithfulness_output_parser.aparse(
                    text, p_value, self.llm, self.max_retries
                )
                for text in nli_result_text
            ]
            faithfulness_list = [
                faith.dicts() for faith in faithfulness_list if faith is not None
            ]
            
            faithfulness_list = ensembler.from_discrete(
                faithfulness_list, "verdict",
            )
            faithfulness_list = StatementFaithfulnessAnswers.parse_obj(
                faithfulness_list
            )
        else:
            nli_result_text = nli_result.generations[0][0].text
            faithfulness_list = await _faithfulness_output_parser.aparse(
                nli_result_text, p_value, self.llm, self.max_retries
            )
            if faithfulness_list is None:
                return np.nan

        return self._compute_score(faithfulness_list)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "LLM is not set"
        assert self.text_to_statements is not None, "Text to statements is not set"
        
        logger.info(f"Adapting Faithfulness metric to {language}")
        
        self.nli_statements_message = self.nli_statements_message.adapt(
            language, self.llm, cache_dir
        )
        self.text_to_statements.statement_prompt = self.text_to_statements.statement_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        
        self.nli_statements_message.save(cache_dir)


faithfulness = Faithfulness()
text_to_statements = TexttoStatements()
