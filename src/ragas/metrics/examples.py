

import stat
from turtle import st
from pydantic import BaseModel, Field
import typing as t
from enum import Enum

from regex import F


class ClaimDecompositionInput(BaseModel):
    response: str = Field(..., title="Response")
    sentences: t.List[str] = Field(..., title="Sentences from response")


class ClaimDecompositionOutput(BaseModel):
    decomposed_claims: t.List[t.List[str]] = Field(..., title="Decomposed Claims")


class SingleNLIOutput(BaseModel):
    reason: str = Field(..., description="the reason of the verdict")
    verdict: bool = Field(..., description="True if statement can be inferred from context, False otherwise")


class NLIOutput(BaseModel):
    output: t.List[SingleNLIOutput] = Field(..., description="The output of the NLI model")


class NLIInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statement: t.List[str] = Field(..., description="The statement to judge")


random_nli_examples = [
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


random_claim_decomposition_examples = [
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




# Define the examples using the new structure
new_claim_decomposition_examples = [
    # Example 2
    (
        ClaimDecompositionInput(
            response="Furthermore, the wheel's influence extended to various technologies, including pottery wheels, which enhanced the production of ceramics.",
            sentences=[
                "Furthermore, the wheel's influence extended to various technologies, including pottery wheels, which enhanced the production of ceramics."
            ],
        ),
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["The wheel's influence extended to technologies like pottery wheels that improved ceramic production."]
            ]
        ),
    ),
        # Example 1
    (
        ClaimDecompositionInput(
            response="Before modern medicine, treating infections was as difficult as navigating through a maze without a map.",
            sentences=[
                "Before modern medicine, treating infections was as difficult as navigating through a maze without a map."
            ],
        ),
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Treating infections before modern medicine was difficult."]
            ]
        ),
    ),
    
    # Example 2
    (
        ClaimDecompositionInput(
            response="Before airplanes, traveling across the ocean was a grueling and time-consuming journey, like trying to cross a desert with no water.",
            sentences=[
                "Before airplanes, traveling across the ocean was a grueling and time-consuming journey, like trying to cross a desert with no water."
            ],
        ),
        ClaimDecompositionOutput(
            decomposed_claims=[
                ["Traveling across the ocean before airplanes was grueling and time-consuming."]
            ]
        ),
    ),
    
    (
    ClaimDecompositionInput(
        response="**Safety from Predators:** The glow and heat of fire deterred predators, offering a sense of security and enabling humans to settle in new areas.",
        sentences=[
            "The glow and heat of fire deterred predators, offering a sense of security and enabling humans to settle in new areas."
        ],
    ),
    ClaimDecompositionOutput(
        decomposed_claims=[
            ["Fire deterred predators and provided a sense of security.", "Fire enabled humans to settle in new areas."]
        ]
    ),
),
    (
    ClaimDecompositionInput(
        response="### Social Tensions and Persecution\n- **Scapegoating and Persecution:** In the chaos and fear, minority groups, particularly Jews, were often scapegoated and persecuted, accused of causing the plague.",
        sentences=[
            "In the chaos and fear, minority groups, particularly Jews, were often scapegoated and persecuted, accused of causing the plague."
        ],
    ),
    ClaimDecompositionOutput(
        decomposed_claims=[
            ["Minority groups, especially Jews, were scapegoated and persecuted during the plague.", "They were falsely accused of causing the plague."]
        ]
    ),
),
    
    
]




nli_examples = [
    (
        NLIInput(
            context="The French Revolution, which began in 1789, was primarily caused by a combination of social, economic, and political factors. Socially, the rigid class structure and the privileges of the nobility and clergy created widespread discontent among the common people.",
            statement=["France was divided into three estates: the clergy, the nobility, and the common people."],
        ),
        NLIOutput(output=[SingleNLIOutput(
            reason="The context mentions the rigid class structure and the privileges of the nobility and clergy from which we can safely infer with common world knowledge that France was divided into three estates.",
            verdict=True,
        )])
    ),
    (
        NLIInput(
            context="With the wheel, carts and chariots could be used to transport goods and people more efficiently, facilitating trade over longer distances and promoting cultural exchange.",
            statement=["Trade routes connected distant lands."],
        ),
        NLIOutput(output=[SingleNLIOutput(
            reason="The context mentions that the wheel facilitated trade over longer distances, which aligns with the statement. Hence, the statement can be inferred from the context.",
            verdict=True,
        )])
    ),
    (
        NLIInput(
            context="In 1492, Christopher Columbus set sail from Spain and reached the Americas, an event that profoundly impacted the course of world history.",
            statement=["Christopher Columbus discovered Australia in 1492."],
        ),
        NLIOutput(output=[SingleNLIOutput(
            reason="The context clearly refers to Columbus reaching the Americas, not Australia. The statement is contradicts the information given in the context, hence False.",
            verdict=False,
        )])
    )
]
