from ragas.prompt import PydanticPrompt
from pydantic import BaseModel


class CreateQuestion(BaseModel):
    term: str
    answer_length: str


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    
    
class ModifyAnswer(BaseModel):
    original_answer: str
    action: str  
class ModifiedAnswer(BaseModel):
    new_answer: str


class QuestionAnswerPrompt(PydanticPrompt[CreateQuestion, QuestionAnswer]):
    instruction: str = (
        "Given a seed term, generate a question that expects an answer that fits the specified answer_length."
    )
    input_model = CreateQuestion
    output_model = QuestionAnswer
    examples = [
        (
            CreateQuestion(
                term="thermodynamics",
                answer_length="long (50+ words)",
            ),
            QuestionAnswer(
                question="What are the fundamental laws of thermodynamics?",
                answer="""
The fundamental laws of thermodynamics are:
1. The Zeroth Law: It establishes the concept of temperature by stating that if two systems are each in thermal equilibrium with a third system, they are in thermal equilibrium with each other.
2. The First Law: It introduces the principle of conservation of energy, stating that energy cannot be created or destroyed, only transformed.
3. The Second Law: It establishes the direction of thermodynamic processes, stating that entropy, or disorder, always increases in an isolated system.
4. The Third Law: It postulates that as a system approaches absolute zero, the entropy of the system approaches a minimum value.
"""
            )
        ),
    ]
    
    
class ResponseTransformationPrompt(PydanticPrompt[ModifyAnswer, ModifiedAnswer]):
    instruction: str = (
        "Given an original answer and two transformation actions to perform—a format transformation and a style transformation—modify the original answer accordingly."
        "Apply both transformations to the answer."
        "Do not change the content of the answer beyond what's necessary for the transformations."
    )
    input_model = ModifyAnswer
    output_model = ModifiedAnswer
    examples = [
        (
            ModifyAnswer(
                original_answer=(
                    "The invention of the printing press in the 15th century revolutionized the way information was disseminated. "
                    "Books could be mass-produced, making them more accessible to the general public."
                ),
                action=(
                    "Format Transformation: Convert between narrative paragraphs and bullet points or vice versa.\n"
                    "Style Transformation: Modify the answer by adding hedging language to make some statements less certain."
                )
            ),
            ModifiedAnswer(
                new_answer=(
                    "- The invention of the printing press in the 15th century might have revolutionized the way information was disseminated.\n"
                    "- Books could be mass-produced, possibly making them more accessible to the general public."
                )
            )
        ),
        (
            ModifyAnswer(
                original_answer=(
                    "Cells are the basic building blocks of all living organisms. They provide structure for the body, take in nutrients from food, and carry out important functions."
                ),
                action=(
                    "Format Transformation: Organize the content using headings and subheadings.\n"
                    "Style Transformation: Introduce ambiguity into the answer by making some statements less precise."
                )
            ),
            ModifiedAnswer(
                new_answer=(
                    "**Cells: The Building Blocks of Life**\n\n"
                    "**Structure and Function**\n"
                    "Cells are considered the basic units of living organisms. They might provide structure for the body, take in nutrients from food, and carry out various functions."
                )
            )
        )
    ]
    
    
    
class ErrorInductionPrompt(PydanticPrompt[ModifyAnswer, ModifiedAnswer]):
    instruction: str = (
        "Given an original answer and an error induction action to perform, modify the original answer accordingly. "
        "Do not introduce more than one or two errors, and feel free to rewrite or reorder the flow of information to subtly alter the structure of the response."
    )
    input_model = ModifyAnswer
    output_model = ModifiedAnswer
    examples = [
        (
            ModifyAnswer(
                original_answer=(
                    "The signing of the Treaty of Versailles in 1919 officially ended World War I. "
                    "The treaty imposed heavy reparations on Germany, restricted its military, and redrew borders across Europe. "
                    "This treaty was a major factor in the economic and political instability in Germany that led to World War II."
                ),
                action="Subtly introduce an ambiguous claim and reorder the flow of information",
            ),
            ModifiedAnswer(
                new_answer=(
                    "The Treaty of Versailles, signed in 1919, officially ended World War I, but not all of its terms were clear. "
                    "The treaty redrew borders across Europe, restricted Germany's military, and imposed heavy reparations on Germany. "
                    "These reparations and military restrictions were a major factor in the economic instability and unrest in Germany, which later contributed to World War II."
                )
            ),
        ),
        (
            ModifyAnswer(
                original_answer=(
                    "The French Revolution, which began in 1789, was a period of significant social and political upheaval in France. "
                    "It led to the overthrow of the monarchy and the establishment of a republic. "
                    "The revolution was marked by the rise of radical groups like the Jacobins and significant violence during the Reign of Terror."
                ),
                action="Subtly alter a factual statement and change the sequence of events",
            ),
            ModifiedAnswer(
                new_answer=(
                    "Beginning in 1789, the French Revolution was a period of political and social turmoil, which eventually led to the overthrow of the monarchy. "
                    "While initially aimed at creating a republic, the revolution saw the rise of radical groups like the Jacobins, and the establishment of a democratic republic. "
                    "This was accompanied by widespread violence, especially during the Reign of Terror."
                )
            ),
        ),
        (
            ModifyAnswer(
                original_answer=(
                    "The Industrial Revolution began in Britain in the late 18th century and spread to other parts of the world. "
                    "It was characterized by the transition from hand production methods to machines and the rise of factory-based production. "
                    "Key inventions such as the steam engine played a critical role in driving this transformation."
                ),
                action="Omit a key fact and change the flow of the answer",
            ),
            ModifiedAnswer(
                new_answer=(
                    "In the late 18th century, Britain experienced the beginning of the Industrial Revolution, which later spread globally. "
                    "The period saw a shift from manual labor to machines, particularly in factories. "
                    "This transformation was largely driven by new technologies and production methods."
                )
            ),
        ),
    ]
    

# # Define the input model
# class RewriteInput(BaseModel):
#     question: str
#     reference_answer: str

# # Define the output model with the rewritten answer
# class RewrittenQuestion(BaseModel):
#     rewritten_answer: str

# # Define the prompt class
# class RewritePrompt(PydanticPrompt[RewriteInput, RewrittenQuestion]):
#     instruction: str = (
#         "Rewrite the provided reference answer using headings, bullet points, and other formatting to make it more structured and engaging. "
#         "Incorporate subjective interpretations, inferences from world knowledge, analogical explanations, and emotional or dramatic framing. "
#         "You Should also modernize comparisons and simplify complex ideas, while maintaining factual accuracy. "
#         "Feel free to extend or shorten the information as needed to improve clarity and impact."
#     )
#     input_model = RewriteInput
#     output_model = RewrittenQuestion
#     examples = [
#         (
#             RewriteInput(
#                 question="What were the main institutions of Athenian democracy, and how did they function?",
#                 reference_answer=(
#                     "The Athenian democracy was characterized by institutions such as the Assembly, where citizens could vote on laws and policies, and the Council of 500, which set the agenda for the Assembly. "
#                     "The Assembly was the central decision-making body where all eligible citizens could participate, debate, and vote on various issues. "
#                     "The Council of 500, selected by lot, was responsible for preparing the topics to be discussed in the Assembly, ensuring an organized and efficient legislative process. "
#                     "Additionally, other institutions like the Courts and various public offices played crucial roles in maintaining the democratic framework, promoting accountability, and enabling citizen involvement in governance."
#                 )
#             ),
#             RewrittenQuestion(
#                 rewritten_answer=(
#                     "# The Main Institutions of Athenian Democracy\n"
#                     "Athenian democracy was a system driven by the active involvement of its citizens. Here's how it worked:\n\n"
#                     "## The Assembly\n"
#                     "- **What it did:** The Assembly was like a large town hall meeting where citizens gathered to debate and vote on laws and policies. Every eligible citizen had a voice.\n"
#                     "- **How it functioned:** Think of it as a forum where ideas collided, and decisions were made collectively. It was the heart of Athenian democracy, allowing for direct participation.\n\n"
#                     "## The Council of 500\n"
#                     "- **Role:** Behind the scenes, the Council of 500 acted like a modern-day executive committee, setting the agenda for what would be discussed in the Assembly.\n"
#                     "- **Chosen by lot:** Council members were randomly selected to ensure fairness, much like drawing names from a hat. They kept the democratic process organized and focused.\n\n"
#                     "## The Courts and Public Offices\n"
#                     "- **Function:** These institutions ensured accountability, much like today’s judiciary and public offices that uphold justice. They safeguarded the integrity of the system, promoting fairness and transparency.\n\n"
#                     "Together, these institutions formed the backbone of Athenian democracy, ensuring that citizens were actively engaged in their governance."
#                 )
#             )
#         ),
#         (
#             RewriteInput(
#                 question="How did the birth of democracy in Athens shape the political landscape of ancient Greece?",
#                 reference_answer=(
#                     "The birth of democracy in Athens, around the 5th century BCE, marked a significant turning point in the political landscape of ancient Greece. "
#                     "It was initiated by reforms attributed to leaders like Cleisthenes, who is often called the 'Father of Athenian Democracy.' These reforms dismantled the power of aristocratic families and established a system where citizens could participate directly in decision-making processes. "
#                     "The Athenian democracy was characterized by institutions such as the Assembly, where citizens could vote on laws and policies, and the Council of 500, which set the agenda for the Assembly. "
#                     "This system allowed for greater political participation and accountability, fostering a sense of civic responsibility among citizens. "
#                     "The democratic principles developed in Athens influenced political thought and systems in subsequent civilizations, laying the groundwork for modern democratic governance. "
#                     "Despite its limitations, such as excluding women, slaves, and non-citizens from participation, Athenian democracy was a pioneering model that emphasized the importance of civic engagement and the rule of law."
#                 )
#             ),
#             RewrittenQuestion(
#                 rewritten_answer=(
#                     "# How Athenian Democracy Changed Ancient Greece\n"
#                     "The birth of democracy in Athens wasn’t just a shift in governance—it redefined politics for an entire civilization. Here’s how:\n\n"
#                     "## Breaking Aristocratic Power\n"
#                     "- **Cleisthenes’ Reforms:** Cleisthenes, known as the 'Father of Athenian Democracy,' spearheaded reforms that dismantled the influence of aristocratic families. For the first time, ordinary citizens were granted power in decision-making.\n"
#                     "- **Impact:** These reforms gave rise to a new political order, where citizens actively participated in shaping their city’s future. It was a radical departure from the elitist systems of the time.\n\n"
#                     "## The Role of Key Institutions\n"
#                     "- **The Assembly:** A forum where citizens gathered to debate and vote on laws, much like a public town hall or a modern-day parliament.\n"
#                     "- **The Council of 500:** Similar to a legislative committee today, the Council ensured debates were organized and productive, keeping the Assembly focused on key issues.\n\n"
#                     "## Long-lasting Impact\n"
#                     "- **A Legacy of Civic Responsibility:** The democratic principles pioneered in Athens created a ripple effect, influencing governance structures across ancient Greece and beyond.\n"
#                     "- **Influence on Modern Democracies:** Athenian democracy laid the groundwork for the systems of governance we see today, despite its flaws in inclusivity. The core principles of citizen participation and accountability still resonate in political thought.\n\n"
#                     "Athens didn’t just change its own political landscape—it set the stage for the democratic ideals that would shape civilizations for centuries to come."
#                 )
#             )
#         ),
#     ]
    
    
# Define the input model
class RewriteReference(BaseModel):
    question: str
    reference_answer: str

# Define the output model with the rewritten answer
class RewrittenAnswer(BaseModel):
    rewritten_answer: str

# Define the prompt class
class RewritePrompt(PydanticPrompt[RewriteReference, RewrittenAnswer]):
    instruction: str = (
        "Rewrite the provided reference answer using headings, bullet points, and other formatting to make it more structured and engaging. "
        "Incorporate subjective interpretations, inferences from world knowledge, analogical explanations, and emotional or dramatic framing. "
        "You may also modernize comparisons, simplify complex ideas, and add relevant details inferred from the reference for clarity. "
        "Maintain factual accuracy while allowing flexibility to extend, shorten, or infer meaning."
    )
    input_model = RewriteReference
    output_model = RewrittenAnswer
    examples = [
        (
            RewriteReference(
                question="What were the main institutions of Athenian democracy, and how did they function?",
                reference_answer=(
                    "The Athenian democracy was characterized by institutions such as the Assembly, where citizens could vote on laws and policies, and the Council of 500, which set the agenda for the Assembly. "
                    "The Assembly was the central decision-making body where all eligible citizens could participate, debate, and vote on various issues. "
                    "The Council of 500, selected by lot, was responsible for preparing the topics to be discussed in the Assembly, ensuring an organized and efficient legislative process. "
                    "Additionally, other institutions like the Courts and various public offices played crucial roles in maintaining the democratic framework, promoting accountability, and enabling citizen involvement in governance."
                )
            ),
            RewrittenAnswer(
                rewritten_answer=(
                    "# The Main Institutions of Athenian Democracy\n"
                    "Athenian democracy thrived because of several key institutions that empowered its citizens and ensured an organized political process. Here’s how each functioned:\n\n"
                    "## The Assembly\n"
                    "- **Role:** The Assembly was the heart of Athenian democracy, where citizens came together to decide on laws and policies. Think of it like a large public forum where every eligible citizen could participate directly in the decision-making process.\n"
                    "- **How it worked:** Much like a modern town hall, citizens gathered to voice their opinions, debate proposals, and cast their votes on important issues. This was not a government of representatives but one where ordinary people had a real say in governance, which was revolutionary for its time.\n\n"
                    "## The Council of 500\n"
                    "- **Role:** The Council of 500 played a crucial role behind the scenes, like a steering committee that ensured the Assembly functioned smoothly. They set the agenda and decided which issues would be discussed.\n"
                    "- **Selection process:** Council members were selected by lot, a method designed to ensure fairness and avoid the domination of political elites. Imagine drawing names from a hat, where every citizen had an equal chance of influencing the Assembly's discussions.\n\n"
                    "## The Courts and Public Offices\n"
                    "- **Role:** The Courts and various public offices functioned as the system’s checks and balances. Like today’s judiciary and public offices, they upheld justice and held leaders accountable.\n"
                    "- **Importance:** These institutions ensured that the democratic process was fair and transparent, promoting citizen engagement and preventing any individual or group from gaining too much power.\n\n"
                    "Together, these institutions formed the foundation of Athenian democracy, creating a system where citizens were actively involved in shaping their city’s future."
                )
            )
        ),
        (
            RewriteReference(
                question="How did the birth of democracy in Athens shape the political landscape of ancient Greece?",
                reference_answer=(
                    "The birth of democracy in Athens, around the 5th century BCE, marked a significant turning point in the political landscape of ancient Greece. "
                    "It was initiated by reforms attributed to leaders like Cleisthenes, who is often called the 'Father of Athenian Democracy.' These reforms dismantled the power of aristocratic families and established a system where citizens could participate directly in decision-making processes. "
                    "The Athenian democracy was characterized by institutions such as the Assembly, where citizens could vote on laws and policies, and the Council of 500, which set the agenda for the Assembly. "
                    "This system allowed for greater political participation and accountability, fostering a sense of civic responsibility among citizens. "
                    "The democratic principles developed in Athens influenced political thought and systems in subsequent civilizations, laying the groundwork for modern democratic governance. "
                    "Despite its limitations, such as excluding women, slaves, and non-citizens from participation, Athenian democracy was a pioneering model that emphasized the importance of civic engagement and the rule of law."
                )
            ),
            RewrittenAnswer(
                rewritten_answer=(
                    "# How Athenian Democracy Reshaped Ancient Greece\n"
                    "The birth of democracy in Athens during the 5th century BCE was not just a local experiment; it was a political transformation that influenced the entire Greek world and beyond. Here’s how:\n\n"
                    "## The Reforms of Cleisthenes\n"
                    "- **Breaking the power of aristocrats:** Cleisthenes, often called the 'Father of Athenian Democracy,' introduced reforms that broke the centuries-old grip of aristocratic families over Athens. These reforms weren’t just about giving people a say—they fundamentally shifted the balance of power from the elite to ordinary citizens.\n"
                    "- **Participation of citizens:** For the first time in history, ordinary citizens—not just wealthy or noble individuals—had the right to participate directly in the decision-making process. This was a radical shift from earlier systems, where political power was held by a privileged few.\n\n"
                    "## The Role of Democratic Institutions\n"
                    "- **The Assembly:** The Assembly was a gathering where citizens could debate, vote, and actively shape laws. Much like modern-day parliaments, it was the place where political decisions were made, but with direct involvement from the people, rather than elected representatives.\n"
                    "- **The Council of 500:** Similar to today’s legislative committees, the Council of 500 organized the discussions, ensuring that the Assembly remained focused and that decisions were made efficiently.\n\n"
                    "## The Legacy of Athenian Democracy\n"
                    "- **Civic engagement:** Athenian democracy fostered a sense of civic responsibility, encouraging citizens to think critically about their society and contribute to its governance. This engagement laid the foundation for many principles we now take for granted in modern democracies.\n"
                    "- **Influence on future systems:** Although the system wasn’t perfect (it excluded women, slaves, and non-citizens), the principles developed in Athens—participation, accountability, and the rule of law—became cornerstones of democratic governance in later civilizations, and their influence can still be seen in governments around the world today.\n\n"
                    "In short, Athenian democracy didn’t just change Athens—it reshaped the political landscape of ancient Greece and inspired democratic thought for centuries to come."
                )
            )
        ),
    ]
    

class Response(BaseModel):
    response: str

class ErroredResponse(BaseModel):
    errored_response: str
    error_description: str

class RewritePromptWithError(PydanticPrompt[Response, ErroredResponse]):
    instruction: str = (
        "Rewrite the provided response by introducing a single factual inaccuracy that directly affects the core content, such as a key date, "
        "numerical detail, or specific event. Avoid introducing the error in metaphors, analogies, or peripheral comparisons. "
        "The inaccuracy should be subtle but identifiable by comparing it with the original response without needing external knowledge. "
        "Ensure all other aspects of the response, including wording and structure, remain unchanged."
    )
    input_model = Response
    output_model = ErroredResponse
    examples = [
        (
            Response(
                response=(
                    "The Space Race was a competitive period between the United States and the Soviet Union. It began with the Soviet Union's "
                    "launch of Sputnik 1, the first artificial satellite, in 1957. This event motivated the U.S. to establish NASA in 1958, "
                    "setting the stage for a series of milestones in space exploration, including the 1969 Apollo 11 Moon landing."
                )
            ),
            ErroredResponse(
                errored_response=(
                    "The Space Race was a competitive period between the United States and the Soviet Union. It began with the Soviet Union's "
                    "launch of Sputnik 1, the first artificial satellite, in 1958. This event motivated the U.S. to establish NASA in 1958, "
                    "setting the stage for a series of milestones in space exploration, including the 1969 Apollo 11 Moon landing."
                ),
                error_description="The date for the launch of Sputnik 1 was changed from 1957 to 1958, introducing a minor factual inaccuracy."
            )
        ),
        (
            Response(
                response=(
                    "The wheel transformed transportation by making it easier to move heavy loads. Before the wheel, people relied on sledges or dragged goods across the ground."
                )
            ),
            ErroredResponse(
                errored_response=(
                    "The wheel transformed transportation by making it easier to move heavy loads. Before the wheel, people relied on carts or dragged goods across the ground."
                ),
                error_description="The errored response incorrectly states that people used 'carts' before the invention of the wheel, which introduces a subtle factual inaccuracy."
            )
        )
    ]