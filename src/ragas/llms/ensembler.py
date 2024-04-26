from email.policy import strict
import json
import os
import typing as t
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from ragas.embeddings.base import embedding_factory
from ragas.llms.base import llm_factory
from ragas.llms.prompt import Prompt

conflict_resolution = Prompt(
    name="conflict_resolution",
    instruction="""
Your task is to Analyze contradicting results and propose a conclusion.

Inputs:

Original Task:
Detail the specific instructions, guidelines,expected output formats and inputs relevant to the task.
Outcome 1:
The first outcome, including its decision and the reasoning behind it.
Outcome 2:
The second outcome, including its decision and the reasoning behind it.

Steps to follow to arrive and propose a resolution.
1. Critic reasonings: critic the reasoning behind both outputs
2. Propose a Resolution: Recommend the most justified decision, synthesizing the critique and context alignment with the task guidelines.

""",
    examples=[
        {
            "original_task": """
        Evaluate the correctness of statements based on a provided context. Each statement should be assessed for its truthfulness with a verdict of 1 if it can be verified by the context or 0 if it cannot.
        Context: Alice is a software developer who specializes in backend technologies. She has been working with Java for over five years.
        Statements: ["Alice develops mobile apps."]""",
            "outcome_1": {
                "statement": "Alice develops mobile apps.",
                "verdict": 0,
                "reason": "The context only mentions her specialization in backend technologies, not mobile app development.",
            },
            "outcome_2": {
                "statement": "Alice develops mobile apps.",
                "verdict": 1,
                "reason": "Backend technologies can include mobile app development, hence it's reasonable to infer that she might develop mobile apps.",
            },
            "resolution": {
                "Critic reasonings": [
                    {
                        "outcome_1": "Outcome 1 argues that the statement cannot be verified because the context explicitly mentions Alice's specialization in backend technologies, which does not necessarily include mobile app development."
                    },
                    {
                        "outcome 2": "Outcome 2 argues that the statement can be verified because backend technologies could potentially include mobile app development, hence it is reasonable to infer that she might develop mobile apps."
                    },
                ],
                "Propose a Resolution": "The task requires the statement to be verified based on the given context. While backend technologies could include mobile app development, there is no explicit mention of mobile app development in the context provided about Alice. Therefore, Outcome 1's reasoning, which adheres more strictly to the context provided without assuming broader applications of her skills, seems more aligned with the task guidelines. Thus, the final verdict should be 0, but with a reason that acknowledges the specific mention of her skills in backend technologies and the absence of any direct link to mobile app development.",
                "Resolution": {
                    "statement": "Alice develops mobile apps.",
                    "verdict": 0,
                    "reason": "The context focuses on Alice's experience with backend technologies and Java, without any direct mention of mobile app development, thus it does not support a broad interpretation that she works on mobile apps.",
                },
            },
        }
    ],
    input_keys=["original_task", "outcome_1", "outcome_2"],
    output_key="resolution",
)


def json_patching(new_data, filename):
    if os.path.exists(filename):
        data = json.load(open(filename))
    else:
        data = []

    # Convert keys to strings
    new_data = {str(key): value for key, value in new_data.items()}
    data.append(new_data)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    return data


@dataclass
class Ensemble:
    embedding_model = embedding_factory()
    llm = llm_factory()
    conflict_resolution = conflict_resolution
    strictness_level: int = 2

    def cosine_similarity(self, embedding_1: np.ndarray, embedding_2: np.ndarray):
        norms_1 = np.linalg.norm(embedding_1, axis=1)
        norms_2 = np.linalg.norm(embedding_2, axis=1)
        embedding_1_normalized = embedding_1 / norms_1[:, np.newaxis]
        embedding_2_normalized = embedding_2 / norms_2[:, np.newaxis]
        similarity_matrix = embedding_1_normalized @ embedding_2_normalized.T

        return similarity_matrix

    # def from_list_of_strings(self, inputs: list[list[str]]):

    #     inputs_flatten = [text for item in inputs for text in item]
    #     input_embeddings = np.array(self.embedding_model.embed_documents(inputs_flatten))
    #     return input_embeddings
    #     similarity_matrix = self.cosine_similarity(input_embeddings, input_embeddings)
    #     output_matrix = np.zeros((len(inputs_flatten), 1))
    #     input_lens = [len(item) for item in inputs]
    #     input_indices = [[sum(input_lens[:i]), sum(input_lens[:i+1])-1] for i in range(len(input_lens))]
    #     for i in range(len(inputs_flatten)):
    #         for j in input_indices:
    #             indices = list(range(j[0], j[1]+1))
    #             if i not in indices:
    #                 output_matrix[i] += similarity_matrix[i, indices].max()

    #     print(output_matrix.flatten())
    #     top_indices = np.argsort(output_matrix.flatten())[::-1]
    #     mean_num_outputs = int(round(np.mean(input_lens)))
    #     return [inputs_flatten[i] for i in top_indices[:mean_num_outputs]]

    async def from_list_of_strings(self, inputs: list[list[str]]):
        json_patching({"inputs": inputs}, "ensembler_track.json")
        inputs_flatten = [text for item in inputs for text in item]
        # subset_len = int(round(np.mean([len(item) for item in inputs])))
        input_embeddings = np.array(
            await self.embedding_model.aembed_documents(inputs_flatten)
        )
        clustering = DBSCAN(eps=0.05, min_samples=2, metric="cosine").fit(
            input_embeddings
        )
        cluster_labels, cluster_counts = np.unique(
            clustering.labels_, return_counts=True
        )
        labels_inputs = {label: [] for label in cluster_labels}
        for i, label in enumerate(clustering.labels_):
            labels_inputs[label].append(inputs_flatten[i])

        cluster_frequency = {
            label: count for label, count in zip(cluster_labels, cluster_counts)
        }
        if -1 in cluster_frequency:
            _ = cluster_frequency.pop(-1)
        cluster_frequency = dict(
            sorted(cluster_frequency.items(), key=lambda x: x[1], reverse=True)
        )
        top_labels = list(
            cluster_frequency.keys()
        )  # selecting all clusters ecxept the noise cluster

        output = []
        for label in top_labels:
            candidates = labels_inputs[label]
            indices = [inputs_flatten.index(candidate) for candidate in candidates]
            input_embeddings_subset = input_embeddings[indices]
            similarity_matrix = self.cosine_similarity(
                input_embeddings_subset, input_embeddings_subset
            )
            np.fill_diagonal(similarity_matrix, 0)
            best_index = similarity_matrix.sum(axis=1).flatten().argmax()
            output.append(candidates[best_index])

        json_patching(labels_inputs, "ensembler_track.json")
        json_patching({"output": output}, "ensembler_track.json")
        # json_patching(labels_inputs, "ensembler_track.json")
        return output

    def from_discrete_with_self_correction(self, inputs: list[list[t.Dict]], attribute: str, partial_prompt):
        assert all(
            len(item) == len(inputs[0]) for item in inputs
        ), "all items should have the same length"
        assert all(
            attribute in item for input in inputs for item in input
        ), "attribute not found in all items"

        verdict_agg = []
        for i in range(len(inputs[0])):
            verdicts = [inputs[k][i][attribute] for k in range(len(inputs))]
            verdict_counts = dict(Counter(verdicts).most_common())
            print(verdict_counts)
            item = inputs[0][i]
            if self.strictness_level == 1:
                item[attribute] = list(verdict_counts.keys())[0]
            else:
                votes = list(verdict_counts.values())[0]
                if votes == len(verdicts):
                    item[attribute] = list(verdict_counts.keys())[0]
                else:
                    # if there is no clear winner then do constitutional AI stuff
                    verdicts = verdict_counts.keys()
                    outcomes = {}
                    for idx, verdict in enumerate(verdicts):
                        outcomes[f"outcome_{idx+1}"] = [
                            inputs[k][i]
                            for k in range(len(inputs))
                            if inputs[k][i][attribute] == verdict
                        ][0]
                        print(outcomes)
                    p_value = self.conflict_resolution.format(
                        original_task=partial_prompt(inputs[0][i]["statement"]),
                        **outcomes,
                    )
                    output = self.llm.generate_text(p_value)
                    print(json.loads(output.generations[0][0].text))
                    item[attribute] = json.loads(output.generations[0][0].text)[
                        "Resolution"
                    ]["verdict"]

            verdict_agg.append(item)

        return verdict_agg
    
    def from_discrete(self, inputs: list[list[t.Dict]], attribute: str):
        if not all(
            len(item) == len(inputs[0]) for item in inputs
        ):
            print(inputs)
            return []
                
        assert all(
            attribute in item for input in inputs for item in input
        ), "attribute not found in all items"
        
        verdict_agg = []
        for i in range(len(inputs[0])):
            item = inputs[0][i]
            verdicts = [inputs[k][i][attribute] for k in range(len(inputs))]
            verdict_counts = dict(Counter(verdicts).most_common())
            item[attribute] = list(verdict_counts.keys())[0]
            verdict_agg.append(item)
            
        return verdict_agg
    
    def from_discrete_with_inference_levels(self, inputs: list[list[t.Dict]], attribute: str):
        assert all(
            len(item) == len(inputs[0]) for item in inputs
        ), "all items should have the same length"
        assert all(
            attribute in item for input in inputs for item in input
        ), "attribute not found in all items"

        verdict_agg = []
        for i in range(len(inputs[0])):
            item = inputs[0][i]
            verdicts = [inputs[k][i][attribute] for k in range(len(inputs))]
            verdict_levels = [f"{inputs[k][i][attribute]}-{inputs[k][i]['inference_level']}" for k in range(len(inputs))]
            print("verdicts", verdicts)
            print("verdict_levels", verdict_levels)
            verdict_counts = dict(Counter(verdicts).most_common())
            first_val = next(iter(verdict_counts.values()))            
            if first_val == len(verdicts):
                item[attribute] = list(verdict_counts.keys())[0]
            else:
                verdict_levels = [f"{inputs[k][i][attribute]}-{inputs[k][i]['inference_level']}" for k in range(len(inputs))]
                verdict_level_count = dict(Counter(verdict_levels).most_common())
                first_val = next(iter(verdict_level_count.values()))
                # if all inferece says same verdict then go with that (even with different levels of inference)
                
                if first_val > len(verdicts)//2:
                    # clear winner
                    item[attribute] = list(verdict_level_count.keys())[0].split("-")[0]
                else:
                    print(verdict_level_count)
                    # analyse levels of inference
                    from collections import defaultdict
                    # analyze verdicts from different levels of inference
                    verdict_to_levels = defaultdict(set)
                    for k in range(len(inputs)):
                        verdict_to_levels[verdict_levels[k].split('-')[0]].add(verdict_levels[k].split('-')[1])
                    # check for conflicting levels
                    # verdict_to_levels = sorted(verdict_to_levels.items(), key=lambda x: x[0])
                    inference_levels = [list(item) for item in verdict_to_levels.values()]
                    inference_levels = [list(map(int, item)) for item in inference_levels]
                    print("iunference levels", inference_levels)
                    conflicting_level = np.intersect1d(*inference_levels)
                    if conflicting_level:
                        # {1: {'1'}, 0: {'1','2'}}
                        # if conflicting levels, go with the verdict from the liberal level
                        remining_levels = []
                        for levels in inference_levels:
                            levels = [level for level in levels if level not in conflicting_level]
                            remining_levels.extend(levels)
                            
                        print(remining_levels)
                        smoothest_level = max(remining_levels)
                        print(smoothest_level)
                        item[attribute] = [inputs[k][i][attribute] for k in range(len(inputs)) if inputs[k][i]['inference_level'] == smoothest_level][0]
                    else:
                        # no conflicting levels
                        # {1: {'1','0'}, 0: {'2'}}
                        # go with the verdict from the strictest level
                        strictest_level = min(sum(inference_levels, []))
                        item[attribute] = [inputs[k][i][attribute] for k in range(len(inputs)) if inputs[k][i]['inference_level'] == strictest_level][0]
                    
                
            # print(verdict_levels)
            # print(verdict_counts)
            # print()
            # votes = list(verdict_counts.values())[0]
            # if votes > len(verdicts)//2:
            #     item[attribute] = list(verdict_counts.keys())[0]
            # else:
            #     # no clear winner - go with verdict from strictest level
            #     level_dict = {}
            #     for level in range(1,5):
            #         verdict_level = [inputs[k][i][attribute] for k in range(len(inputs)) if inputs[k][i]["inference_level"] == level]
            #         if verdict_level:
            #             level_dict[level] = verdict_level
            #     verdict_level = dict(sorted(level_dict.items(), key=lambda x: x[0], reverse=False))
            #     print(verdict_level)
            #     item[attribute] = list(verdict_level.values())[0][0]
                
            verdict_agg.append(item)
                    
        return verdict_agg
    
    # def from_discrete(self, inputs: list[list[t.Dict]], attribute: str):
    #     assert all(
    #             len(item) == len(inputs[0]) for item in inputs
    #         ), "all items should have the same length"
    #     assert all(
    #             attribute in item for input in inputs for item in input
    #         ), "attribute not found in all items"
        
    #     verdict_agg = []
    #     for i in range(len(inputs[0])):
    #         item = inputs[0][i]
    #         verdicts = [inputs[k][i][attribute] for k in range(len(inputs))]
    #         verdict_counts = dict(Counter(verdicts).most_common())
    #         first_val = next(iter(verdict_counts.values()))            
    #         if first_val > len(verdicts)//2:
    #             item[attribute] = list(verdict_counts.keys())[0]
    #         else:
    #             pass
                
                
            
        
ensembler = Ensemble()


# if __name__ == "__main__":
#     x = ["I have a blue car","I have a bike","I have a cat",]
#     y = ["I have a car","I have a bike","I have a lorry"]
#     z = ["I have a car","I have a bike","I have a lorry"]

#     ensemble = Ensemble()
#     X = await ensemble.from_list_of_strings(inputs=[x,y,z])
