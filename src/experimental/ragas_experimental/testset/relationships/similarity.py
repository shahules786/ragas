import typing as t
from dataclasses import dataclass

import os
import numpy as np

from ragas_experimental.testset.graph import Node
from ragas_experimental.testset.relationships.base import Similarity


@dataclass
class JaccardSimilarity(Similarity):
    type: t.Optional[str] = None
    threshold: t.Optional[int] = 80

    def __post_init__(self):
        if self.type == "fuzzy":
            try:
                from fuzzywuzzy import fuzz
            except ImportError:
                raise ImportError(
                    "fuzzywuzzy is not installed. Run pip install fuzzywuzzy"
                )
            self.fuzz = fuzz
            self.threshold = self.threshold or 80

    def _calculate_fuzz(self, x: t.List[str], y: t.List[str]):
        fuzz_scores = 0
        for item in x:
            fuzz_scores = 0
            for element in y:
                if self.fuzz.ratio(item, element) >= (self.threshold or 0):
                    fuzz_scores += 1
        return fuzz_scores

    def extract(self, x_nodes: t.List[Node], y_nodes: t.List[Node]):
        jaccard_similarity_matrix = np.zeros((len(x_nodes), len(y_nodes)))

        doc1_items = [self.get_attribute(doc, self.attribute1) for doc in x_nodes]
        doc2_items = [self.get_attribute(doc, self.attribute2) for doc in y_nodes]
        if all(isinstance(item, list) for item in doc1_items) and all(
            isinstance(item, list) for item in doc2_items
        ):
            for i, a in enumerate(doc1_items):
                for k, b in enumerate(doc2_items):
                    if self.type == "fuzzy":
                        intersection = self._calculate_fuzz(a, b)
                    else:
                        intersection = len(set(a).intersection(set(b)))
                    union = len(set(a).union(set(b)))
                    jaccard_similarity_matrix[i][k] = (
                        intersection / union if union != 0 else 0
                    )
                    jaccard_similarity_matrix[i][k] = intersection / union

        return jaccard_similarity_matrix


@dataclass
class CosineSimilarity(Similarity):
    def extract(self, x_nodes: t.List[Node], y_nodes: t.List[Node]) -> t.Any:
        embeddings_1 = [self.get_attribute(doc, self.attribute1) for doc in x_nodes]
        embeddings_2 = [self.get_attribute(doc, self.attribute2) for doc in y_nodes]
        embeddings_1 = np.array(embeddings_1)
        embeddings_2 = np.array(embeddings_2)
        cosine_similarity_matrix = np.dot(embeddings_1, embeddings_2.T) / (
            np.linalg.norm(embeddings_1, axis=1) * np.linalg.norm(embeddings_2, axis=1)
        )
        return 1-cosine_similarity_matrix


class FilePathLinker:
    
    def extract(self, x_nodes: t.List[Node],  y_nodes: t.List[Node]):
    
        # assume that all links are preprocessed and validated. 
        for node in x_nodes:
            local_file_paths = node.properties["metadata"].get("local_links", {})
            for key, path in local_file_paths.items():
                basename = os.path.basename(path)
                if basename.startswith("#"):
                    file_path, section = os.path.dirname(path), basename.replace("#", "")
                else:
                    file_path, section = path, None
                   
                if os.path.exists(file_path):
                    related_nodes = [y_node for y_node in y_nodes if y_node.properties["metadata"].get("source") == file_path]
                    if section:
                        section = section.replace('-', ' ').lower()
                        related_nodes = [y_node for y_node in related_nodes if section.lower() in [title.lower() for title in y_node.properties["metadata"].get("markdown_headlines", [])]]
                        
                    for related_node in related_nodes:
                        node.add_relationship(related_node, "file_link", {"path": path, "section": section, "text": key})            
        return x_nodes