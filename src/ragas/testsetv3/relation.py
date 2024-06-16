import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from graphene.types.schema import Schema

from ragas.testsetv3.graph import Node, NodeType, Relationship


@dataclass
class Similarity(ABC):
    name: str
    attribute1: str
    attribute2: str

    def get_attribute(self, doc: Node, attribute: str):
        if attribute == "page_content":
            return doc.properties["page_content"]
        elif attribute in doc.properties["metadata"]:
            return doc.properties["metadata"][attribute]
        else:
            raise ValueError(f"Attribute {attribute} not found in node")

    @abstractmethod
    def extract(self, doc1: t.List[Node], doc2: t.List[Node]) -> t.Any:
        pass


@dataclass
class Jaccard(Similarity):
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
            fuzz_scores += sum(
                [self.fuzz.ratio(item, element) >= self.threshold for element in y]
            )
        return fuzz_scores

    def extract(self, doc1: t.List[Node], doc2: t.List[Node]):
        jaccard_similarity_matrix = np.zeros((len(doc1), len(doc2)))

        doc1_items = [self.get_attribute(doc, self.attribute1) for doc in doc1]
        doc2_items = [self.get_attribute(doc, self.attribute2) for doc in doc2]
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
                    jaccard_similarity_matrix[i][k] = intersection / union

        return jaccard_similarity_matrix


@dataclass
class Cosine(Similarity):
    def extract(self, doc1: t.List[Node], doc2: t.List[Node]) -> t.Any:
        embeddings_1 = [self.get_attribute(doc, self.attribute1) for doc in doc1]
        embeddings_2 = [self.get_attribute(doc, self.attribute2) for doc in doc2]
        embeddings_1 = np.array(embeddings_1)
        embeddings_2 = np.array(embeddings_2)
        cosine_similarity_matrix = np.dot(embeddings_1, embeddings_2.T) / (
            np.linalg.norm(embeddings_1, axis=1) * np.linalg.norm(embeddings_2, axis=1)
        )
        return cosine_similarity_matrix


@dataclass
class Graph(ABC):
    schema: Schema

    @abstractmethod
    def form_relation(
        self, query, nodes: t.List[Node], relationships: t.List[Relationship], kwargs
    ) -> t.Any:
        pass


@dataclass
class SimilarityGraph(Graph):
    schema: Schema
    extractors: t.List[Similarity]
    score_threshold: float = 0.0

    def form_relation(
        self, query, nodes: t.List[Node], relationships: t.List[Relationship], kwargs
    ):
        result = self.schema.execute(
            query, context={"nodes": nodes, "relationships": relationships}
        )
        if result is None:
            return None
        node_ids = [item.get("id") for item in result.data["filterNodes"]]
        nodes_ = [node for node in nodes if node.id in node_ids]
        for extractor in self.extractors:
            similarity_matrix = extractor.extract(nodes_, nodes_)
            for i, row in enumerate(similarity_matrix):
                new_relationships = []
                for j, score in enumerate(row):
                    if i != j and score >= self.score_threshold:
                        relationship = Relationship(
                            source=nodes_[i],
                            target=nodes_[j],
                            label=extractor.name,
                            properties={"score": score},
                        )
                        new_relationships.append(relationship)
                        relationship.source.relationships.append(relationship)
                        relationship.target.relationships.append(relationship)

                relationships.extend(new_relationships)

        return (nodes, relationships)


if __name__ == "__main__":
    text = """
    Contact us at info@example.com or visit https://www.example.com for more information.
    Alternatively, email support@service.com or check http://service.com.
    You can also visit our second site at www.secondary-site.org or email us at secondary-info@secondary-site.org.
    """

    docs = [Node(id="1", label=NodeType.DOC, properties={"headlines": ["Doc 1"]})]
    jaccard_overlap = Jaccard(
        name="jaccard", attribute1="headlines", attribute2="headlines"
    )
    _ = jaccard_overlap.extract(docs, docs)

    # from ragas.embeddings import embedding_factory
    # cosine = Cosine(name='cosine',attribute1="embedding",attribute2="embedding")
    # model=embedding_factory()
    # doc = Node(properties= {"page_content":text,"metadata":{'embedding':await model.embed_text(text)}})
    # cosine.extract([doc],[doc])