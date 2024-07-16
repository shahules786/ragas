import typing as t
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas_experimental.testset.graph import Node, NodeLevel, NodeType, Relationship
from ragas_experimental.testset.utils import merge_dicts

@dataclass
class HierarchialParser:
    node_parser_ids: t.List[str]
    node_parser_map: t.Dict[str, RecursiveCharacterTextSplitter]
    common_metadata_keys: t.List[str] = field(default_factory=lambda: ['x'])

    @classmethod
    def from_defaults(
        cls, chunk_sizes: t.List[int] = [2048, 1024, 512], chunk_overlap=20
    ):
        node_parser_ids = [f"chunk_size_{chunk_size}" for chunk_size in chunk_sizes]
        node_parser_map = {}
        for chunk_size, node_parser_id in zip(chunk_sizes, node_parser_ids):
            node_parser_map[node_parser_id] = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        return cls(
            node_parser_ids=node_parser_ids,
            node_parser_map=node_parser_map,
        )
        
    
    def _parent_child_relationship(
        self, parent_node: Node, child_chunks: t.List[Document]
    ):
        nodes = []
        node_level = parent_node.level.next_level()
        for idx, chunk in enumerate(child_chunks):
            nodes.append(
                Node(
                    label=NodeType.CHUNK,
                    properties={
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                    relationships=[],
                    level=node_level,
                )
            )
            properties = {"seperator": chunk.metadata.get("seperator")} if chunk.metadata.get("seperator") else {}
            relationship = Relationship(
                source=parent_node,
                target=nodes[-1],
                label="child",
                properties=properties,
            )
            parent_node.relationships.append(relationship)
            relationship = Relationship(
                source=nodes[-1],
                target=parent_node,
                label="parent",
                properties=properties,
            )
            nodes[-1].relationships.append(relationship)
        if parent_node.level.value == 0:
            nodes.insert(0, parent_node)
        return nodes
    
    def _reassign_metadata(self, document: Document, chunks: t.List[Document]):
        extractive_metadata_keys = document.metadata.get("extractive_metadata_keys", [])
        for chunk in chunks:
            page_content = chunk.page_content
            text_chunk_metadata = {"extractive_metadata_keys": extractive_metadata_keys}
            for metadata_key in extractive_metadata_keys:
                metadata = document.metadata.get(metadata_key)
                if isinstance(metadata, str):
                    idx = page_content.find(metadata)
                    if idx != -1:
                        text_chunk_metadata[metadata_key] = metadata
                elif isinstance(metadata, list):
                    metadata_match_idx = [page_content.find(item) for item in metadata]
                    metadata_idx = [
                        idx
                        for idx, match_idx in enumerate(metadata_match_idx)
                        if match_idx != -1
                    ]
                    if metadata_idx:
                        text_chunk_metadata[metadata_key] = [
                            metadata[i] for i in metadata_idx
                        ]
            text_chunk_metadata = merge_dicts(chunk.metadata, text_chunk_metadata)
            chunk.metadata = text_chunk_metadata
            for key in self.common_metadata_keys:
                if key in document.metadata:
                    chunk.metadata[key] = document.metadata[key]
        return chunks

    def _recursive_node_from_nodes(self, nodes: t.List[Node], level: int = 0):
        sub_nodes = []
        for node in nodes:
            
            doc_chunks = self.node_parser_map[self.node_parser_ids[level]].split_documents(
                [node.to_document()]
            )
            doc_chunks = self._reassign_metadata(node.to_document(), doc_chunks)
            child_nodes = self._parent_child_relationship(node, doc_chunks)
            sub_nodes.extend(child_nodes)
            
        if level < len(self.node_parser_ids) - 1:
            nodes_to_recurse = [node for node in sub_nodes if node.level.value == level + 1]
            sub_sub_nodes = self._recursive_node_from_nodes(nodes_to_recurse, level + 1)
        else:
            sub_sub_nodes = []
            
        return sub_nodes + sub_sub_nodes

    def get_nodes_from_document(
        self,
        document: Document,
    ) -> t.List[str]:
        nodes = []
        node = Node(
            label=NodeType.DOC,
            properties={
                "page_content": document.page_content,
                "metadata": document.metadata,
            },
            relationships=[],
            level=NodeLevel.LEVEL_0,
        )
        current_nodes = self._recursive_node_from_nodes([node], 0)
        nodes.extend(current_nodes)
        return nodes

    def from_documents(self, documents: t.List[Document]) -> t.List[str]:
        """Split documents into hierarchical chunks."""

        nodes = []
        for doc in documents:
            doc_nodes = self.get_nodes_from_document(doc)
            nodes.extend(doc_nodes)

        return nodes
