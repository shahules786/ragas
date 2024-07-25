import re
from turtle import clear
import typing as t
import json
from collections import defaultdict
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument

from ragas_experimental.testset.extractors.base import Extractor, Regex
from ragas_experimental.testset.graph import Node


@dataclass
class RulebasedExtractor(Extractor):
    regex: t.Optional[Regex] = None
    is_multiline: bool = False

    def __post_init__(self):
        assert self.regex is not None, "Regex pattern is not initialized"
        self.pattern = self.regex()

    async def aextract_text(self, text):
        raise NotImplementedError(
            "aextract() is not implemented for RulebasedExtractor"
        )

    def extract_text(self, text):
        text = json.loads(text)
        matches = (
            re.finditer(self.pattern, text, re.MULTILINE)
            if self.is_multiline
            else re.finditer(self.pattern, text)
        )
        result = defaultdict(list)
        for m in matches:
            m = {k: v for k, v in m.groupdict().items() if v is not None}
            for key in m:
                result[key].append(m[key])

        return result

    def extract(self, node: t.Union[Node, LCDocument]) -> t.Any:
        return super().extract(node)

    def merge_extractors(self, *extractors) -> t.List[Extractor]:
        if isinstance(
            self, RulebasedExtractor
        ):  # Check if called by an initiated class
            extractors = (self,) + extractors

        assert all(
            isinstance(extractor, RulebasedExtractor) for extractor in extractors
        ), "All extractors must be of type RulebasedExtractor"

        final_extractors: t.List[t.List[RulebasedExtractor]] = []
        added_indices = []
        for idx, extractor in enumerate(extractors):
            if idx not in added_indices:
                final_extractors.append([extractor])
                added_indices.append(idx)
                other_extractors = [
                    ext for i, ext in enumerate(extractors) if i not in added_indices
                ]
                filtered_extractors = [
                    ext
                    for ext in other_extractors
                    if extractor.attribute == ext.attribute
                    if extractor.is_multiline == ext.is_multiline
                ]
                for ext in filtered_extractors:
                    final_extractors[-1].append(ext)
                    added_indices.append(extractors.index(ext))

        extractors_to_return = []
        for group_index, extractors in enumerate(final_extractors):
            if len(extractors) > 1:
                # Process each pattern individually
                processed_patterns = []
                for extractor in extractors:
                    pattern = extractor.pattern
                    # Extract flags from the beginning of the pattern
                    flags = ""
                    if pattern.startswith("(?"):
                        flag_end = pattern.index(")")
                        flags = pattern[2:flag_end]
                        pattern = pattern[flag_end + 1:]
                    # Wrap the pattern in a non-capturing group with flags
                    processed_patterns.append(f"(?{flags}:{pattern})")

                # Join all processed patterns
                merged_pattern = "|".join(processed_patterns)

                updated_regex = Regex(name="merged_extractor", pattern=merged_pattern)
            else:
                updated_regex = extractors[0].regex

            extractors_to_return.append(
                RulebasedExtractor(
                    attribute=extractors[0].attribute,
                    regex=updated_regex,
                    is_multiline=extractors[0].is_multiline,
                )
            )
        return extractors_to_return


class MarkdownHeadingExtractor(Extractor):
   
    async def aextract_text(self, text):
        raise NotImplementedError(
            "aextract() is not implemented for RulebasedExtractor"
        )
        
    def extract_text(self, text) -> t.Dict[str, t.Any]:
        
        text = json.loads(text)
        # Regular expressions to capture headings and subheadings
        heading_pattern = r"##\s+(.+)"
        subheading_pattern = r"###\s+(.+)"

        output = defaultdict(list)
        current_heading = None

        # Process the text line by line
        for line in text.split('\n'):
            heading_match = re.match(heading_pattern, line)
            subheading_match = re.match(subheading_pattern, line)
            
            if heading_match:
                current_heading = heading_match.group(1).strip()
            elif subheading_match and current_heading:
                subheading = subheading_match.group(1).strip()
                output[current_heading].append(subheading)
        
        output = dict(output)
        return {"headings": output}
    
    def extract(self, node: t.Union[Node, LCDocument]) -> t.Any:
        return super().extract(node)
    
    
    
class MarkdownLinkExtractor(Extractor):
   
    async def aextract_text(self, text):
        raise NotImplementedError(
            "aextract() is not implemented for RulebasedExtractor"
        )
        
    def extract_text(self, text) -> t.Dict[str, t.Any]:
        
        text = json.loads(text)
        pattern = r'\[([^\[\]]+)\]\(((?!http[s]?:\/\/|{{<?\s*ref)[^)]+)\)'
        local_links_regex = re.compile(pattern)
        matches = local_links_regex.findall(text)

        results = {}
        for match in matches:
            link_text, link_url = match
            results[link_text] = link_url
        
        return {"local_links": results}
        
    def extract(self, node: t.Union[Node, LCDocument]) -> t.Any:
        return super().extract(node)
    
    
class MarkdownHeadingsFlatExtractor(Extractor):
    
    def extract_text(self, text: str):
        text = json.loads(text)
        
        markdown_headings = r"^(#{1,6})\s+(.*)"
        matches = re.findall(markdown_headings, text, re.MULTILINE)
        cleaned_headings = [match[1].strip() for match in matches]
        
        return {"markdown_headlines": cleaned_headings}
    
    def extract(self, node: t.Union[Node, LCDocument]) -> t.Any:
        return super().extract(node)
        
    async def aextract_text(self, text):
        raise NotImplementedError(
            "aextract() is not implemented for RulebasedExtractor"
        )
   
   
links_extractor_pattern = r"(?i)\b(?:https?://|www\.)\S+\b"
emails_extractor_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

email_extractor = RulebasedExtractor(
    regex=Regex(name="email", pattern=emails_extractor_pattern)
)
link_extractor = RulebasedExtractor(
    regex=Regex(name="link", pattern=links_extractor_pattern)
)
