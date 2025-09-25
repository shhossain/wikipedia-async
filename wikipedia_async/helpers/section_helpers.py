from pydantic import BaseModel
from pydantic import Field
from wikipedia_async.models.section_models import Section, SectionJson
from wikipedia_async.helpers.content_helpers import parse_sections
from wikipedia_async.helpers.html_helpers import parse_wiki_html
from typing import List, Optional, Dict, Any, overload
import re


class SectionHelper(BaseModel):
    """Helper class to manage and interact with a list of Section objects."""

    sections: list[Section] = Field(
        default_factory=list, description="List of root-level sections"
    )

    @classmethod
    def from_content(cls, content: str) -> "SectionHelper":
        """Create a SectionHelper from raw Wikipedia content."""
        sections = parse_sections(content)
        return cls(sections)

    @classmethod
    def from_html(cls, html: str) -> "SectionHelper":
        """Create a SectionHelper from raw Wikipedia HTML."""
        sections = parse_wiki_html(html)
        return cls(sections)

    def __init__(self, sections: Optional[List[Section]] = None):
        # Use BaseModel.__init__ to ensure pydantic internals are initialized
        # Delegate initialization to BaseModel to avoid missing __pydantic_fields_set__
        super().__init__(sections=sections or [])
        # derived attribute, not a pydantic field
        object.__setattr__(self, "_length", len(self.sections))

    def iter_sections(self):
        """Recursively iterate through all sections (including nested ones)."""

        def _iter_recursive(sections):
            for section in sections:
                yield section
                yield from _iter_recursive(section.children)

        return _iter_recursive(self.sections)

    def flattened_sections(self) -> List[Section]:
        """Get a flat list of all sections (including nested ones)."""
        return list(self.iter_sections())

    def first_content(self) -> str:
        """Get the content of the first section, if available."""
        for section in self.sections:
            if section.section_content.strip():
                return section.section_content.strip()
        return ""

    # add slice support
    @overload
    def __getitem__(self, i: slice) -> "SectionHelper": ...

    @overload
    def __getitem__(self, i: int) -> Section: ...

    def __getitem__(self, i):
        if isinstance(i, slice):
            return SectionHelper(self.sections[i])
        elif isinstance(i, int):
            return self.sections[i]
        else:
            raise TypeError("Invalid argument type.")

    def get_section_by_title(
        self,
        title: str,
        case_sensitive: bool = False,
        prioritize_top_level: bool = False,
        regex: bool = False,
    ) -> Section | None:
        """Find a section by its title.

        If prioritize_top_level is True, prefer matches among root-level sections
        (self.sections) before searching the whole tree.
        """

        def normalize(s: str) -> str:
            return s if case_sensitive else s.lower()

        def compare(t1: str, t2: str) -> bool:
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                return re.fullmatch(t1, t2, flags) is not None
            else:
                return normalize(t1) == normalize(t2)

        def search_recursive(nodes: List[Section]) -> Section | None:
            for node in nodes:
                if compare(node.title, title):
                    return node
                result = search_recursive(node.children)
                if result:
                    return result
            return None

        # Try root-level match first if requested
        if prioritize_top_level:
            for node in self.sections:
                if normalize(node.title) == normalize(title):
                    return node

        # Otherwise search the whole tree
        return search_recursive(self.sections)

    def get_sections_by_titles(
        self,
        titles: List[str],
        case_sensitive: bool = False,
        prioritize_top_level: bool = False,
        regex: bool = False,
    ) -> List[Section | None]:
        """Find sections by their titles.

        If prioritize_top_level is True, prefer matches among root-level sections
        (self.sections) before searching the whole tree.
        """

        def normalize(s: str) -> str:
            return s if case_sensitive else s.lower()

        def compare(t1: str, t2: str) -> bool:
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                return re.fullmatch(t1, t2, flags) is not None
            else:
                return normalize(t1) == normalize(t2)

        def search_recursive(nodes: List[Section], target: str) -> Section | None:
            for node in nodes:
                if compare(node.title, target):
                    return node
                found = search_recursive(node.children, target)
                if found:
                    return found
            return None

        results: List[Section | None] = []
        for title in titles:
            target = normalize(title)

            # Try root-level match first if requested
            found: Section | None = None
            if prioritize_top_level:
                for node in self.sections:
                    if compare(node.title, target):
                        found = node
                        break

            # If not found (or not prioritizing), search the whole tree
            if not found:
                found = search_recursive(self.sections, target)

            results.append(found)

        return results

    def tree_view(self, content_limit: int = 0) -> str:
        """Return a tree-like view of all sections."""
        text = ""
        for section in self.sections:
            text += section.tree_view(content_limit)
        return text.strip()

    def summary(self) -> dict:
        """Get a summary of the section structure."""

        def count_sections(sections):
            total = len(sections)
            max_depth = 0
            for section in sections:
                if section.children:
                    child_count, child_depth = count_sections(section.children)
                    total += child_count
                    max_depth = max(max_depth, child_depth + 1)
            return total, max_depth

        total_sections, max_depth = count_sections(self.sections)

        return {
            "total_sections": total_sections,
            "root_sections": len(self.sections),
            "max_depth": max_depth,
            "has_nested_structure": max_depth > 0,
        }


    def to_string(self) -> str:
        """Convert the entire section tree to a string."""
        return "\n".join(section.to_string() for section in self.sections)
    
    def to_json(self, keep_links: bool = True) -> list[SectionJson]:
        """Convert the entire section tree to a list of dictionaries."""
        ress = [section.to_json() for section in self.sections]
        def clean_links(section_json):
            for pi in range(len(section_json["paragraphs"])):
                section_json["paragraphs"][pi]["links"] = []
            if "tables" in section_json:
                for ti in range(len(section_json["tables"])):
                    section_json["tables"][ti]["links"] = []
            if "children" in section_json:
                for child in section_json["children"]:
                    clean_links(child)

        if not keep_links:
            for res in ress:
                clean_links(res)
        return ress


def get_summary(content: str) -> str:
    # before headings
    pat = re.compile(r"^(={2,})\s*(.*?)\s*\1$", re.MULTILINE)
    match = pat.search(content)
    if match:
        return content[: match.start()].strip()

    return content.strip()[:500] + ("..." if len(content) > 500 else "")
