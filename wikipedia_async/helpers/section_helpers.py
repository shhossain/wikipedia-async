from pydantic import BaseModel
from pydantic import Field
from wikipedia_async.models.section_models import (
    Section,
    SectionJson,
    SectionTreeJson,
    Table,
    TableConfig,
)
from wikipedia_async.helpers.content_helpers import parse_sections
from wikipedia_async.helpers.html_helpers import parse_wiki_html
from typing import Optional, overload
import re
from functools import cached_property


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

    def __init__(self, sections: Optional[list[Section]] = None):
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

    def flattened_sections(self) -> list[Section]:
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

        def search_recursive(nodes: list[Section]) -> Section | None:
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

    def get_table_by_caption(
        self,
        caption: str,
        case_sensitive: bool = False,
        regex: bool = False,
        section: Optional[Section | str] = None,
    ) -> Optional[Table]:
        """Find a table by its caption."""

        def normalize(s: str) -> str:
            return s if case_sensitive else s.lower()

        def compare(t1: str, t2: str) -> bool:
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                return re.fullmatch(t1, t2, flags) is not None
            else:
                return normalize(t1) == normalize(t2)

        secs = []
        if section:
            title: str = section.title if isinstance(section, Section) else section
            target_section = self.get_section_by_title(title)
            if not target_section:
                return None
            secs = [target_section]
        else:
            secs = self.sections

        for sec in secs:
            for table in sec.tables:
                if not table.caption:
                    continue
                if compare(table.caption, caption):
                    return table

        return None

    def get_sections_by_titles(
        self,
        titles: list[str],
        case_sensitive: bool = False,
        prioritize_top_level: bool = False,
        regex: bool = False,
    ) -> list[Section | None]:
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

        def search_recursive(nodes: list[Section], target: str) -> Section | None:
            for node in nodes:
                if compare(node.title, target):
                    return node
                found = search_recursive(node.children, target)
                if found:
                    return found
            return None

        results: list[Section | None] = []
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

    def tree_view_json(self, content_limit: int = 0) -> list[SectionTreeJson]:
        """Return a tree-like view of all sections in JSON format."""
        return [section.tree_view_json(content_limit) for section in self.sections]

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

    def to_json(
        self,
        table_limit: Optional[int] = None,
        rows_limit: Optional[int] = None,
        keep_links: bool = True,
        content_limit: Optional[int] = None,
        content_start: int = 0,
        as_paragraphs: bool = False,
        show_children: bool = True,
    ) -> list[SectionJson]:
        """Convert the entire section tree to JSON format.
        Args:
            table_limit: Maximum number tables to include per section. If None, include all tables.
            rows_limit: Maximum number of rows to include per table. If None, include all rows.
            keep_links: Whether to keep hyperlinks in the content.
            content_limit: Maximum number of characters to include from the section content. If None, include all content.
            content_start: Starting character index for the content.
            as_paragraphs: Whether to return content as a list of paragraphs instead of a single string.
            show_children: Whether to show per section like a tree (with children) or a flat list.
        """
        ress = [
            section.to_json(
                table_limit=table_limit,
                rows_limit=rows_limit,
                keep_links=keep_links,
                content_limit=content_limit,
                content_start_index=content_start,
                as_paragraphs=as_paragraphs,
                show_children=show_children,
            )
            for section in self.sections
        ]

        return ress


def get_summary(content: str) -> str:
    # before headings
    pat = re.compile(r"^(={2,})\s*(.*?)\s*\1$", re.MULTILINE)
    match = pat.search(content)
    if match:
        return content[: match.start()].strip()

    return content.strip()[:500] + ("..." if len(content) > 500 else "")
