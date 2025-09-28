from pydantic import BaseModel
from pydantic import Field
from wikipedia_async.models.section_models import (
    Link,
    Paragraph,
    Section,
    SectionJson,
    SectionSearchResult,
    SectionSnippet,
    SectionTreeJson,
    Table,
    TableSearchResult,
)
from wikipedia_async.helpers.content_helpers import parse_sections
from wikipedia_async.helpers.html_parsers import parse_wiki_html
from typing import Any, Literal, Optional, overload
from functools import cached_property
import re


class SectionHelper(BaseModel):
    """Helper class to manage and interact with a list of Section objects."""

    sections: list[Section] = Field(
        default_factory=list, description="List of root-level sections"
    )

    class Config:
        frozen = True

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

    @overload
    def __getitem__(self, i: str, _default: Optional[Any]) -> Section | None: ...

    @overload
    def __getitem__(self, i: str, _default: Section) -> Section: ...

    def __getitem__(self, i, _default: Optional[Section] = None):
        if isinstance(i, slice):
            return SectionHelper(self.sections[i])
        elif isinstance(i, int):
            return self.sections[i]
        elif isinstance(i, str):
            section = self.get_section_by_title(i, prioritize_top_level=True)
            if section is None and _default is not None:
                return _default
            return section
        else:
            raise TypeError("Invalid argument type.")

    def get_section_by_title(
        self,
        title: str,
        parent_title: Optional[str] = None,
        case_sensitive: bool = False,
        prioritize_top_level: bool = False,
    ) -> Section | None:
        """Find a section by its title.
        Args:
            title: Title of the section to find. Supports regex.
            parent_title: If provided, only consider sections whose parent has this title.
            case_sensitive: Whether the search is case-sensitive.
            prioritize_top_level: If True, prefer matches among root-level sections
                (self.sections) before searching the whole tree.
        Returns:
            The matching Section object, or None if not found.
        """

        def compare(t1: str, t2: str) -> bool:
            flags = 0 if case_sensitive else re.IGNORECASE
            return re.search(t1, t2, flags) is not None

        def search_recursive(nodes: list[Section]) -> Section | None:
            for node in nodes:
                if parent_title:
                    if not node.parent or not compare(node.parent.title, parent_title):
                        continue
                if compare(node.title, title):
                    return node
                result = search_recursive(node.children)
                if result:
                    return result
            return None

        # Try root-level match first if requested
        if prioritize_top_level:
            for node in self.sections:
                if compare(node.title, title):
                    if parent_title:
                        if node.parent and compare(node.parent.title, parent_title):
                            return node
                    else:
                        return node

        # Otherwise search the whole tree
        return search_recursive(self.sections)

    def get_table_by_caption(
        self,
        caption: str,
        case_sensitive: bool = False,
        section: Optional[Section | str] = None,
    ) -> Optional[Table]:
        """Find a table by its caption.
        Args:
            caption: Caption of the table to find. Supports regex.
            case_sensitive: Whether the search is case-sensitive.
            section: If provided, limit search to this section (by title or Section object).
        Returns:
            The matching Table object, or None if not found.
        """

        def compare(t1: str, t2: str) -> bool:
            flags = 0 if case_sensitive else re.IGNORECASE
            return re.search(t1, t2, flags) is not None

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

    def find_sections_containing(
        self,
        text: str,
        section: Optional[Section | str] = None,
        case_sensitive: bool = False,
        by: Optional[Literal["title", "content", "all"]] = "title",
        surrounding_text_length: int = 30,
    ) -> list[SectionSearchResult]:
        """Find a section containing the given text.
        Args:
            text: Text to search for.
            section: If provided, limit search to this section (by title or Section object).
            case_sensitive: Whether the search is case-sensitive.
            by: Where to search for the text: "title", "content", or "all".
            surrounding_text_length: Number of characters to include before and after the match in the snippet.
        Returns:
            List of SectionSearchResult objects that contain the sections with matching text and snippets.
        """
        secs = []
        if section:
            title: str = section.title if isinstance(section, Section) else section
            target_section = self.get_section_by_title(title)
            if not target_section:
                return []
            secs = [target_section]
        else:
            secs = self.sections

        res = []
        for sec in secs:
            snippets = []
            srange = []
            for s in [sec] + sec.children:
                haystack = ""
                title_start_index = 0
                title_end_index = 0
                if by in ("title", "all"):
                    haystack += s.title + "\n"
                    title_end_index = len(s.title)
                if by in ("content", "all"):
                    haystack += s.section_content + "\n"

                needle = text
                flags = 0 if case_sensitive else re.IGNORECASE
                matches = list(re.finditer(needle, haystack, flags))
                for match in matches:
                    start, end = match.span()
                    for surrounding in srange:
                        if start >= surrounding[0] and end <= surrounding[1]:
                            break
                    else:
                        found_in = "content"
                        if start >= title_start_index and end <= title_end_index:
                            found_in = "title"

                        snippet_start = max(0, start - surrounding_text_length)
                        snippet_end = min(len(haystack), end + surrounding_text_length)
                        snippet = haystack[snippet_start:snippet_end]
                        snippets.append(
                            SectionSnippet(
                                start_index=start,
                                end_index=end,
                                snippet=snippet,
                                found_in=found_in,
                            )
                        )
                        srange.append((start, end))

            if snippets:
                res.append(SectionSearchResult(section=s, snippets=snippets))

        return res

    def find_tables_containing(
        self,
        text: str,
        section: Optional[Section | str] = None,
        table: Optional[Table | str] = None,
        case_sensitive: bool = False,
        column_name: Optional[str] = None,
        by: Literal[
            "caption",
            "column_name",
            "cell_content",
            "caption+column_name",
            "caption+cell_content",
            "column_name+cell_content",
            "all",
        ] = "caption",
    ) -> list[TableSearchResult]:
        """Find tables containing the given text.
        Args:
            text: Text to search for.
            section: If provided, limit search to this section (by title or Section object).
            table: If provided, limit search to this table (by caption or Table object).
            case_sensitive: Whether the search is case-sensitive.
            column_name: If searching by column name, the specific column name to look for.
            by: Where to search for the text: "caption", "column_name", or "cell_content".
        Returns:
            List of TableSearchResult objects that contain the tables with matching text and relevant rows.
        """
        sec = None
        if section:
            if isinstance(section, str):
                sec = self.get_section_by_title(section)
            else:
                sec = section

        # tables = sec.tables if sec else self.tables
        tables = []
        if table:
            if isinstance(table, str):
                tbl = self.get_table_by_caption(
                    table, case_sensitive=case_sensitive, section=sec
                )
                if tbl:
                    tables = [tbl]
            else:
                tables = [table]
        else:
            tables = sec.tables if sec else self.tables

        res = []
        for table in tables:
            r = None
            if "caption" in by or by == "all":
                caption = table.caption or ""
                flags = 0 if case_sensitive else re.IGNORECASE
                if re.search(text, caption, flags):
                    r = TableSearchResult(
                        table=table,
                        found_in="caption",
                        rows=[],
                        total_rows=len(table),
                    )

            if not r and ("column_name" in by or by == "all"):
                for col in table.headers:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    if re.search(text, str(col), flags):
                        r = TableSearchResult(
                            table=table,
                            found_in="column_name",
                            rows=[],
                            total_rows=len(table),
                        )
                        break

            elif not r and ("cell_content" in by or by == "all"):
                rows = []
                for row in table.records:
                    for col, cell in row.items():
                        if column_name and col != column_name:
                            continue
                        flags = 0 if case_sensitive else re.IGNORECASE
                        if isinstance(cell, str) and re.search(text, cell, flags):
                            rows.append(row)
                            break
                if rows:
                    r = TableSearchResult(
                        table=table,
                        found_in="cell_value",
                        rows=rows,
                        total_rows=len(table),
                    )
            if r:
                res.append(r)

        return res

    def tree_view(self, content_limit: int = 0) -> str:
        """Return a tree-like view of all sections."""
        text = ""
        for section in self.sections:
            text += section.tree_view(content_limit)
        return text.strip()

    def tree_view_json(self, content_limit: int = 0) -> list[SectionTreeJson]:
        """Return a tree-like view of all sections in JSON format."""
        return [section.tree_view_json(content_limit) for section in self.sections]

    @cached_property
    def tables(self) -> list[Table]:
        """Get a flat list of all tables in all sections."""
        tables = []
        for section in self.iter_sections():
            tables.extend(section.tables)
        return tables

    @cached_property
    def content(self) -> str:
        """Get the combined content of all sections."""
        return "\n".join(section.content for section in self.sections)

    @cached_property
    def paragraphs(self) -> list[Paragraph]:
        """Get a flat list of all paragraphs in all sections."""
        return [p for section in self.sections for p in section.paragraphs]

    @cached_property
    def links(self) -> list[Link]:
        """Get a flat list of all links in all sections."""
        return [link for section in self.sections for link in section.links]

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

    def to_string(self, markdown: bool = False) -> str:
        """Convert the entire section tree to a string."""
        return (
            "---\n"
            if markdown
            else "\n\n".join(
                section.to_string(markdown=markdown) for section in self.sections
            )
        )

    def to_json(
        self,
        table_limit: Optional[int] = None,
        rows_limit: Optional[int] = None,
        keep_links: bool = False,
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
