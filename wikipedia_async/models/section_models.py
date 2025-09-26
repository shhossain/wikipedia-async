import uuid
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from typing import TypedDict
from io import StringIO
import pandas as pd
import re

class ParagraphJson(TypedDict):
    text: str
    links: list[dict[str, str]]


class TableJson(TypedDict):
    caption: Optional[str]
    headers: list[str]
    records: list[dict[str, Any]]
    links: list[str]
    total_rows: int
    rows_limit: Optional[int]
    rows_start_index: Optional[int]


class SectionContentJson(TypedDict):
    text: str
    total_length: int
    start_index: int
    content_limit: Optional[int]
    is_content_ended: bool


class SectionJson(TypedDict):
    title: str
    level: int
    paragraphs: list[ParagraphJson]
    content: Optional[SectionContentJson]
    tables: list[TableJson]
    children: list["SectionJson"]


class TablePreviewJson(TypedDict):
    headers: list[str]
    caption: Optional[str]
    total_rows: int
    first_row: dict[str, Any]


class ContentPreviewJson(TypedDict):
    text: str
    total_length: int


class SectionTreeJson(TypedDict):
    title: str
    content_preview: ContentPreviewJson
    children: list["SectionTreeJson"]
    tables_preview: list[TablePreviewJson]


class TableConfig(TypedDict, total=False):
    rows_limit: Optional[int]
    rows_start_index: Optional[int]
    cols: Optional[list[str]]
    exclude_cols: Optional[list[str]]
    exclude_empty: Optional[bool]
    keep_links: Optional[bool]


class Link(str):
    url_title: str
    parsed_url: Any

    def __new__(cls, url: str, title: str = ""):
        obj = str.__new__(cls, url)
        # store the title on the instance; str subclasses can have attributes
        obj.url_title = title
        obj.parsed_url = urlparse(url)
        return obj

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))

    @property
    def domain(self) -> str:
        """Return the domain of the URL."""
        return self.parsed_url.netloc

    @property
    def path(self) -> str:
        """Return the path of the URL."""
        return self.parsed_url.path

    @property
    def last_path_segment(self) -> str:
        """Return the last segment of the path of the URL."""
        return (
            self.parsed_url.path.rstrip("/").split("/")[-1]
            if self.parsed_url.path
            else ""
        )

    @property
    def query_params(self) -> dict:
        """Return the query parameters of the URL as a dictionary."""
        return {
            k: v[0] if len(v) == 1 else v
            for k, v in parse_qs(self.parsed_url.query).items()
        }

    @property
    def scheme(self) -> str:
        """Return the scheme of the URL."""
        return self.parsed_url.scheme

    @property
    def query(self) -> str:
        """Return the query of the URL."""
        return self.parsed_url.query

    @property
    def fragment(self) -> str:
        """Return the fragment of the URL."""
        return self.parsed_url.fragment

    @property
    def url(self) -> str:
        """Return the URL string for compatibility with callers expecting `.url`."""
        return str(self)

    def to_string(self, markdown: bool = False) -> str:
        title = self.url_title or self.last_path_segment
        return f"[{title}]({self.url}) " if markdown else f"{title} ({self.url}) "

    def to_json(self) -> dict[str, str]:
        return {
            "title": self.url_title or self.last_path_segment,
            "url": self.url,
        }


class Paragraph(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    links: list[Link] = Field(default_factory=list)
    list_items: list[str] = Field(default_factory=list)

    def to_string(self, markdown: bool = False) -> str:
        text = self.text
        # Build link text using a list to avoid repeated string allocation
        link_text_items = []
        for link in self.links:
            if markdown and link.url_title in text:
                text = text.replace(
                    link.url_title, link.to_string(markdown=markdown), 1
                )
            elif markdown and link.last_path_segment in text:
                text = text.replace(
                    link.last_path_segment, link.to_string(markdown=markdown), 1
                )
            else:
                link_text_items.append(link.to_string(markdown=markdown))

        link_text = ", ".join(link_text_items)
        return text + ("\nLinks: " + link_text if link_text else "")

    def to_json(self, keep_links: bool = True) -> ParagraphJson:
        return {
            "text": self.text,
            "links": [link.to_json() for link in self.links] if keep_links else [],
        }


def ensure_serializable(data: Any) -> Any:
    """Ensure the data is JSON serializable by converting non-serializable types to strings."""
    if isinstance(data, dict):
        return {k: ensure_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, set, tuple)):
        return [ensure_serializable(i) for i in data]
    elif isinstance(data, (str, int, float, bool)) or data is None:
        return data
    elif pd.isna(data):
        return None
    elif hasattr(data, "item"):
        return data.item()
    elif hasattr(data, "isoformat"):
        return data.isoformat()
    else:
        return str(data)
    
def clean_string(s: Any) -> str:
    if not isinstance(s, str):
        return s
    try:
        # Replace common whitespace-like chars with normal space
        s = s.replace("\xa0", " ").replace("\u202f", " ").replace("\ufeff", "")
        # Remove zero-width characters and soft hyphen
        s = re.sub(r"[\u200b\u00ad]", "", s)
    except Exception:
        pass
    return s

class Table(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    html: str
    caption: str
    mem: dict = Field(default_factory=dict, repr=False)
    links: list[Link] = Field(default_factory=list)

    @property
    def dataframe(self):
        if self.mem.get("dataframe") is not None:
            return self.mem["dataframe"]

        dfs = pd.read_html(StringIO(self.html))
        if dfs:
            df = dfs[0]
            # Make all nan to empty string
            df = df.fillna("")
            df = df.applymap(clean_string) # type: ignore
            self.mem["dataframe"] = df
            return df

        return pd.DataFrame()

    @property
    def records(self) -> list[dict[str, Any]]:
        df = self.dataframe
        res: list[dict[str, Any]] = df.to_dict(orient="records")  # type: ignore
        for i, r in enumerate(res):
            r["idx"] = i + 1
        res = ensure_serializable(res)
        return res

    @property
    def headers(self):
        df = self.dataframe
        return df.columns.tolist()

    def to_string(self, markdown: bool = False) -> str:
        df = self.dataframe
        if markdown:
            return df.to_markdown(index=False)
        return df.to_string(index=False)

    def to_json(
        self,
        rows_limit: Optional[int] = None,
        rows_start_index: Optional[int] = None,
        cols: Optional[list[str]] = None,
        exclude_cols: Optional[list[str]] = None,
        exclude_empty: bool = False,
        keep_links: bool = True,
    ) -> TableJson:

        start = rows_start_index or 0
        end = start + rows_limit if rows_limit else len(self.records)
        records = self.records[start:end]

        if cols:
            records = [{col: rec.get(col, "") for col in cols} for rec in records]

        if exclude_cols:
            records = [
                {col: val for col, val in rec.items() if col not in exclude_cols}
                for rec in records
            ]

        if exclude_empty:
            def _empty(v):
                if v is None:
                    return True
                if isinstance(v, str):
                    return v.strip() == ""
                if isinstance(v, (list, tuple, set, dict)):
                    return len(v) == 0
                return False

            # remove empty fields from each record
            records = [
                {k: v for k, v in rec.items() if not _empty(v)} for rec in records
            ]

        return {
            "caption": self.caption,
            "total_rows": len(self.records),
            "rows_limit": rows_limit,
            "rows_start_index": start,
            "headers": self.headers,
            "records": records,
            "links": [str(link) for link in self.links] if keep_links else [],
        }


class Section(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    level: int
    section_paragraphs: list[Paragraph] = Field(default_factory=list)
    parent: Optional["Section"] = Field(default=None, repr=False)
    children: list["Section"] = Field(default_factory=list, repr=False)
    section_tables: list[Table] = Field(default_factory=list)

    @property
    def section_content(self) -> str:
        """Return current section content"""
        return "\n\n".join(p.text for p in self.section_paragraphs)

    @property
    def section_links(self) -> list[Link]:
        links = []
        for p in self.section_paragraphs:
            links.extend(p.links)
        return links

    def section_to_string(self) -> str:
        text = f"# {self.title}"
        for p in self.section_paragraphs:
            text += f"\n\n{p.to_string()}"

        for table in self.section_tables:
            text += f"\n\nTable: {table.caption if table.caption else self.title}\n{table.to_string()}"
        return text

    @property
    def content(self) -> str:
        return (
            self.section_content
            + "\n\n"
            + "\n\n".join(child.content for child in self.children)
        ).strip()

    @property
    def links(self) -> list[Link]:
        links = self.section_links
        for child in self.children:
            links.extend(child.links)
        return links

    @property
    def tables(self) -> list[Table]:
        tables = self.section_tables
        for child in self.children:
            tables.extend(child.tables)
        return tables

    @property
    def paragraphs(self) -> list[Paragraph]:
        paragraphs = self.section_paragraphs
        for child in self.children:
            paragraphs.extend(child.paragraphs)
        return paragraphs

    def to_string(self, markdown: bool = False) -> str:
        # text = f"{self.title}\n\n"
        text = ""
        if markdown:
            text += "#" * (self.level) + " "
        text += f"{self.title}\n"

        for p in self.section_paragraphs:
            text += f"{p.to_string(markdown=markdown)}\n"

        for child in self.children:
            text += f"{child.to_string(markdown=markdown)}\n"

        for table in self.section_tables:
            caption = table.caption or self.title
            ll = []
            for link in table.links:
                if markdown and link.url_title in caption:
                    caption = caption.replace(
                        link.url_title, link.to_string(markdown=markdown), 1
                    )
                elif markdown and link.last_path_segment in caption:
                    caption = caption.replace(
                        link.last_path_segment, link.to_string(markdown=markdown), 1
                    )
                else:
                    ll.append(link.to_string(markdown=markdown))

            text += f"\n\nTable: {caption}\n{table.to_string(markdown=markdown)}"
            if ll:
                text += "\nLinks: " + ", ".join(ll)
                text += "\n"

        return text

    def to_json(
        self,
        table_limit: Optional[int] = None,
        rows_limit: Optional[int] = None,
        keep_links: bool = True,
        content_limit: Optional[int] = None,
        content_start_index: int = 0,
        as_paragraphs: bool = False,
        show_children: bool = True,
    ) -> SectionJson:
        """Convert the section to JSON format.
        Args:
            table_limit: Maximum number tables to include. If None, include all tables.
            keep_links: Whether to keep hyperlinks in the content.
            content_limit: Maximum number of characters to include from the section content. If None, include all content.
            content_start_index: Starting character index for the content.
            as_paragraphs: Whether to return content as a list of paragraphs instead of a single string.
            show_children: Whether to show per section like a tree (with children) or a flat list.
        """

        content = self.content
        if content_limit and content_limit > 0:
            content = content[content_start_index : content_start_index + content_limit]
            as_paragraphs = False
            show_children = False

        return {
            "title": self.title,
            "level": self.level,
            "paragraphs": (
                [
                    p.to_json(keep_links=keep_links)
                    for p in (
                        self.section_paragraphs if show_children else self.paragraphs
                    )
                ]
                if as_paragraphs
                else []
            ),
            "content": (
                {
                    "text": content,
                    "total_length": len(self.content),
                    "start_index": content_start_index,
                    "content_limit": content_limit,
                    "is_content_ended": (
                        True
                        if content_limit is None
                        else (content_start_index + content_limit) >= len(self.content)
                    ),
                }
                if not as_paragraphs
                else None
            ),
            "tables": [
                t.to_json(keep_links=keep_links, rows_limit=rows_limit)
                for t in (
                    self.section_tables
                    if show_children
                    else self.tables[:table_limit] if table_limit else self.tables
                )
            ],
            "children": (
                [child.to_json(keep_links=keep_links) for child in self.children]
                if show_children
                else []
            ),
        }

    def add_child(self, child: "Section"):
        self.children.append(child)
        child.parent = self

    def tree_view(self, content_limit: int = 0) -> str:
        """Return a tree-like view of the section and its children."""

        def render_section(section: Section, level: int = 0) -> str:
            indent = "  " * level
            content_preview = section.section_content[:content_limit] + (
                "..."
                if content_limit and len(section.section_content) > content_limit
                else ""
            )
            result = f"{indent}- {section.title}"
            if content_preview:
                result += f":\n{indent} {content_preview}"
            result += "\n"
            for child in section.children:
                result += render_section(child, level + 1)
            result += "\n" if content_preview else ""
            return result

        return render_section(self)

    def tree_view_json(self, content_limit: int = 0) -> SectionTreeJson:
        """Return a tree-like view of the section and its children as JSON."""

        def render_section_json(section: Section) -> SectionTreeJson:
            content_preview = section.section_content[:content_limit] + (
                "..."
                if content_limit and len(section.section_content) > content_limit
                else ""
            )
            return {
                "title": section.title,
                "content_preview": {
                    "text": content_preview,
                    "total_length": len(section.section_content),
                },
                "children": [render_section_json(child) for child in section.children],
                "tables_preview": [
                    {
                        "headers": table.headers,
                        "caption": table.caption,
                        "total_rows": len(table.records),
                        "first_row": table.records[0] if table.records else {},
                    }
                    for table in section.section_tables
                ],
            }

        return render_section_json(self)

    def __repr__(self):
        return f"Section(title='{self.title}', level={self.level}, children={len(self.children)}, paragraphs={len(self.section_paragraphs)}, tables={len(self.section_tables)})"

    def __str__(self):
        return self.to_string(markdown=False)
