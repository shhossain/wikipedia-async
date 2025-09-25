import uuid
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from typing import TypedDict


class ParagraphJson(TypedDict):
    text: str
    links: list[dict[str, str]]


class TableJson(TypedDict):
    caption: Optional[str]
    headers: list[str]
    records: list[dict[str, Any]]
    links: list[str]


class SectionJson(TypedDict):
    title: str
    level: int
    paragraphs: list[ParagraphJson]
    tables: list[TableJson]
    children: list["SectionJson"]


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

    def to_json(self) -> ParagraphJson:
        return {
            "text": self.text,
            "links": [link.to_json() for link in self.links],
        }


class Table(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    html: str
    caption: Optional[str] = None
    mem: dict = Field(default_factory=dict, repr=False)
    links: list[Link] = Field(default_factory=list)

    @property
    def dataframe(self):
        if self.mem.get("dataframe") is not None:
            return self.mem["dataframe"]

        from io import StringIO

        import pandas as pd

        dfs = pd.read_html(StringIO(self.html))
        if dfs:
            df = dfs[0]
            # Make all nan to empty string
            df = df.fillna("")
            self.mem["dataframe"] = df
            return df

        return pd.DataFrame()

    @property
    def records(self) -> list[dict[str, Any]]:
        df = self.dataframe
        return df.to_dict(orient="records") # type: ignore

    @property
    def headers(self):
        df = self.dataframe
        return df.columns.tolist()

    def to_string(self, markdown: bool = False) -> str:
        df = self.dataframe
        if markdown:
            return df.to_markdown(index=False)
        return df.to_string(index=False)

    def to_json(self) -> TableJson:
        return {
            "caption": self.caption,
            "headers": self.headers,
            "records": self.records,
            "links": [str(link) for link in self.links],
        }


class Section(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    level: int
    paragraphs: list[Paragraph] = Field(default_factory=list)
    parent: Optional["Section"] = Field(default=None, repr=False)
    children: list["Section"] = Field(default_factory=list, repr=False)
    section_tables: list[Table] = Field(default_factory=list)

    @property
    def section_content(self) -> str:
        """Return current section content"""
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def section_links(self) -> list[Link]:
        links = []
        for p in self.paragraphs:
            links.extend(p.links)
        return links

    def section_to_string(self) -> str:
        text = f"# {self.title}"
        for p in self.paragraphs:
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

    def to_string(self, markdown: bool = False) -> str:
        # text = f"{self.title}\n\n"
        text = ""
        if markdown:
            text += "#" * (self.level) + " "
        text += f"{self.title}\n"

        for p in self.paragraphs:
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

    def to_json(self) -> SectionJson:
        return {
            "title": self.title,
            "level": self.level,
            "paragraphs": [p.to_json() for p in self.paragraphs],
            "tables": [t.to_json() for t in self.section_tables],
            "children": [child.to_json() for child in self.children],
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

    def __repr__(self):
        return f"Section(title='{self.title}', level={self.level}, children={len(self.children)}, paragraphs={len(self.paragraphs)}, tables={len(self.section_tables)})"

    def __str__(self):
        return self.to_string(markdown=False)
