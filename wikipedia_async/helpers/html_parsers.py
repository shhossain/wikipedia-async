from typing import Any, Dict, Optional
from bs4 import BeautifulSoup
from wikipedia_async.models.section_models import Section, Paragraph, Table, Link
import re


# for type checking
class DemoTag:
    """A minimal Tag-like class for type hinting purposes."""

    name: Optional[str]
    attrs: Dict[str, Any]

    def get(self, key: str, default=None) -> Any: ...
    def find(
        self, name: Any, attrs: Optional[Dict[str, Any]] = None
    ) -> Optional["DemoTag"]: ...
    def find_all(
        self, name: Any, attrs: Optional[Dict[str, Any]] = None
    ) -> list["DemoTag"]: ...
    @property
    def children(self) -> Any: ...


def parse_wiki_html(html: str) -> list[Section]:
    soup = BeautifulSoup(html, "html.parser")
    content_div: DemoTag = soup.find("div", {"class": "mw-parser-output"})  # type: ignore
    if not content_div:
        return []

    sections: list[Section] = [
        Section(title="Introduction", level=1, section_paragraphs=[]),
    ]
    prev_section = sections[0]
    for elem in content_div.children:
        if not hasattr(elem, "name") or elem.name is None:
            continue

        name = elem.name.lower()
        classes = elem.get("class", [])
        is_heading = any(c.startswith("mw-heading") for c in classes)
        if name == "div" and is_heading:
            # It's a heading
            h_tag = elem.find(re.compile("^h[1-6]$"))
            if not h_tag:
                continue

            level = int(h_tag.name[1]) - 1  # h2 -> level 1
            title = h_tag.get_text(strip=True)
            # print(f"Found heading: {title} (Level {level})")

            section = Section(title=title, level=level)
            prev_section = section
            if level > 1:
                # Find the nearest parent section with level < current level
                for prev_sec in reversed(sections):
                    if prev_sec.level < level:
                        prev_sec.add_child(section)
                        break
            else:
                sections.append(section)

        elif name == "p":
            if not sections:
                continue

            paragraph_text = elem.get_text(separator=" ", strip=True)
            if not paragraph_text:
                continue

            links = extract_links(elem)

            paragraph = Paragraph(text=paragraph_text, links=links)
            prev_section.section_paragraphs.append(paragraph)

        elif (name == "div" and elem.find("li")) or name == "ol" or name == "ul":
            lis = elem.find_all("li")
            if not lis:
                continue

            paragraph_text = "\n".join(
                li.get_text(separator=" ", strip=True) for li in lis
            ).strip()
            if not paragraph_text:
                continue

            links = extract_links(elem)

            paragraph = Paragraph(
                text=paragraph_text,
                list_items=[li.get_text(separator=" ", strip=True) for li in lis],
                links=links,
            )
            # sections[-1].paragraphs.append(paragraph)
            prev_section.section_paragraphs.append(paragraph)

        elif name == "table" and "wikitable" in classes:
            # It's a table
            table_html = str(elem)
            caption_tag = elem.find("caption")
            caption = (
                caption_tag.get_text(strip=True) if caption_tag else prev_section.title
            )
            table = Table(html=table_html, caption=caption)
            prev_section.section_tables.append(table)

            # Extract links from the table
            table.links = extract_links(elem)

    return sections


def extract_links(elem):
    links = []
    for a in elem.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#") or href.startswith("/wiki/"):
            full_url = (
                "https://en.wikipedia.org" + href
                if href.startswith("/wiki/")
                else "https://en.wikipedia.org/wiki/" + href[1:]
            )
            title = a.get("title", "")
            links.append(Link(full_url, title.strip()))

    return links



