from wikipedia_async.helpers.section_helpers import Section
from typing import List
import re

from wikipedia_async.models.section_models import Paragraph


def parse_sections(content: str) -> List[Section]:
    """
    Parse Wikipedia content and create a tree-like structure of sections.

    Args:
        content: The Wikipedia page content as a string

    Returns:
        List of root-level Section objects representing the section hierarchy
    """
    # Match section headers with varying levels (==, ===, ====, etc.)
    pat = re.compile(r"^(={2,})\s*(.*?)\s*\1$", re.MULTILINE)
    matches = list(pat.finditer(content))

    if not matches:
        # No sections found, return the entire content as a single section
        return [
            Section(
                title="Content",
                level=0,
                section_paragraphs=[Paragraph(text=content.strip())],
            )
        ]

    sections = []
    stack: list[Section] = []  # Stack to keep track of parent sections

    for i, match in enumerate(matches):
        # Calculate section level based on number of equals signs
        equals_count = len(match.group(1))
        level = equals_count - 1  # Level 1 for ==, Level 2 for ===, etc.
        title = match.group(2).strip()

        # Extract content for this section
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_content = content[start_pos:end_pos].strip()
        paras = section_content.split("\n\n")

        # Create new section node
        section = Section(
            title=title,
            level=level,
            section_paragraphs=[Paragraph(text=p.strip()) for p in paras if p.strip()],
        )

        stack.append(section)

    paras = content[: matches[0].start()].strip().split("\n\n")
    sections.append(
        Section(
            title="*",
            level=0,
            section_paragraphs=[Paragraph(text=p.strip()) for p in paras if p.strip()],
        )
    )

    for i, s in enumerate(stack):
        if s.level > 1:
            sections[-1].add_child(s)
        else:
            sections.append(s)

    return sections
