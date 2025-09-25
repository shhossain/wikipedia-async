from io import StringIO
from typing import Any
from bs4 import BeautifulSoup
import pandas as pd


def extract_tables_from_html(html: str) -> dict[str, list[dict[str, Any]]]:
    """Extract tables from HTML content and return as a dictionary.

    Args:
        html (str): The HTML content to parse.
    Returns:
        dict[str, dict[str, Any]]: A dictionary where keys are table headings
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    extracted_tables = {}

    for i, table in enumerate(tables):
        heading = table.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
        heading_text = heading.text.strip() if heading else f"Table_{i+1}"
        dfs = pd.read_html(StringIO(str(table)))
        if dfs:
            if heading_text not in extracted_tables:
                extracted_tables[heading_text] = []
            extracted_tables[heading_text].append(
                dfs[0].fillna("").to_dict(orient="records")
            )

    return extracted_tables
