from functools import lru_cache
import re
from bs4 import BeautifulSoup


@lru_cache(maxsize=10)
def clean_html_table(html: str) -> str:
    # 1. Fix rowspan / colspan attributes
    def fix_span_attr(match):
        attr = match.group(1)  # "rowspan" or "colspan"
        val = match.group(2)  # inside quotes
        # Extract digits
        digits = re.findall(r"\d+", val)
        if digits:
            return f'{attr}="{digits[0]}"'
        else:
            return f'{attr}="1"'

    html = re.sub(r'(rowspan|colspan)="([^"]*)"', fix_span_attr, html)

    # 2. Fix attributes without quotes (rowspan=3 → rowspan="3")
    html = re.sub(r"\b(rowspan|colspan)=(\d+)", r'\1="\2"', html)

    # 3. Remove span attributes with percentage or invalid values
    html = re.sub(r'(rowspan|colspan)="\d+%"', "", html)

    # 4. Remove HTML comments
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    # 5. Fix broken entities
    html = html.replace("&nbsp", "&nbsp;")

    # 6. Use BeautifulSoup to reserialize and fix tag nesting
    try:
        # Use a lenient parser (html5lib) if available for broken markup
        soup = BeautifulSoup(html, "html5lib")
    except Exception:
        soup = BeautifulSoup(html, "lxml")

    # Optionally: drop unwanted attributes from <td>/<th>
    for tag in soup.find_all(["td", "th"]):
        # Remove unknown or custom attributes
        allowed = {"rowspan", "colspan", "align", "valign", "style"}
        for attr in list(tag.attrs):
            if attr not in allowed:
                del tag.attrs[attr]

    return str(soup)
