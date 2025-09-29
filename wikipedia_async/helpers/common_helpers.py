from urllib.parse import urlparse, parse_qsl, urlunparse, urlencode
from typing import Any
import re
import pandas as pd

def normalize_url(url: str) -> str:
    """Normalize a URL by removing fragments and sorting query parameters."""
    parsed = urlparse(url)
    normalized = parsed._replace(fragment="").geturl()
    if parsed.query:
        query_params = parse_qsl(parsed.query)
        sorted_query = urlencode(sorted(query_params))
        normalized = urlunparse(
            parsed._replace(query=sorted_query, fragment="")
        )
    return normalized


def ensure_serializable(data: Any) -> Any:
    """Ensure the data is JSON serializable by converting non-serializable types to strings."""
    if isinstance(data, dict):
        return {str(k): ensure_serializable(v) for k, v in data.items()}
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