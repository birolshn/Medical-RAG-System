from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
QUERY_EXPANSIONS = {
    "çocuklarda": "pediatric children",
    "çocuk": "child pediatric",
    "akut": "acute",
    "tedavisi": "treatment therapy management",
    "nasıl yapılır": "guideline recommended care",
    "çölyak": "celiac",
    "hastalığı": "disease",
    "tanı": "diagnosis",
    "kriterleri": "criteria",
    "nelerdir": "definition recommendations",
}


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def expand_query(query: str) -> str:
    lowered = query.lower()
    additions = [expansion for source, expansion in QUERY_EXPANSIONS.items() if source in lowered]
    if not additions:
        return query
    return normalize_whitespace(f"{query} {' '.join(additions)}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
