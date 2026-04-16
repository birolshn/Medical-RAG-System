from __future__ import annotations

import csv
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from medical_rag.utils import normalize_whitespace, write_json

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedClient:
    def __init__(self, email: str | None = None, tool: str = "medical-rag-system") -> None:
        self.email = email or os.getenv("PUBMED_EMAIL")
        self.tool = tool
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "medical-rag-system/1.0"})
        self._last_request_time = 0.0
        self._min_interval_seconds = 1 / 3

    def _wait_for_slot(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        remaining = self._min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _request(self, url: str, params: dict[str, Any], expect_json: bool = False) -> Any:
        base_params = {"tool": self.tool}
        if self.email:
            base_params["email"] = self.email
        base_params.update(params)
        last_error: Exception | None = None
        for attempt in range(4):
            self._wait_for_slot()
            try:
                response = self.session.get(url, params=base_params, timeout=30)
                self._last_request_time = time.monotonic()
                if response.status_code == 429:
                    time.sleep(1 + attempt)
                    continue
                response.raise_for_status()
                return response.json() if expect_json else response.text
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                time.sleep(1 + attempt)
        raise RuntimeError(f"Request failed for {url}: {last_error}")

    def search(self, term: str, retmax: int = 20) -> list[str]:
        payload = self._request(
            ESEARCH_URL,
            {
                "db": "pubmed",
                "term": term,
                "retmode": "json",
                "sort": "pub date",
                "retmax": retmax,
            },
            expect_json=True,
        )
        return payload.get("esearchresult", {}).get("idlist", [])

    def fetch(self, pmids: list[str]) -> str:
        return self._request(
            EFETCH_URL,
            {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        )


def load_terms(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["term"].strip() for row in reader if row.get("term")]


def parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    articles: list[dict[str, Any]] = []
    for entry in root.findall(".//PubmedArticle"):
        pmid = text_or_default(entry.find(".//MedlineCitation/PMID"))
        title = text_or_default(entry.find(".//ArticleTitle"))
        abstract = extract_abstract(entry)
        journal = normalize_whitespace(entry.findtext(".//Journal/Title", default="").strip())
        year = extract_year(entry)
        doi = extract_doi(entry)
        first_author = extract_first_author(entry)
        if not pmid or not title or not abstract:
            continue
        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "first_author": first_author,
                "journal": journal,
                "year": year,
                "doi": doi,
            }
        )
    return articles


def text_or_default(node: ET.Element | None, default: str = "") -> str:
    return normalize_whitespace("".join(node.itertext()).strip()) if node is not None else default


def extract_abstract(entry: ET.Element) -> str:
    parts: list[str] = []
    for node in entry.findall(".//Abstract/AbstractText"):
        label = node.attrib.get("Label", "").strip()
        text = normalize_whitespace("".join(node.itertext()).strip())
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


def extract_year(entry: ET.Element) -> int | None:
    candidates = [
        entry.findtext(".//JournalIssue/PubDate/Year"),
        entry.findtext(".//ArticleDate/Year"),
        entry.findtext(".//PubDate/MedlineDate"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        match = re.search(r"(19|20)\d{2}", candidate)
        if match:
            return int(match.group())
    return None


def extract_doi(entry: ET.Element) -> str | None:
    for node in entry.findall(".//ArticleIdList/ArticleId"):
        if node.attrib.get("IdType") == "doi":
            value = normalize_whitespace("".join(node.itertext()).strip())
            if value:
                return value
    return None


def extract_first_author(entry: ET.Element) -> str | None:
    author = entry.find(".//AuthorList/Author")
    if author is None:
        return None
    collective = author.findtext("CollectiveName")
    if collective:
        return normalize_whitespace(collective.strip())
    last_name = author.findtext("LastName", default="").strip()
    fore_name = author.findtext("ForeName", default="").strip()
    full_name = normalize_whitespace(f"{fore_name} {last_name}".strip())
    return full_name or None


def build_corpus(terms_path: Path, output_path: Path, retmax: int = 20) -> dict[str, Any]:
    client = PubMedClient()
    terms = load_terms(terms_path)
    articles_by_pmid: dict[str, dict[str, Any]] = {}
    matched_terms = defaultdict(list)
    errors: list[dict[str, str]] = []
    duplicates_removed = 0
    total_fetched = 0

    for term in terms:
        try:
            pmids = client.search(term=term, retmax=retmax)
            if not pmids:
                errors.append({"term": term, "error": "No PubMed results"})
                continue
            items = parse_pubmed_xml(client.fetch(pmids))
            selected = items[:5]
            if len(selected) < 5:
                errors.append({"term": term, "error": f"Only {len(selected)} articles with abstracts"})
            for article in selected:
                total_fetched += 1
                pmid = article["pmid"]
                if pmid in articles_by_pmid:
                    duplicates_removed += 1
                else:
                    articles_by_pmid[pmid] = article
                matched_terms[pmid].append(term)
        except Exception as exc:
            errors.append({"term": term, "error": str(exc)})

    records = []
    for pmid, article in articles_by_pmid.items():
        record = {
            **article,
            "authors": [article["first_author"]] if article["first_author"] else [],
            "matched_terms": sorted(set(matched_terms[pmid])),
            "text": normalize_whitespace(f'{article["title"]} {article["abstract"]}'),
        }
        records.append(record)

    records.sort(key=lambda item: ((item["year"] or 0), item["pmid"]), reverse=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_terms": terms,
        "stats": {
            "terms_processed": len(terms),
            "articles_fetched": total_fetched,
            "unique_articles": len(records),
            "duplicates_removed": duplicates_removed,
            "errors": len(errors),
        },
        "errors": errors,
        "articles": records,
    }
    write_json(output_path, payload)
    return payload
