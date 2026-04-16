from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_TERMS_PATH = INPUT_DIR / "medical_terms.csv"
DEFAULT_CORPUS_PATH = DATA_DIR / "articles.json"
DEFAULT_EVALUATION_PATH = RESULTS_DIR / "evaluation.json"
DEFAULT_BM25_ANALYSIS_PATH = RESULTS_DIR / "bm25_analysis.json"
DEFAULT_RAG_PATH = RESULTS_DIR / "rag_demo.json"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENV_PATH = ROOT_DIR / ".env"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ENV_PATH)

for path in [DATA_DIR, INPUT_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
