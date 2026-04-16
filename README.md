# Medical RAG System

Python-based take-home implementation for a medical RAG workflow over PubMed abstracts.

## Setup & Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build the corpus from PubMed:

```bash
python main.py fetch
```

Run retrieval evaluation:

```bash
python main.py evaluate
```

Run both steps together:

```bash
python main.py run-all
```

Run the RAG demo after setting a Groq API key:

```bash
export GROQ_API_KEY=your_key
export GROQ_MODEL=llama-3.3-70b-versatile
python main.py rag
```

Or place the same values in a local `.env` file:

```bash
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Outputs:

- `data/articles.json`: structured PubMed corpus
- `results/evaluation.json`: retrieval rankings and metrics
- `results/bm25_analysis.json`: BM25 parameter comparison
- `results/rag_demo.json`: RAG demo output when `GROQ_API_KEY` is set

## Project Structure

- `main.py`: small entry point that forwards all commands to the CLI module
- `medical_rag/pipeline.py`: PubMed data collection, XML parsing, deduplication, and corpus export
- `medical_rag/retrieval.py`: BM25, semantic retrieval, and hybrid RRF fusion
- `medical_rag/evaluation.py`: benchmark queries, retrieval metrics, and best-model selection
- `medical_rag/rag.py`: grounded answer generation with Groq using retrieved PubMed context

## Sample Run

```bash
python main.py fetch
terms processed: 10
unique articles: 50
duplicates removed: 0
errors: 0

python main.py evaluate
evaluation summary:
bm25: recall@5=0.6400, mrr@5=0.8500, ndcg@5=0.6797
semantic: recall@5=0.7600, mrr@5=1.0000, ndcg@5=0.8124
hybrid: recall@5=0.8000, mrr@5=0.9000, ndcg@5=0.8213
best model is: hybrid

python main.py rag
/path/to/project/results/rag_demo.json
```

## Example RAG Output

Example query:

```text
What are the latest guidelines for managing type 2 diabetes?
```

Example model behavior from `results/rag_demo.json`:

```text
The provided context is insufficient to determine the latest guidelines for managing type 2 diabetes.
```

Example retrieved documents:

- `Kidney disease and heart failure: recent advances and current challenges: conclusions from a Kidney Disease: Improving Global Outcomes (KDIGO) Controversies Conference.` (PMID: `41791738`)
- `Breaking the cycle: patient activation role in improving diabetes self-care adherence for alleviating diabetes distress.` (PMID: `41848224`)

Why this example is useful:

- it shows the system is grounded in retrieved context rather than hallucinating
- it refuses to invent guidelines when the retrieved papers do not actually contain them
- that behavior is safer for a medical QA workflow

## Approach

### Part 1: Data Pipeline

- Read `data/input/medical_terms.csv`
- Query PubMed with `esearch`
- Fetch article metadata with `efetch`
- Extract `PMID`, `title`, `abstract`, `first_author`, `journal`, `year`, `doi`
- Store matched source terms per article
- Save a JSON corpus for retrieval

Observed run summary on this machine:

- Terms processed: `10`
- Articles fetched: `50`
- Unique articles: `50`
- Duplicates removed: `0`
- Errors: `0`

The appendix notes that overlap is expected. In this run, the current top-5-most-recent sets for the 10 terms happened not to overlap, so deduplication logic executed but did not remove any records.

### Part 2: Retrieval

Implemented three retrievers over `title + abstract`:

1. `BM25` with `rank-bm25`
2. `Semantic` search with `sentence-transformers/all-MiniLM-L6-v2`
3. `Hybrid` fusion with a scratch implementation of Reciprocal Rank Fusion

Dense model choice:

- I chose `sentence-transformers/all-MiniLM-L6-v2`
- Reason: small, fast, widely used baseline, practical for a take-home and CPU-friendly
- Trade-off: it is English-centric, so I added a small Turkish medical query-expansion layer as a bonus improvement because the corpus is English but some evaluation queries are Turkish
- If I had more time, I would compare it against `intfloat/multilingual-e5-small` and optionally add a cross-encoder reranker

### Bonus Improvement

I added lightweight Turkish-to-English medical query expansion for terms such as `çocuklarda`, `tedavisi`, `çölyak`, `tanı`, and `kriterleri`. This keeps the dense model lightweight while improving retrieval for Turkish doctor queries over English literature.

Result:

- It helped the system handle Turkish evaluation queries over an English-only PubMed corpus without switching to a much heavier multilingual embedding model.
- In the final evaluation, the hybrid retriever still performed strongly on the Turkish queries, with `nDCG@5=0.8539` for the otitis media query and `nDCG@5=0.8688` for the celiac diagnosis query.
- I did not run a separate ablation without query expansion, so I present this as a practical improvement rather than a fully isolated experiment.

### Part 3: RAG

- Best retriever is selected from evaluation results
- RAG layer retrieves top documents, builds a grounded context, and calls the Groq chat completions API
- The system prompt forces context-only answers and source citation by PMID or title

Provider choice:

- I chose `Groq`
- Reason: simple OpenAI-compatible API, low friction setup, fast inference, easy env-var based auth

Note:

- The app reads `GROQ_API_KEY` and `GROQ_MODEL` from either exported environment variables or the local `.env` file

## BM25 Analysis

`k1` controls term-frequency saturation:

- Lower `k1` makes repeated term matches saturate quickly
- Higher `k1` keeps rewarding repeated occurrences longer

`b` controls document-length normalization:

- `b=0` means no length normalization
- `b=1` means full normalization
- Higher `b` penalizes longer documents more strongly

Observed behavior on the diabetes query:

- `k1=0.6, b=0.25`: more conservative scoring, weaker separation among top documents
- `k1=1.5, b=0.75`: stronger separation and better balance
- `k1=2.2, b=0.95`: repeated-term reward and length normalization become stronger, but the ranking still overweights lexical overlap and keeps some irrelevant documents near the top

This is why BM25 alone underperformed dense and hybrid retrieval on the multilingual/semantic-style queries.

## RRF Analysis

What `k` does:

- RRF score is `sum(1 / (k + rank))`
- `k` smooths how aggressively top ranks dominate the final fused score
- Larger `k` compresses the difference between rank positions

Behavior at extremes:

- `k=0`: rank 1 gets a very large advantage, so fusion is dominated by whichever retriever places a document near the top
- `k=60`: standard compromise, still rewards high ranks but keeps fusion stable
- `k=1000`: rank differences contribute only tiny changes, so the fusion becomes much flatter and less responsive to top-ranked agreement

Why use rank positions instead of raw scores:

- BM25 and cosine similarity are on different scales
- Their score distributions are not calibrated
- A BM25 score of `8` and a cosine score of `0.55` are not directly comparable
- Rank-based fusion avoids scale mismatch and only uses ordering information, which is much more stable across heterogeneous retrievers

## Evaluation

Metric choice:

- Primary metric: `nDCG@5`
- Supporting metrics: `Recall@5` and `MRR@5`

Why:

- The task is top-k retrieval for downstream RAG, so quality near the top matters most
- `nDCG@5` rewards putting relevant documents earlier in the ranking
- `Recall@5` shows coverage
- `MRR@5` highlights whether the first highly relevant document appears early

Weak supervision setup:

- Each evaluation query is mapped to its intended source medical term
- Documents whose `matched_terms` contain that term are treated as relevant

Average results:

| Method | Recall@5 | MRR@5 | nDCG@5 |
| --- | ---: | ---: | ---: |
| BM25 | 0.64 | 0.85 | 0.6797 |
| Semantic | 0.76 | 1.00 | 0.8124 |
| Hybrid | 0.80 | 0.90 | 0.8213 |

Conclusion:

- `Hybrid` performed best overall because it had the highest `nDCG@5` and `Recall@5`
- `Semantic` had the best `MRR@5`
- BM25 alone struggled most on semantically phrased and Turkish queries

## Limitations

- Evaluation uses weak supervision: a document is treated as relevant when its `matched_terms` field matches the intended query topic.
- This is useful for a take-home benchmark, but it is not the same as having manually labeled relevance judgments.
- Because of that, the reported retrieval metrics are informative but should not be treated as a final clinical-quality evaluation.

## Hardest Problem

The hardest problem was balancing multilingual query handling with a lightweight local setup. The corpus is English, but two benchmark queries are Turkish. A larger multilingual embedding model would likely be stronger, but it also increases setup cost and latency. I resolved this by pairing a compact dense model with targeted Turkish query expansion, then validating the trade-off with retrieval metrics rather than guessing.

## What I Would Change With More Time

- Compare against `intfloat/multilingual-e5-small`
- Add a cross-encoder reranker
- Build manually reviewed relevance judgments instead of weak supervision
- Expand PubMed retrieval depth beyond 5 recent abstracts per term and then rerank
- Add automated tests for XML parsing and ranking

## Scenario Question

If L40S is unavailable and results are needed by end of week, I would not block on the original provider.

Immediate plan:

1. Freeze the benchmark spec first: model name, quantization level, prompt set, output format, latency and quality metrics, and acceptance criteria.
2. Send my manager an async update with the fallback plan and proceed unless they object.
3. Spin up the benchmark on an alternative platform the same day.

Fallback platforms I would check in parallel:

- `RunPod`: fast access to `A100` or `H100`, good for short-lived benchmark runs
- `Lambda Labs`: straightforward GPU rental, often good `A100 80GB` availability
- `Vast.ai`: cheapest option, but more operational variance
- `Together AI` or `Fireworks`: hosted inference if raw GPU access is not necessary
- `Hugging Face Inference Endpoints`: clean deployment path if we want a reproducible managed endpoint

Hardware fallback logic:

- If the 70B model fits and speed matters, use `H100 80GB`
- If `H100` is expensive or unavailable, use `2x A100 80GB` or tensor-parallel `A100`
- If only smaller cards are available, use an `AWQ` or `GPTQ` quantized checkpoint and clearly document the quantization trade-off

Trade-offs:

- Managed inference is fastest to start but gives less control over kernels and batching
- Raw GPU rental gives more control and cheaper repeated runs, but setup time is higher
- Quantization improves feasibility and cost, but may shift medical QA quality, so I would benchmark both full precision and quantized if time allows

Execution details:

- Start with a 20-question smoke test to validate prompt formatting, throughput, and output logging
- Then run the full benchmark overnight
- Save prompts, raw outputs, latency, token counts, GPU type, VRAM, model revision, and decoding settings so results are reproducible
- Deliver the end-of-week report with both quality and cost/latency trade-offs, not just a single score

## AI Usage

I used AI assistance selectively for a few harder parts of the take-home:
- understanding and validating the Reciprocal Rank Fusion (RRF) implementation details
- refining parts of the README, especially the written explanations of retrieval trade-offs and evaluation
- debugging a few integration issues during development

The overall project structure, PubMed pipeline, retrieval flow, evaluation setup, environment configuration, and final integration were implemented and reviewed manually.
