# Samsung PRISM Hackathon — Submission

## Project
**Smart Guided Troubleshooting Engine** — turns vague Galaxy device complaints into validated, ordered troubleshooting plans with one-tap Settings deeplinks, served over a REST API.

## Problem
Users describe device problems in vague, emotional language ("screen flickers and battery dies fast"). Support content (SIIS articles) is long and generic. The gap between complaint → correct Settings screen costs users time and support teams money.

## Solution
A deterministic-first pipeline powered by **Google Gemini** (via its OpenAI-compatible endpoint):

- **Muse** handles only what needs language: query understanding, paraphrase generation, and step extraction with source spans (structured output via `messages.parse`).
- **Code** handles everything verifiable: grounding each step against its source span, grouping steps into one-screen actions, hybrid (BM25 + dense + cross-encoder rerank) deeplink retrieval with leaf-screen preference, rule-based categorization (`auto → manual → critical`), field validation, and URL scrubbing.
- A **multi-vector paraphrase cache** (FAISS + SQLite, slot-guarded) serves repeat complaints in ~40 ms P95 with zero LLM calls.

## Key results

| Metric | Result |
|---|---|
| Schema / rule compliance | 0 errors, 0 violations, 0 URL leaks (20/20 responses) |
| Deeplink accuracy (16 hand-labelled cases) | **100%** |
| Cache hit rate on unseen paraphrases | 98% (0% false hits on negatives) |
| Cache-hit latency P95 | 39.5 ms (budget 300 ms) |
| New-query latency P95 | 595.5 ms (budget 8000 ms) |

Ablations show each component earns its place: dense-only retrieval 69% → full hybrid 100%; single-vector cache 48% hit rate → multi-vector 98%; removing the cross-encoder collapses hit rate to 5%.

## Why this design wins
1. **No hallucinations by construction** — every step must cite a verbatim source span or the plan is rejected.
2. **Cheap at scale** — the fast path makes zero LLM calls; rules mode runs at $0.00/query.
3. **Reproducible** — full eval suite (`pytest`, `eval/run_eval.py`, `eval/ablations.py`) runs without an API key.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY for full Gemini mode (optional)
python scripts/build_index.py
python scripts/prewarm.py --reset
uvicorn app.api:app --port 8000
# POST /v1/troubleshoot  {"query": "...", "siis_response": {"title": "...", "content": "..."}}
```

## Links
- Repo: <add GitHub URL>
- Demo video: <add link>
- Full eval report: `eval/metrics.md`
