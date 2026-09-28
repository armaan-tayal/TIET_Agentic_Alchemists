# Smart Guided Troubleshooting Engine

Turns vague Galaxy device complaints ("screen flickers and battery dies fast") into **validated, ordered troubleshooting plans** with one-tap Settings deeplinks — served over a REST API.

Built for the **Samsung PRISM Hackathon**. Powered by **Muse** (Anthropic) with a **deterministic-first** design: the LLM only does language understanding, paraphrasing, and step extraction — code does all grounding, deeplink matching, ordering, formatting, and validation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![LLM: Muse](https://img.shields.io/badge/LLM-Muse%20(Anthropic)-orange.svg)](https://www.anthropic.com/)
[![API: FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## TL;DR

| Metric | Result |
|---|---|
| Schema / rule compliance | 0 errors, 0 violations, 0 URL leaks (20/20 responses) |
| Deeplink accuracy (hand-labelled gold set) | **100%** (16/16) |
| Cache hit rate on unseen paraphrases | 98% (0% false hits on negatives) |
| Cache-hit latency P95 | **183 ms** (budget ≤ 300 ms) |
| New-query latency P95 (rules mode) | 3468 ms (budget ≤ 8000 ms) |

Full report: [`eval/metrics.md`](eval/metrics.md) · ablations: [`eval/ablations.md`](eval/ablations.md)

---

## How it works

```
query ─► local embed ─► FAISS paraphrase index ─► slot check ─► HIT → return cached plan (≤300 ms)
                                                        │
                                                       MISS
                                                        ▼
[0] Enrich (Muse): canonical query + slots {domain, symptom, trigger} + 8–10 paraphrases
[1] Extract (Muse, structured output): steps WITH source spans from SIIS text
    → grounding validator drops unsupported steps
[2] Path parser → group steps into actions (One Action = One Screen)
    → hybrid retrieval (BM25 + dense, RRF, rerank, leaf-screen preference) → deeplinks
    → rule-based category + stable sort (auto → manual → critical)
    → field validators / repair → URL scrub → Pydantic validate
[3] Write plan + all paraphrase vectors to cache
[4] Return JSON + meta {latency_ms, cache_hit, model, cost_usd}
```

**Design principle: deterministic-first.** When in doubt, logic lives in Python, not in the prompt.

### Where Muse is used (and where it isn't)

All LLM calls go through a single wrapper, [`app/llm.py`](app/llm.py), which uses Anthropic's structured-output API (`messages.parse` against Pydantic models), with retries, timeouts, and per-call token/cost logging. No other provider is wired in — there is no OpenAI/Gemini/local-LLM path.

| Stage | Muse role | File |
|---|---|---|
| Enrich | Canonical query, slots, 8–10 paraphrases | `app/pipeline/enrich.py` |
| Extract | Candidate steps with source spans from SIIS text | `app/pipeline/extract.py` |
| Repair | Fix malformed action descriptions (1 retry) | `app/engine.py` |

Everything else is code: grounding validation, screen-path parsing, hybrid deeplink retrieval, category rules, ordering, word-count/casing/URL scrubbing, cache (FAISS + SQLite), latency budgets.

Without an `ANTHROPIC_API_KEY`, `PIPELINE_MODE=auto` falls back to a deterministic rules pipeline — every result above was produced in this mode, so the repo is fully reproducible without a key.

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY (optional; without it, PIPELINE_MODE=auto uses rules)

python scripts/build_index.py # index data/deeplinks.json (downloads local models on first run)
python scripts/prewarm.py --reset   # run all provided queries -> cache + results.jsonl
uvicorn app.api:app --port 8000     # run the API
```

Then:

```bash
curl -X POST localhost:8000/v1/troubleshoot \
  -H 'Content-Type: application/json' \
  -d '{"query": "My screen keeps flickering and the battery dies fast",
       "siis_response": {"title": "...", "content": "..."}}'
```

### Docker

```bash
docker build -t troubleshoot . && docker run -p 8000:8000 troubleshoot
```

(The image bakes in the models, deeplink index, and a prewarmed cache.)

## API

| Endpoint | Method | Description |
|---|---|---|
| `/v1/troubleshoot` | POST | `{query, siis_response?}` → validated plan + `meta {latency_ms, cache_hit, model, cost_usd}` |
| `/health` | GET | 200 when models, catalog, and cache are loaded; includes `llm` availability |

The response contract is fixed in [`app/schema.py`](app/schema.py) — field names, types, and enums must not change.

### Output guarantees (enforced in code, graded automatically)

1. **Zero URL leaks** — regex scrub on every string field.
2. **Catalog integrity** — every deeplink exists verbatim in `data/deeplinks.json`, is `voiceassist://dummy_positive`, or is `null`. Never generated or guessed.
3. **No hallucinated steps** — every step traces to a span in the SIIS text; otherwise a `no_match` / `no_siis_context` fallback.
4. **Pure JSON** — no markdown fences, no preamble.

---

## Evaluation

```bash
pytest -q                          # unit + end-to-end tests (fake Claude client, no key needed)
python eval/run_eval.py             # compliance + deeplink accuracy + hit rate + latency -> eval/metrics.md
python eval/run_eval.py --tune      # re-tune thresholds on the even split -> thresholds.json
python eval/run_eval.py --latency   # latency gates only
python eval/ablations.py            # full-LLM vs hybrid vs rules vs no-slot-guard -> eval/ablations.md
```

### Ablation highlights

- Deeplink retrieval: BM25-only 25% → dense-only 69% → **full hybrid + rerank + leaf preference + threshold 100%**
- Multi-vector paraphrase cache: single-vector 48% hit rate → **full 98%**; removing the cross-encoder collapses it to 5%
- Rules-only pipeline: 20/20 fully compliant at $0.00/query — Muse (hybrid) adds language robustness on top

---

## Project structure

```
app/
  api.py            # FastAPI: POST /v1/troubleshoot, GET /health
  schema.py         # Provided Pydantic contract — DO NOT MODIFY
  llm.py            # Single Muse client wrapper (structured output, token/cost logging, retries)
  engine.py         # Pipeline orchestrator
  embed.py          # Local models (bi-encoder + 2 cross-encoders), loaded once
  pipeline/         # enrich, extract, grounding, path_parser, deeplinks, ordering, validators
  cache/            # SQLite plans + FAISS paraphrase index, slot guard
data/               # siis_responses.json, deeplinks.json, input.txt, samples/, index/ (built)
scripts/            # build_index.py, prewarm.py
eval/               # run_eval.py, ablations.py, paraphrase_gen.py, metrics.md, ablations.md
tests/              # pytest suite (fake Claude client)
results.jsonl       # one API response per line (from prewarm)
```

## Configuration

Thresholds live in `thresholds.json` (written by `eval/run_eval.py --tune`); env vars override both. Key settings in [`.env.example`](.env.example): `ANTHROPIC_API_KEY`, `LLM_MODEL` (default `claude-haiku-4-5`), `PIPELINE_MODE` (`auto` / `rules` / `llm`).

## License

MIT — see [LICENSE](LICENSE).
