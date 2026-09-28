# CLAUDE.md — Smart Guided Troubleshooting Engine (Samsung PRISM)

## What this project is
A service that turns vague Galaxy device complaints ("screen flickers and battery dies fast") into validated, ordered troubleshooting plans with one-tap Settings deeplinks (`voiceassist://...`, masked in the provided catalog). It is served over a REST API. Cached answers must return in ≤300 ms (P95) and new queries in ≤8 s (P95).

**Design principle: deterministic-first.** The LLM does only three jobs: understand language, paraphrase queries, and extract candidate steps from source text. Code does all grounding, grouping, deeplink matching, ordering, formatting, and validation. When in doubt, move logic out of the prompt and into Python.

## Pipeline
```
query ─► local embed ─► FAISS paraphrase index ─► slot check ─► HIT → return cached plan
                                                        │
                                                       MISS
                                                        ▼
[0] Enrich (LLM): canonical query + slots {domain, symptom, trigger} + 8–10 paraphrases
[1] Extract (LLM, structured output): steps WITH source spans from SIIS text
    → grounding validator drops unsupported steps
[2] Path parser → group steps into actions (One Action = One Screen)
    → hybrid retrieval (BM25 + dense, RRF, rerank, leaf-screen preference) → deeplinks
    → rule-based category + stable sort (auto → manual → critical)
    → field validators / repair → URL scrub → Pydantic validate
[3] Write plan + all paraphrase vectors to cache
[4] Return JSON + meta {latency_ms, cache_hit, model, cost_usd}
```

## Directory layout (target)
```
app/
  api.py            # FastAPI: POST /v1/troubleshoot, GET /health
  schema.py         # Provided Pydantic contract — DO NOT MODIFY
  pipeline/
    enrich.py       # canonical query, slots, paraphrases
    extract.py      # LLM step extraction with source spans
    grounding.py    # verify each step against its source span
    path_parser.py  # steps → screen paths → actions
    deeplinks.py    # hybrid retrieval, leaf preference, thresholds
    ordering.py     # category rules + disruption sort
    validators.py   # word counts, casing, templates, URL scrub, repair
  cache/
    store.py        # SQLite plans + FAISS paraphrase index
    slots.py        # slot extraction/guard for cache hits
  llm.py            # single LLM client wrapper; logs tokens + cost
data/               # siis_responses.json (queries + SIIS), input.txt, deeplinks.json, samples/, index/ (built)
scripts/
  prewarm.py        # run all queries.json offline → fill cache
  build_index.py    # index deeplinks.json (BM25 + embeddings)
eval/
  run_eval.py       # schema/rule compliance, accuracy, latency, hit rate
  paraphrase_gen.py # unseen paraphrases for hit-rate testing
  ablations.py      # full-LLM vs hybrid vs rules vs no-slot-guard
  metrics.md        # filled-in evaluation report (Appendix C template)
tests/
results.jsonl       # one API response per line
```

## Non-negotiable rules
These are graded automatically. Any code change must keep every one of them true.

1. **Zero URL leaks.** No `http`, `https`, `www.`, or markdown links in any output string. Run the regex scrub on every string field before returning, even if the prompt already told the LLM not to produce them.
2. **Catalog integrity.** Every `actionableDeeplink.deeplink` must exist verbatim in `deeplinks.json`, or be exactly `voiceassist://dummy_positive`, or be `null`. Never generate, edit, or guess URIs. Match on `description`, `message`, and `qna_description`, never on the masked URI string.
3. **No hallucinated steps.** Every step must trace back to a span in the SIIS text. If no valid steps remain, return `{"contexts": []}` with fallback `"no_match"`. If there is no SIIS text and no cache hit, use fallback `"no_siis_context"`.
4. **Pure JSON.** Responses contain no markdown fences and no preamble.
5. **`schema.py` is the contract.** Don't change field names, types, or enums.

## Field rules (enforce in `validators.py`, not in prompts)
| Field | Rule | Produced by |
|---|---|---|
| `goal` | `Follow these steps to perform this <Topic> Troubleshooting` (or `Configuration`) | Code template |
| `title` | 2–3 words, sentence case | LLM draft → code enforces |
| `score` | Float from 0.0 to 1.0 | Code: retrieval similarity × validator pass rate |
| `actionName` | Title Case; exactly one screen/feature | Path parser + LLM label |
| `description` | Starts with "It will", 5–7 words total | LLM draft → validate → 1 repair retry → template fallback |
| `steps` | Imperative, one physical interaction each, no URLs | Extraction + scrub |
| `category` | `auto` (reachable via deeplink) / `manual` (physical, no actionable deeplink) / `critical` (reset, restart, safe mode, firmware update — always last) | Keyword rules, LLM as tie-breaker |
| `query_variations` | 8–10 distinct paraphrases across registers (formal, casual, keyword-only, frustrated, with typos) | Enrich step |

- `manual` actions must have `actionableDeeplink: null`.
- Sort actions with a stable sort by disruption level: `auto` → `manual` → `critical`.

## Cache rules
- The fast path makes **no LLM calls**. It embeds the raw query locally and searches the paraphrase index.
- Each cached plan is stored under all its paraphrase vectors (multi-vector cache).
- A hit requires both similarity ≥ `HIT_THRESHOLD` and matching slots (domain + symptom). A borderline score triggers a cross-encoder rerank. Otherwise, treat it as a miss.
- Only plans that fully pass validation get written to the cache.
- Keep thresholds in config, not hardcoded, and tune them with `eval/`.

## Deeplink mapping rules
- Query retrieval with the full parsed screen path (e.g. "Display Navigation bar gesture"), not a single step.
- Prefer the most specific (leaf) screen, and penalize candidates that are parents of a better match.
- Below the confidence threshold: use `voiceassist://dummy_positive` only if the step clearly opens a real Settings screen; otherwise use `null`.

## Coding conventions
- Python 3.11+, type hints everywhere, Pydantic v2.
- All LLM calls go through `app/llm.py`, which uses structured/JSON output, logs tokens and cost, and has retries plus a timeout.
- Load models, indexes, and the DB once at startup. `/health` returns 200 only when all of them are ready.
- Pipeline functions should be pure where possible so they're easy to unit-test.
- Don't add prompt text to enforce a rule that a validator can enforce.
- Keep secrets in `.env`. Never commit API keys.

## Commands
```bash
pip install -r requirements.txt
cp .env.example .env                   # add GEMINI_API_KEY (free at https://aistudio.google.com/apikey); without it PIPELINE_MODE=auto falls back to rules
python scripts/build_index.py          # index deeplinks.json (downloads local models on first run)
python scripts/prewarm.py --reset      # run data/siis_responses.json through the pipeline -> cache + results.jsonl
uvicorn app.api:app --port 8000        # run API
pytest -q                              # unit + end-to-end tests (fake Gemini client, no key needed)
python eval/run_eval.py                # compliance + deeplink accuracy + hit rate + latency -> eval/metrics.md
python eval/run_eval.py --tune         # re-tune thresholds on the even split -> thresholds.json
python eval/ablations.py               # ablation tables -> eval/ablations.md
python eval/paraphrase_gen.py          # optional: extra LLM-written unseen paraphrases (needs key)
docker build -t troubleshoot . && docker run -p 8000:8000 troubleshoot
```
On Windows, set `PYTHONIOENCODING=utf-8` when printing SIIS text to the console.

## Where things live
- `app/engine.py` orchestrates everything; `app/pipeline/compliance.py` is the single rule checker (cache-write gate + eval).
- Tuned thresholds: `thresholds.json` (env vars override). Deeplink gold set: `eval/gold_deeplinks.json`. Unseen paraphrases: `eval/paraphrases.json`.

## Before finishing any change
- [ ] `pytest` passes
- [ ] All 5 `samples/` still validate against `schema.py` and match the expected structure
- [ ] No URL matches in `results.jsonl` (`grep -E "https?://|www\." results.jsonl` returns nothing)
- [ ] Every deeplink is in the catalog, `dummy_positive`, or null
- [ ] Cache-hit P95 is still ≤300 ms (`eval/run_eval.py --latency`)

## Pitfalls to remember
- Keying the cache on exact strings: paraphrases will miss. Always key on embeddings.
- Asking the LLM to count words: it's unreliable. Validate and repair in code.
- LLMs inject "visit samsung.com/support" from pretraining. Scrub every string.
- Matching the parent menu ("Display") when the leaf ("Navigation bar") exists costs points on deeplink relevance.
- Putting one action per tap fragments the plan, and bundling several screens into one action breaks deeplinks. Stick to one screen per action.
