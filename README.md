# Retail Medallion Pipeline

An agentic Bronze -> Silver -> Gold data pipeline with a Streamlit UI and
human-in-the-loop (HITL) approvals at every source-to-target mapping (STTM)
stage. Orchestration is powered by LangGraph with interruption and resume, so a
run pauses for reviewer approval and continues exactly where it left off.

## What the app does

1. Profiles an uploaded orders CSV and flags quality issues (currency
   formatting, mixed date formats, missing values).
2. Generates an STTM for each layer and pauses for human approval.
3. Ingests to Bronze (typed, cleansed), refines to Silver (derived fields,
   deduplication, date normalization), and aggregates to Gold (dynamic
   analytics tables inferred from the schema, with no hardcoded business
   columns).
4. Generates an on-demand HTML report with embedded Plotly charts.
5. Records append-only audit logs and per-agent observability traces.

## Architecture

```
Upload CSV
    |
    v
[Phase 1] Profiler ---> Bronze STTM ---> (HITL: approve/reject) --+
    |                                                             |
    v  approve                                        reject (loop back)
[Phase 2] Bronze Agent ---> Silver STTM ---> (HITL: approve/reject) --+
    |                                                                 |
    v  approve                                            reject (loop back)
[Phase 3] Silver Agent ---> Gold STTM ---> (HITL: approve/reject) --+
    |                                                               |
    v  approve                                          reject (loop back)
[Phase 4] Gold Agent ---> Gold tables
    |
    v
On-demand Reporter ---> HTML report (Plotly)
```

Cross-cutting core modules: `config`, `state`, `audit`, `observability`,
`memory` (ChromaDB with in-memory fallback).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env and set one provider credential
```

## Environment variables

| Variable | Purpose |
| --- | --- |
| `CLAUDE_API_KEY` / `ANTHROPIC_API_KEY` | Selects the Claude provider |
| `GITHUB_TOKEN` | Selects the GitHub-hosted OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | Selects the standard OpenAI provider |
| `LLM_BASE_URL` | Optional override for the OpenAI-compatible base URL |
| `LLM_MODEL` | Model ID (default `claude-sonnet-5`) |
| `LLM_TEMPERATURE` | Sampling temperature (ignored for `claude-sonnet-5`) |
| `CHROMA_PERSIST_DIR` | ChromaDB persistence directory |
| `LOG_LEVEL` | Logging level |
| `STREAMLIT_SERVER_FILE_WATCHER_TYPE` | Set to `none` for stability |

Provider priority: Claude -> GitHub OpenAI-compatible -> OpenAI.

## Run

```powershell
python -m streamlit run streamlit_app.py --server.port 8501 --server.fileWatcherType none
```

Open http://localhost:8501.

## Walkthrough by phase

1. Upload `sample_orders.csv`, optionally click **Analyze File** for three
   suggested business intents, then **Run Pipeline**.
2. **Bronze STTM review** appears: inspect the mapping, then approve or reject
   with feedback (rejection loops back to Phase 1 with cumulative feedback).
3. **Silver STTM review**: approve to run the Silver agent.
4. **Gold STTM review**: approve to run the Gold agent and produce tables.
5. Optionally revise the Gold analysis, then click **Generate Report** to view
   and download the HTML report.

## Output artifacts

- `data/profiles/profile_*.json` - profile output
- `data/sttm/{bronze,silver,gold}/sttm_*.csv` - STTMs
- `data/bronze/*.parquet`, `data/silver/*.parquet`, `data/gold/*.parquet`
- `data/reports/report_*.html` - HTML report
- `data/traces/audit_*.jsonl`, `data/traces/trace_*.json` - audit and traces

## Troubleshooting

- **Model not found (404):** the configured `LLM_MODEL` is not accessible for
  your account. Switch `LLM_MODEL` to an ID you can access.
- **`claude-sonnet-5` temperature error:** this model rejects an explicit
  temperature; the code already omits it for that model.
- **Chroma unavailable:** vector memory silently falls back to an in-memory
  store; the pipeline still runs.
- **Analyze File fails:** confirm a provider credential is set in `.env`.

## Tests

```powershell
pytest -q
```

Pure transformation tests are hermetic (no network) and isolate the filesystem
via `tmp_path`. LLM-dependent tests skip cleanly unless `RUN_LLM_TESTS` and a
provider credential are set.

## Expected sample outcomes

Running `sample_orders.csv` through all phases produces Bronze/Silver/Gold
parquet files, at least one `revenue_by_*` Gold table, a monthly trend, a top
customers table, and an HTML report summarizing total revenue by dimension.

## Provider compatibility notes

- Valid model IDs vary by account and provider.
- A `model not found` 404 means you must switch `LLM_MODEL` to an accessible ID.
- For `claude-sonnet-5`, do not pass a temperature parameter.
