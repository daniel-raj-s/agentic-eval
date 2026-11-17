# 🤖 AI Agent Evaluation Pipeline & CI/CD Quality Gate

An end-to-end evaluation harness for a tool-calling customer-support agent. It
combines **DeepEval** semantic metrics, **Promptfoo** deterministic/security
assertions and an explicit **quality gate** so a regression blocks a merge
instead of disappearing into a log file.

The default agent mode is deterministic: free, reproducible and safe for CI.
A live OpenAI-compatible tool-calling loop is available as an opt-in mode for
evaluating current Gemini, Groq or OpenAI models.

---

## What gets evaluated

```
                    +--------------------------------------+
                    | SupportAgent (RAG + tool calling)    |
                    | deterministic OR live LLM tool loop  |
                    +------------------+-------------------+
                                       |
             +-------------------------+-------------------------+
             |                                                   |
             v                                                   v
  +----------------------------+                    +-----------------------------+
  | DeepEval (pytest, LLM)     |                    | Promptfoo (Node, free)      |
  |                            |                    |                             |
  | Faithfulness               |                    | Golden-case assertions      |
  | Answer relevancy           |                    | Tool selection + arguments  |
  | Hallucination              |                    | Latency / non-empty output  |
  | Tool selection accuracy    |                    | Prompt-injection / PII      |
  | Tool-reasoning G-Eval      |                    | Guardrail + red-team scan   |
  +-------------+--------------+                    +--------------+--------------+
                |                                                  |
                +------------------------+-------------------------+
                                         v
                        +-----------------------------------+
                        | scripts/quality_gate.py           |
                        | thresholds + missing-artifact fail|
                        | Markdown + JSON + PR comment      |
                        +-----------------------------------+
```

| Engine | Role |
| :--- | :--- |
| **DeepEval** | Semantic quality: is the answer grounded, relevant, non-hallucinated, and did the agent use tools correctly? |
| **Promptfoo** | Fast regression suite: exact facts, tool arguments, latency, guardrails, injection/PII cases, optional adversarial red-team scan. |
| **Quality gate** | Turns artifacts into a merge decision. Missing required metrics fail closed. |

---

## Quickstart

### 1. Requirements

- Python **3.13+**
- Node.js **20.18+** (Node 24 LTS recommended)
- At least one provider key for LLM-judged metrics (Gemini free tier recommended)

### 2. Install

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install
Copy-Item .env.example .env
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY`, `GROQ_API_KEY`, or `OPENAI_API_KEY`.
Never commit `.env`; it is already in `.gitignore`.

### 3. Run everything

```powershell
# Windows
.\scripts\run_all_evals.ps1                # full run (LLM suite runs if a key exists)
.\scripts\run_all_evals.ps1 -SkipLlm       # free deterministic + Promptfoo + gate
.\scripts\run_all_evals.ps1 -RunRedteam    # add the adversarial scan
```

```bash
# macOS / Linux
chmod +x scripts/run_all_evals.sh
./scripts/run_all_evals.sh
./scripts/run_all_evals.sh --skip-llm
./scripts/run_all_evals.sh --redteam
```

Artifacts are written to `results/`:

- `deepeval_metrics*.json` — one record per metric measurement
- `promptfoo-results.json` — assertion/security suite
- `promptfoo-redteam-results.json` — optional red-team scan
- `quality_gate_report.md` — human-readable report
- `quality_gate_summary.json` — machine-readable decision

### 4. Step-by-step

```bash
# Deterministic tests: no network, no API quota
pytest tests --ignore=tests/deepeval -m "not llm" -q

# LLM-judged metrics (requires a key). EVAL_KEEP_METRICS preserves the
# deterministic ToolSelectionAccuracy records from the previous step.
# PowerShell: $env:EVAL_KEEP_METRICS="1"; pytest tests/deepeval -m llm -q
EVAL_KEEP_METRICS=1 pytest tests/deepeval -m llm -q

# Promptfoo suite (writes results/promptfoo-results.json)
npm run eval

# Optional adversarial scan (requires a judge key)
npm run redteam

# Enforce policy
python scripts/quality_gate.py
```

Open the Promptfoo viewer with `npm run view`.

---

## Agent modes

| Mode | How to enable | Network | Used for |
| :--- | :--- | :---: | :--- |
| `deterministic` | default / `AGENT_MODE=deterministic` | No | Reproducible CI, free regression tests, prompt/security gates |
| `llm` | `AGENT_MODE=llm` + a provider key | Yes | Live tool-calling evaluation against real models |

In `llm` mode the agent uses the same tools and guardrails, and falls back to
the deterministic path if the provider errors, is rate-limited, or has no key.
The response records `mode="deterministic-fallback"` and the failure reason so
the fallback cannot silently pass as a live success.

### Current model defaults (November 2026)

| Provider | Agent / judge default | Fallback chain |
| :--- | :--- | :--- |
| Gemini | `gemini-3.8-flash` | `gemini-flash-lite-latest`, `gemini-3.6-flash`, `gemini-4-hydrogen` |
| Groq | `openai/gpt-oss-120b` | `openai/gpt-oss-20b` |
| OpenAI | `gpt-6.1-terra` | `gpt-6-luna`, `gpt-6.1-sol` |

The fallback chain is availability-first: `gemini-flash-lite-latest` is a stable
rolling alias that keeps evaluation running when flagship preview tiers are
saturated. The judge also throttles requests (`EVAL_JUDGE_MIN_INTERVAL_SECONDS`,
default 2s) and retries transient 429/503 responses for up to
`EVAL_JUDGE_RETRY_BUDGET_SECONDS` (default 120s).

All providers are accessed through their OpenAI-compatible chat-completions
endpoints, so one judge implementation and one live-agent loop work everywhere.
Override any default with `EVAL_JUDGE_MODEL`, `EVAL_JUDGE_FALLBACKS`,
`AGENT_MODEL`, or `EVAL_JUDGE_PROVIDER`.

---

## Quality gate policy

`scripts/quality_gate.py` reads every `results/deepeval_metrics*.json` plus the
Promptfoo exports and applies these blocking rules:

| Metric | Rule | Why |
| :--- | :---: | :--- |
| Faithfulness | ≥ 0.85 | Answers must be supported by retrieved evidence/tool results |
| Answer Relevancy | ≥ 0.80 | Answer must address the actual request |
| Hallucination | ≥ 0.85 | No invented amounts, dates, or policies (DeepEval ≥ 4.3 scores 1 = grounded) |
| Tool Selection Accuracy | ≥ 0.90 | Correct tool with correct arguments |
| Tool Reasoning G-Eval | ≥ 0.85 | Final answer must translate raw tool output accurately |
| Promptfoo Assertion Rate | 100% | Any assertion regression blocks merge |
| Security Violations | 0 | Prompt injection/PII suites must all pass |
| Red Team Violations | 0 | Only enforced when `--require-redteam` is set |

Important behavior:

- **Missing metrics fail.** A missing judge key does not turn into a silent pass;
  the gate exits 1 unless `--allow-missing-judge` is explicitly supplied.
- **Missing security tests fail.** Tagging a suite `metadata.category: security`
  is required; an empty security set fails closed.
- **Promptfoo failures are not swallowed.** CI uses `continue-on-error` while
  running the suite so the JSON artifact is still produced, then the gate fails.
- **Stale artifacts are removed** by the one-click runners before each run.

Exit codes: `0` = all blocking checks pass, `1` = at least one failure.
`GITHUB_STEP_SUMMARY` is populated automatically when set.

---

## Promptfoo suites

### Main suite (`tests/promptfoo/promptfooconfig.yaml`)

- 10 golden functional cases covering status, refunds, cancellation,
  non-refundable items, shipping, unknown orders, warranty and fallback.
- 6 security/privacy cases tagged `metadata.category: security`.
- Assertions on exact facts, forbidden strings, latency (< 3 s), tool name and
  tool arguments, and guardrail decisions.
- Every attack must be flagged by the guardrail and must not call a tool.
- `tests/promptfoo/assertions.js` provides reusable tool-call assertions.

### Red-team suite (`tests/promptfoo/redteam.yaml`)

Uses the current `redteam:` configuration block with Groq generation/grading,
custom policy probes, and PII/excessive-agency/RBAC/tool-discovery plugins. The
default scan uses the `basic` strategy so it finishes quickly on free-tier
quota; the slower base64/leetspeak/rot13 obfuscation strategies can be enabled
in `redteam.yaml` for a deeper scan. The target is the same Python provider.
`redteam generate` drops `defaultTest.options`, so the runners call
`scripts/inject_redteam_grader.js` to restore the Groq grader with
`reasoning_format: hidden`; a plain `--grader` string stalls on gpt-oss. The
grader uses `gpt-oss-20b` (cheaper quota) while probe generation stays on
`gpt-oss-120b`.

```bash
# Force local generation (no attack prompts leave your machine)
PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION=1 npm run redteam
```

Promptfoo asks for a work email the first time a red-team eval runs outside CI.
Use `CI=true` to skip that prompt in an automated local run. The custom policy
covers prompt-extraction and system-prompt-override attacks because those
plugins require remote generation or a provider that will emit extraction
probes; the main Promptfoo suite also exercises both attacks directly.

The red-team artifact is optional by default. Require it in CI with
`--require-redteam` or `QUALITY_GATE_REQUIRE_REDTEAM=1`, typically on a
scheduled or manually dispatched security run.

---

## CI/CD

`.github/workflows/ai-quality-gate.yml` runs on pushes, pull requests and manual
dispatch:

1. Installs Python 3.13 and Node 24 with dependency caching.
2. Runs the deterministic test suite (must pass).
3. Runs DeepEval LLM metrics when a provider key is configured.
4. Runs the Promptfoo suite; failures are captured, not masked.
5. Optionally runs the red-team scan (`workflow_dispatch` → `run_redteam`).
6. Enforces `scripts/quality_gate.py` and uploads all artifacts.
7. Posts/updates the Markdown report on the pull request and job summary.

### Repository secrets and variables

| Name | Type | Purpose |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | secret | Gemini judge (recommended) |
| `GROQ_API_KEY` | secret | Groq judge fallback / red-team generation and grading |
| `OPENAI_API_KEY` | secret | OpenAI judge fallback |
| `EVAL_JUDGE_PROVIDER` | variable | Override judge provider (default `gemini`) |
| `EVAL_JUDGE_MODEL` | variable | Override judge model |
| `EVAL_JUDGE_FALLBACKS` | variable | Comma-separated fallback chain |

Without any provider secret, the workflow still runs the deterministic and
Promptfoo suites and tells the gate to skip only the LLM metrics. A fork PR does
not receive repository secrets, so LLM metrics are skipped there by design.
The gate still enforces all deterministic and security checks.

---

## Project structure

```
agentic-eval/
├── .github/workflows/ai-quality-gate.yml
├── data/
│   ├── golden_dataset.json          # queries, expected tools/args/context
│   └── knowledge_base.json          # orders, policies, FAQ corpus
├── src/agent/
│   ├── core_agent.py                # deterministic + live tool loop, guardrails
│   ├── settings.py                  # model defaults, keys, runtime settings
│   ├── tools.py                     # order/refund/cancel/KB tools
│   ├── mock_rag.py                  # tokenized lexical retrieval
│   └── __init__.py
├── tests/
│   ├── conftest.py                  # fixtures, artifact lifecycle, metrics log
│   ├── judge.py                     # OpenAI-compatible judge + fallback chain
│   ├── test_agent_behavior.py       # guardrails, isolation, fallback, live smoke
│   ├── test_quality_gate.py         # deterministic policy tests
│   ├── deepeval/
│   │   ├── _metrics.py              # measure, persist, assert
│   │   ├── test_agent_rag.py        # Faithfulness/Relevancy/Hallucination
│   │   └── test_agent_tools.py      # ToolSelectionAccuracy/ToolReasoningGEval
│   └── promptfoo/
│       ├── agent_provider.py        # persistent Python provider bridge
│       ├── assertions.js            # reusable tool-call assertions
│       ├── promptfooconfig.yaml     # functional + security suite
│       └── redteam.yaml             # adversarial configuration
├── scripts/
│   ├── quality_gate.py              # policy engine, report + JSON output
│   ├── run_all_evals.ps1            # Windows runner
│   └── run_all_evals.sh             # Unix runner
├── pyproject.toml                   # pytest config and markers
├── requirements.txt
├── package.json
└── .env.example
```

---

## Test markers

| Marker | Meaning |
| :--- | :--- |
| `fast` | Deterministic, no network |
| `llm` | Calls a live judge or live model; consumes API quota |

Use `pytest tests --ignore=tests/deepeval -m "not llm"` for a completely free
run, or `pytest tests/deepeval -m llm` for judge-backed metrics.

---

## Troubleshooting

**`No module named 'src'` or `No module named 'deepeval'` from Promptfoo**
Set `PROMPTFOO_PYTHON` to the virtualenv interpreter. The included runners do
this automatically. The Python provider also adds a local `.venv` site-packages
directory as a fallback for direct `npx promptfoo` runs.

**`spawn py -3 ENOENT` on Windows**
Set `PROMPTFOO_PYTHON` to the full path of `.venv\Scripts\python.exe`, or use the
PowerShell runner.

**Quality gate fails with `missing` metrics**
That is intentional. Run the LLM suite with a configured key, or add
`--allow-missing-judge` / `QUALITY_GATE_ALLOW_MISSING_JUDGE=1` for a local run
that deliberately has no judge. A failed judge call is not treated as missing:
it is recorded as a zero and fails.

**Gemini returns 503 / rate limits**
The judge chain automatically tries the configured fallback models, then other
providers that have keys (`EVAL_JUDGE_CROSS_PROVIDER=1`). Check
`EVAL_JUDGE_FALLBACKS` if you change the primary model. Free tiers are
RPM/TPM-limited, so the judge serializes metric requests and retries with
backoff; tune `EVAL_JUDGE_MIN_INTERVAL_SECONDS` (default 2),
`EVAL_JUDGE_RETRY_DELAY_SECONDS` (default 6) and
`EVAL_JUDGE_RETRY_BUDGET_SECONDS` (default 120) if your quota differs. The
rolling alias `gemini-flash-lite-latest` is a reliable fallback when high-end
preview models are saturated.

**Promptfoo serves stale responses**
The suite runs with `--no-cache` in npm scripts and CI. Remove `.promptfoo/` if
you changed the provider and still see cached output.

**Red-team generation is rejected**
Provider safety filters can refuse some adversarial generation. The local
configuration uses Groq, which handles the included plugins but refuses
prompt-extraction probes. Lower `numTests` or remove a failing plugin, or run a
manual security review. The main Promptfoo suite still enforces deterministic
injection/PII coverage without generation.

**PR comment step is skipped**
Fork pull requests receive a read-only token and no secrets. The report is
still uploaded as an artifact and written to the job summary.

---

## Security notes

- Mock judges are intentionally disabled: they hide real regressions.
- The guardrail blocks payment-data requests, cross-customer data, prompt
  injection and policy bypass attempts **before** any tool is called.
- `guardrails.flagged` is emitted by the provider, so `not-guardrails`
  assertions prove the block happened rather than merely checking the prose.
- API keys are read from the environment or `.env`, never from YAML.
- `.env`, `results/`, `.venv/` and `node_modules/` are gitignored.
