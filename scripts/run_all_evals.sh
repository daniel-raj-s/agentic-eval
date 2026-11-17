#!/usr/bin/env bash
# One-click evaluation pipeline for macOS/Linux.
#
#   ./scripts/run_all_evals.sh                # deterministic + Promptfoo + LLM metrics (if a key exists)
#   ./scripts/run_all_evals.sh --skip-llm     # free, offline-friendly CI equivalent
#   ./scripts/run_all_evals.sh --redteam      # add the adversarial red-team scan
set -u

SKIP_LLM=0
RUN_REDTEAM=0
SKIP_PROMPTFOO=0
for arg in "$@"; do
  case "$arg" in
    --skip-llm) SKIP_LLM=1 ;;
    --redteam) RUN_REDTEAM=1 ;;
    --skip-promptfoo) SKIP_PROMPTFOO=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULTS="$ROOT/results"
mkdir -p "$RESULTS"
rm -f "$RESULTS"/deepeval_metrics*.json \
      "$RESULTS/promptfoo-results.json" \
      "$RESULTS/promptfoo-redteam-results.json" \
      "$RESULTS/quality_gate_report.md" \
      "$RESULTS/quality_gate_summary.json"

PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3 || command -v python || true)"
fi
if [ -z "$PYTHON" ]; then
  echo "Python was not found. Create .venv or install Python 3.13+." >&2
  exit 2
fi

export PROMPTFOO_PYTHON="$PYTHON"
export PROMPTFOO_DISABLE_TELEMETRY=1
export DEEPEVAL_TELEMETRY_OPT_OUT=1

HAS_KEY="$("$PYTHON" -c "from src.agent.settings import get_api_key; print('1' if any(get_api_key(p) for p in ('gemini','groq','openai')) else '0')")"
PROMPTFOO_BIN="$ROOT/node_modules/.bin/promptfoo"
if [ ! -x "$PROMPTFOO_BIN" ]; then
  PROMPTFOO_BIN=""
fi

run_promptfoo() {
  if [ -n "$PROMPTFOO_BIN" ]; then
    "$PROMPTFOO_BIN" "$@"
  else
    npx --no-install promptfoo "$@"
  fi
}

FAILED=()

echo ""
echo "[1/4] Deterministic test suite"
"$PYTHON" -m pytest tests --ignore=tests/deepeval -m "not llm" -q || FAILED+=("deterministic tests")

if [ "$SKIP_LLM" -eq 0 ] && [ "$HAS_KEY" = "1" ]; then
  echo ""
  echo "[2/4] DeepEval LLM-judge suite"
  EVAL_KEEP_METRICS=1 "$PYTHON" -m pytest tests/deepeval -m llm -q || FAILED+=("DeepEval LLM metrics")
else
  echo ""
  echo "[2/4] DeepEval LLM suite skipped (no key or --skip-llm)."
fi

if [ "$SKIP_PROMPTFOO" -eq 0 ]; then
  echo ""
  echo "[3/4] Promptfoo assertion and security suite"
  run_promptfoo eval \
    -c tests/promptfoo/promptfooconfig.yaml \
    -o results/promptfoo-results.json \
    --no-cache || FAILED+=("Promptfoo suite")

  if [ "$RUN_REDTEAM" -eq 1 ]; then
    echo ""
    echo "[3b/4] Promptfoo red-team scan"
    export PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION=1
    if run_promptfoo redteam generate \
      -c tests/promptfoo/redteam.yaml \
      -o tests/promptfoo/redteam-generated.yaml \
      --no-cache; then
      # redteam generate drops defaultTest.options; restore the Groq grader
      # config (a plain --grader string cannot set reasoning_format).
      if node scripts/inject_redteam_grader.js tests/promptfoo/redteam-generated.yaml; then
        CI=true run_promptfoo eval \
          -c tests/promptfoo/redteam-generated.yaml \
          -o results/promptfoo-redteam-results.json \
          -j 1 \
          --delay 10000 \
          --no-cache || FAILED+=("Promptfoo red-team scan")
      else
        FAILED+=("Red-team grader setup")
      fi
    else
      FAILED+=("Promptfoo red-team generation")
    fi
    unset PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION
  fi
else
  echo ""
  echo "[3/4] Promptfoo suite skipped (--skip-promptfoo)."
fi

GATE_ARGS=(scripts/quality_gate.py)
if [ "$SKIP_LLM" -eq 1 ] || [ "$HAS_KEY" != "1" ]; then
  GATE_ARGS+=(--allow-missing-judge)
fi
if [ "$RUN_REDTEAM" -eq 1 ]; then
  GATE_ARGS+=(--require-redteam)
fi

echo ""
echo "[4/4] Quality gate"
"$PYTHON" "${GATE_ARGS[@]}"
GATE_CODE=$?

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo ""
  echo "Some evaluation steps failed: ${FAILED[*]}" >&2
fi
echo "Report: $RESULTS/quality_gate_report.md"
exit "$GATE_CODE"
