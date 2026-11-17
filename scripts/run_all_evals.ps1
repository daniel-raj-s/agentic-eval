# One-click evaluation pipeline for Windows PowerShell.
#
#   .\scripts\run_all_evals.ps1                 # deterministic + Promptfoo + LLM metrics (if a key exists)
#   .\scripts\run_all_evals.ps1 -SkipLlm        # free, offline-friendly CI equivalent
#   .\scripts\run_all_evals.ps1 -RunRedteam     # add the adversarial red-team scan
[CmdletBinding()]
param(
    [switch]$SkipLlm,
    [switch]$RunRedteam,
    [switch]$SkipPromptfoo
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$results = Join-Path $root "results"
New-Item -ItemType Directory -Force -Path $results | Out-Null

# Never mix artifacts from a previous run into the gate.
Get-ChildItem -Path $results -Filter "deepeval_metrics*.json" -ErrorAction SilentlyContinue |
    Remove-Item -Force
foreach ($name in @(
    "promptfoo-results.json",
    "promptfoo-redteam-results.json",
    "quality_gate_report.md",
    "quality_gate_summary.json"
)) {
    $path = Join-Path $results $name
    if (Test-Path $path) { Remove-Item $path -Force }
}

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $python) {
    throw "Python was not found. Create .venv or install Python 3.13+."
}

$env:PROMPTFOO_PYTHON = $python
$env:PROMPTFOO_DISABLE_TELEMETRY = "1"
$env:DEEPEVAL_TELEMETRY_OPT_OUT = "1"

$hasKey = (& $python -c "from src.agent.settings import get_api_key; print('1' if any(get_api_key(p) for p in ('gemini','groq','openai')) else '0')").Trim()
$allowMissingJudge = -not ($hasKey -eq "1")
$runLlm = (-not $SkipLlm) -and ($hasKey -eq "1")

$promptfoo = Join-Path $root "node_modules\.bin\promptfoo.cmd"
if (-not (Test-Path $promptfoo)) { $promptfoo = $null }

function Invoke-Promptfoo {
    param([string[]]$Arguments)
    if ($promptfoo) {
        & $promptfoo @Arguments
    } else {
        & npx --no-install promptfoo @Arguments
    }
}

$failedSteps = New-Object System.Collections.Generic.List[string]

Write-Host "`n[1/4] Deterministic test suite" -ForegroundColor Cyan
& $python -m pytest tests --ignore=tests/deepeval -m "not llm" -q
if ($LASTEXITCODE -ne 0) { $failedSteps.Add("deterministic tests") }

if ($runLlm) {
    Write-Host "`n[2/4] DeepEval LLM-judge suite" -ForegroundColor Cyan
    $env:EVAL_KEEP_METRICS = "1"
    & $python -m pytest tests/deepeval -m llm -q
    if ($LASTEXITCODE -ne 0) { $failedSteps.Add("DeepEval LLM metrics") }
    Remove-Item Env:\EVAL_KEEP_METRICS -ErrorAction SilentlyContinue
} else {
    Write-Host "`n[2/4] DeepEval LLM suite skipped (no key or -SkipLlm)." -ForegroundColor DarkYellow
}

if (-not $SkipPromptfoo) {
    Write-Host "`n[3/4] Promptfoo assertion and security suite" -ForegroundColor Cyan
    Invoke-Promptfoo @(
        "eval",
        "-c", "tests/promptfoo/promptfooconfig.yaml",
        "-o", "results/promptfoo-results.json",
        "--no-cache"
    )
    if ($LASTEXITCODE -ne 0) { $failedSteps.Add("Promptfoo suite") }

    if ($RunRedteam) {
        Write-Host "`n[3b/4] Promptfoo red-team scan" -ForegroundColor Cyan
        $env:PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION = "1"
        $previousCI = $env:CI
        # CI=true skips Promptfoo's interactive work-email prompt.
        $env:CI = "true"
        Invoke-Promptfoo @(
            "redteam", "generate",
            "-c", "tests/promptfoo/redteam.yaml",
            "-o", "tests/promptfoo/redteam-generated.yaml",
            "--no-cache"
        )
        if ($LASTEXITCODE -ne 0) {
            $failedSteps.Add("Promptfoo red-team generation")
        } else {
            # redteam generate drops defaultTest.options; restore the Groq
            # grader config (plain --grader strings cannot set reasoning_format).
            & node scripts/inject_redteam_grader.js tests/promptfoo/redteam-generated.yaml
            if ($LASTEXITCODE -ne 0) { $failedSteps.Add("Red-team grader setup") }
            Invoke-Promptfoo @(
                "eval",
                "-c", "tests/promptfoo/redteam-generated.yaml",
                "-o", "results/promptfoo-redteam-results.json",
                "-j", "1",
                "--delay", "10000",
                "--no-cache"
            )
            if ($LASTEXITCODE -ne 0) { $failedSteps.Add("Promptfoo red-team scan") }
        }
        if ($null -eq $previousCI) {
            Remove-Item Env:\CI -ErrorAction SilentlyContinue
        } else {
            $env:CI = $previousCI
        }
        Remove-Item Env:\PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "`n[3/4] Promptfoo suite skipped (-SkipPromptfoo)." -ForegroundColor DarkYellow
}

Write-Host "`n[4/4] Quality gate" -ForegroundColor Cyan
$gateArgs = @("scripts/quality_gate.py")
if ($allowMissingJudge) { $gateArgs += "--allow-missing-judge" }
if ($RunRedteam) { $gateArgs += "--require-redteam" }
& $python @gateArgs
$gateCode = $LASTEXITCODE

if ($failedSteps.Count -gt 0) {
    Write-Host "`nSome evaluation steps failed: $($failedSteps -join ', ')" -ForegroundColor Red
}
Write-Host "Report: $results\quality_gate_report.md" -ForegroundColor DarkGray
exit $gateCode
