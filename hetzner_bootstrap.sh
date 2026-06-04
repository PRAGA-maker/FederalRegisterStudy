#!/usr/bin/env bash
# ============================================================================
# Hetzner persistent bootstrap + full-year production rerun
# ----------------------------------------------------------------------------
# Regenerates all FederalRegisterStudy data for ONE year on the box, with the
# new primary-source response extraction (dev). Persists outputs to /root/fr_data
# so future iterations build on them.
#
# *** DO NOT TEAR DOWN THE BOX BETWEEN RUNS. Persistent data is the whole point. ***
#
# One-time per box (the script handles it): clones repo, installs uv + deps.
# Each run: updates code to latest dev, runs the pipeline for <year>, snapshots
# outputs to /root/fr_data/<year>/.
#
# Setup (once): create /root/fr_keys.sh with the API keys, e.g. (no $env: prefix):
#   export REGS_API_KEYS="key1:5000,key2:2000,key3:2000"
#   export OPENAI_API_KEY="sk-..."
#   export GEMINI_API_KEY="AIza..."
#   export XAI_API_KEY="xai-jQAXAq2k...."     # latest valid key (from plainpolitics_webdev/.env)
#
# Usage:
#   bash hetzner_bootstrap.sh 2014 --test     # smoke test first (limit_docs=50, ~minutes)
#   bash hetzner_bootstrap.sh 2014            # full production run (hours)
#   then 2015, 2024, 2025 in that order.
# ============================================================================
set -euo pipefail

REPO_URL="https://github.com/PRAGA-maker/FederalRegisterStudy.git"
REPO_DIR="/root/FederalRegisterStudy"
PERSIST="/root/fr_data"            # persistent across runs — NEVER delete
KEYS_FILE="/root/fr_keys.sh"

YEAR="${1:?usage: hetzner_bootstrap.sh <year> [extra pipeline args e.g. --test]}"; shift || true
mkdir -p "$PERSIST/logs" "$PERSIST/$YEAR"

echo "==> [1/6] Code: clone/update to latest origin/dev"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone -b dev "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch origin dev
  git -C "$REPO_DIR" checkout -f dev
  git -C "$REPO_DIR" reset --hard origin/dev   # data is safe in $PERSIST; repo is code-only here
fi
cd "$REPO_DIR"

echo "==> [2/6] Python env + deps (uv)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e . >/dev/null

echo "==> [3/6] Load + sanity-check API keys"
[ -f "$KEYS_FILE" ] || { echo "ERROR: create $KEYS_FILE with the API keys (see header)."; exit 1; }
# tolerate a pasted PowerShell-style keys.txt ($env:VAR=...) by stripping the prefix
sed -E 's/^\$env:/export /' "$KEYS_FILE" > /tmp/_frkeys.sh
# shellcheck disable=SC1091
source /tmp/_frkeys.sh
: "${REGS_API_KEYS:?REGS_API_KEYS not set}"
: "${XAI_API_KEY:?XAI_API_KEY not set}"
echo "    keys present: REGS, XAI$( [ -n "${GEMINI_API_KEY:-}" ] && echo ', GEMINI' )$( [ -n "${OPENAI_API_KEY:-}" ] && echo ', OPENAI' )"
xai_http=$(curl -s -o /dev/null -w "%{http_code}" https://api.x.ai/v1/models -H "Authorization: Bearer $XAI_API_KEY")
[ "$xai_http" = "200" ] || { echo "ERROR: XAI key invalid (HTTP $xai_http). Refresh from plainpolitics_webdev/.env"; exit 1; }
echo "    xAI key OK (200)"

echo "==> [4/6] Clean any stale in-repo CSVs for $YEAR (force fresh regeneration)"
rm -f stratification_scripts/output/federal_register_${YEAR}_comments.csv \
      stratification_scripts/output/makeup_data_${YEAR}.csv \
      stratification_scripts/makeup/data/*_${YEAR}.csv 2>/dev/null || true

echo "==> [5/6] Run full pipeline for $YEAR (production: ALL docs, stratified comments)"
LOG="$PERSIST/logs/run_${YEAR}_$(date +%Y%m%d_%H%M%S).log"
echo "    logging to $LOG"
nohup python -m stratification_scripts --years "$YEAR" --verbose "$@" > "$LOG" 2>&1 &
PID=$!
echo "    pipeline PID=$PID. Tail with: tail -f $LOG"
wait "$PID" || { echo "PIPELINE FAILED (exit $?). See $LOG"; exit 1; }

echo "==> [6/6] Snapshot outputs to persistent storage ($PERSIST/$YEAR)"
cp -f stratification_scripts/output/federal_register_${YEAR}_comments.csv "$PERSIST/$YEAR/" 2>/dev/null || true
cp -f stratification_scripts/output/makeup_data_${YEAR}.csv "$PERSIST/$YEAR/" 2>/dev/null || true
cp -f stratification_scripts/makeup/data/*_${YEAR}.csv "$PERSIST/$YEAR/" 2>/dev/null || true
cp -rf stratification_scripts/makeup/data/plots/${YEAR} "$PERSIST/$YEAR/plots" 2>/dev/null || true
echo "==> DONE: year $YEAR persisted under $PERSIST/$YEAR. DO NOT tear down the box."
