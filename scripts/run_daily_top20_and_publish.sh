#!/usr/bin/env bash
set -euo pipefail

ROOT="${FIN_QUANT_ROOT:-/home/luochangyu/Financial_Quantification}"
PYTHON="${FIN_QUANT_PYTHON:-/home/luochangyu/anaconda3/envs/py310/bin/python}"
DEPLOY_REPO="${FIN_QUANT_DEPLOY_REPO:-$ROOT/data_cache/github_pages_repo}"
REPO_URL="${FIN_QUANT_REPO_URL:-https://github.com/chyuuuuuuu/Financial_Quantification.git}"
BRANCH="${FIN_QUANT_BRANCH:-main}"
TODAY="${1:-$(date +%F)}"

if [ -n "${GIT_SSH_KEY_FILE:-}" ]; then
  export GIT_SSH_COMMAND="ssh -i $GIT_SSH_KEY_FILE -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
fi

cd "$ROOT"
mkdir -p "$(dirname "$DEPLOY_REPO")" data_cache/daily_top20_snapshots data_cache/formula_breakout_snapshots

echo "[$(date '+%F %T')] run daily Top20 for $TODAY"
"$PYTHON" daily_top20_pipeline.py --run-once --target-date "$TODAY" ${TOP20_EXTRA_ARGS:-}

echo "[$(date '+%F %T')] run formula breakout port for $TODAY"
"$PYTHON" formula_breakout_pipeline.py --run-once --target-date "$TODAY" ${FORMULA_EXTRA_ARGS:-}

if [ ! -d "$DEPLOY_REPO/.git" ]; then
  echo "[$(date '+%F %T')] clone deploy repo to $DEPLOY_REPO"
  git clone "$REPO_URL" "$DEPLOY_REPO"
fi

git -C "$DEPLOY_REPO" fetch origin "$BRANCH"
git -C "$DEPLOY_REPO" checkout "$BRANCH"
git -C "$DEPLOY_REPO" pull --ff-only origin "$BRANCH"

mkdir -p "$DEPLOY_REPO/static/reports" "$DEPLOY_REPO/templates" "$DEPLOY_REPO/scripts" "$DEPLOY_REPO/.github/workflows"
cp daily_top20_pipeline.py "$DEPLOY_REPO/daily_top20_pipeline.py"
cp formula_breakout_pipeline.py "$DEPLOY_REPO/formula_breakout_pipeline.py"
cp formula_breakout_backtest.py "$DEPLOY_REPO/formula_breakout_backtest.py"
cp formula_breakout_cash_backtest.py "$DEPLOY_REPO/formula_breakout_cash_backtest.py"
cp formula_breakout_top2_backtest.py "$DEPLOY_REPO/formula_breakout_top2_backtest.py"
cp formula_breakout_top_slots_limit_backtest.py "$DEPLOY_REPO/formula_breakout_top_slots_limit_backtest.py"
cp formula_breakout_extend_history.py "$DEPLOY_REPO/formula_breakout_extend_history.py"
cp live_formula_trading.py "$DEPLOY_REPO/live_formula_trading.py"
cp app.py "$DEPLOY_REPO/app.py"
cp templates/daily_top20.html "$DEPLOY_REPO/templates/daily_top20.html"
cp templates/formula_breakout.html "$DEPLOY_REPO/templates/formula_breakout.html"
cp templates/live_formula_trading.html "$DEPLOY_REPO/templates/live_formula_trading.html"
cp scripts/run_daily_top20_and_publish.sh "$DEPLOY_REPO/scripts/run_daily_top20_and_publish.sh"
cp scripts/run_live_formula_trading_and_publish.sh "$DEPLOY_REPO/scripts/run_live_formula_trading_and_publish.sh"
cp .github/workflows/pages-report.yml "$DEPLOY_REPO/.github/workflows/pages-report.yml"
cp static/index.html "$DEPLOY_REPO/static/index.html"
cp static/daily-top20.html "$DEPLOY_REPO/static/daily-top20.html"
cp static/formula-breakout.html "$DEPLOY_REPO/static/formula-breakout.html"
cp static/live-trading.html "$DEPLOY_REPO/static/live-trading.html"
cp static/reports/daily_top20.json "$DEPLOY_REPO/static/reports/daily_top20.json"
cp static/reports/formula_breakout.json "$DEPLOY_REPO/static/reports/formula_breakout.json"
if [ -f static/reports/live_formula_trading.json ]; then
  cp static/reports/live_formula_trading.json "$DEPLOY_REPO/static/reports/live_formula_trading.json"
fi
cp static/reports/formula_breakout_*.json "$DEPLOY_REPO/static/reports/"

git -C "$DEPLOY_REPO" add \
  daily_top20_pipeline.py \
  formula_breakout_pipeline.py \
  formula_breakout_backtest.py \
  formula_breakout_cash_backtest.py \
  formula_breakout_top2_backtest.py \
  formula_breakout_top_slots_limit_backtest.py \
  formula_breakout_extend_history.py \
  live_formula_trading.py \
  app.py \
  templates/daily_top20.html \
  templates/formula_breakout.html \
  templates/live_formula_trading.html \
  scripts/run_daily_top20_and_publish.sh \
  scripts/run_live_formula_trading_and_publish.sh \
  .github/workflows/pages-report.yml \
  static/index.html \
  static/daily-top20.html \
  static/formula-breakout.html \
  static/live-trading.html \
  static/reports/daily_top20.json \
  static/reports/formula_breakout.json \
  static/reports/formula_breakout_*.json

if [ -f "$DEPLOY_REPO/static/reports/live_formula_trading.json" ]; then
  git -C "$DEPLOY_REPO" add static/reports/live_formula_trading.json
fi

if git -C "$DEPLOY_REPO" diff --cached --quiet; then
  echo "[$(date '+%F %T')] no public report changes to publish"
  exit 0
fi

git -C "$DEPLOY_REPO" config user.name "${GIT_AUTHOR_NAME:-daily-top20-bot}"
git -C "$DEPLOY_REPO" config user.email "${GIT_AUTHOR_EMAIL:-daily-top20-bot@users.noreply.github.com}"
git -C "$DEPLOY_REPO" commit -m "Update daily stock reports $TODAY"

if [ -n "${GITHUB_TOKEN:-}" ] || [ -n "${GITHUB_TOKEN_FILE:-}" ]; then
  ASKPASS="$(mktemp /tmp/github_askpass_XXXXXX.sh)"
  TOKEN_TMP=""
  cleanup() {
    rm -f "$ASKPASS"
    if [ -n "$TOKEN_TMP" ]; then
      rm -f "$TOKEN_TMP"
    fi
  }
  trap cleanup EXIT

  cat > "$ASKPASS" <<'EOF'
#!/bin/sh
case "$1" in
  *Username*) printf "%s" "x-access-token" ;;
  *Password*)
    if [ -n "$GITHUB_TOKEN" ]; then
      printf "%s" "$GITHUB_TOKEN"
    else
      cat "$GITHUB_TOKEN_FILE"
    fi
    ;;
  *) printf "%s" "" ;;
esac
EOF
  chmod 700 "$ASKPASS"
  GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 git -C "$DEPLOY_REPO" push origin "$BRANCH"
else
  git -C "$DEPLOY_REPO" push origin "$BRANCH"
fi

echo "[$(date '+%F %T')] published daily stock reports $TODAY"
