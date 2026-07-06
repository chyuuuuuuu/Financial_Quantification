#!/usr/bin/env bash
set -euo pipefail

ROOT="${FIN_QUANT_ROOT:-/home/luochangyu/Financial_Quantification}"
PYTHON="${FIN_QUANT_PYTHON:-/home/luochangyu/anaconda3/envs/py310/bin/python}"
DEPLOY_REPO="${FIN_QUANT_DEPLOY_REPO:-/tmp/Financial_Quantification_github}"
REPO_URL="${FIN_QUANT_REPO_URL:-git@github.com:chyuuuuuuu/Financial_Quantification.git}"
BRANCH="${FIN_QUANT_BRANCH:-main}"
TODAY="${1:-$(date +%F)}"

if [ -n "${GIT_SSH_KEY_FILE:-}" ]; then
  export GIT_SSH_COMMAND="ssh -i $GIT_SSH_KEY_FILE -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
fi

cd "$ROOT"
mkdir -p data_cache/live_trading "$(dirname "$DEPLOY_REPO")"

echo "[$(date '+%F %T')] run live formula trading plan for $TODAY"
"$PYTHON" live_formula_trading.py --run-once --target-date "$TODAY" --refresh --apply-state ${LIVE_EXTRA_ARGS:-}

if [ ! -d "$DEPLOY_REPO/.git" ]; then
  echo "[$(date '+%F %T')] clone deploy repo to $DEPLOY_REPO"
  git clone "$REPO_URL" "$DEPLOY_REPO"
fi

git -C "$DEPLOY_REPO" fetch origin "$BRANCH"
git -C "$DEPLOY_REPO" checkout "$BRANCH"
git -C "$DEPLOY_REPO" pull --ff-only origin "$BRANCH"

mkdir -p "$DEPLOY_REPO/static/reports" "$DEPLOY_REPO/templates" "$DEPLOY_REPO/scripts" "$DEPLOY_REPO/.github/workflows"
cp app.py "$DEPLOY_REPO/app.py"
cp live_formula_trading.py "$DEPLOY_REPO/live_formula_trading.py"
cp formula_breakout_pipeline.py "$DEPLOY_REPO/formula_breakout_pipeline.py"
cp formula_breakout_cash_backtest.py "$DEPLOY_REPO/formula_breakout_cash_backtest.py"
cp formula_breakout_top2_backtest.py "$DEPLOY_REPO/formula_breakout_top2_backtest.py"
cp templates/live_formula_trading.html "$DEPLOY_REPO/templates/live_formula_trading.html"
cp templates/formula_breakout.html "$DEPLOY_REPO/templates/formula_breakout.html"
cp templates/daily_top20.html "$DEPLOY_REPO/templates/daily_top20.html"
cp static/live-trading.html "$DEPLOY_REPO/static/live-trading.html"
cp static/formula-breakout.html "$DEPLOY_REPO/static/formula-breakout.html"
cp static/daily-top20.html "$DEPLOY_REPO/static/daily-top20.html"
cp static/reports/live_formula_trading.json "$DEPLOY_REPO/static/reports/live_formula_trading.json"
cp scripts/run_live_formula_trading_and_publish.sh "$DEPLOY_REPO/scripts/run_live_formula_trading_and_publish.sh"
cp .github/workflows/pages-report.yml "$DEPLOY_REPO/.github/workflows/pages-report.yml"

git -C "$DEPLOY_REPO" add \
  app.py \
  live_formula_trading.py \
  formula_breakout_pipeline.py \
  formula_breakout_cash_backtest.py \
  formula_breakout_top2_backtest.py \
  templates/live_formula_trading.html \
  templates/formula_breakout.html \
  templates/daily_top20.html \
  static/live-trading.html \
  static/formula-breakout.html \
  static/daily-top20.html \
  static/reports/live_formula_trading.json \
  scripts/run_live_formula_trading_and_publish.sh \
  .github/workflows/pages-report.yml

if git -C "$DEPLOY_REPO" diff --cached --quiet; then
  echo "[$(date '+%F %T')] no live trading report changes to publish"
  exit 0
fi

git -C "$DEPLOY_REPO" config user.name "${GIT_AUTHOR_NAME:-live-formula-bot}"
git -C "$DEPLOY_REPO" config user.email "${GIT_AUTHOR_EMAIL:-live-formula-bot@users.noreply.github.com}"
git -C "$DEPLOY_REPO" commit -m "Update live formula trading plan $TODAY"
git -C "$DEPLOY_REPO" push origin "$BRANCH"

echo "[$(date '+%F %T')] published live formula trading plan $TODAY"
