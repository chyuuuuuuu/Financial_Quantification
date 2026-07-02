#!/usr/bin/env python
"""Snapshot daily uptrend candidates and review their next-day behavior."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from main_uptrend_model import _format_amount, normalize_hist


def normalize_code(value: object) -> str:
    return str(value).strip().zfill(6)[-6:]


def prediction_path(log_dir: Path, date_text: str) -> Path:
    return log_dir / f"predictions_{date_text.replace('-', '')}.csv"


def review_path(review_dir: Path, prediction_date: str, eval_date: str) -> Path:
    return review_dir / f"review_{prediction_date.replace('-', '')}_to_{eval_date.replace('-', '')}.csv"


def snapshot(args: argparse.Namespace) -> Dict[str, object]:
    src = Path(args.candidates_file)
    if not src.exists():
        raise FileNotFoundError(f"candidates file not found: {src}")
    df = pd.read_csv(src, dtype={"code": str})
    if args.require_entry_close_retest and "entry_close_retest_signal" in df.columns:
        df = df[df["entry_close_retest_signal"].astype(bool)]
    if "rank" in df.columns:
        df = df.sort_values("rank")
    else:
        df = df.sort_values("final_score", ascending=False)
    out = df.head(args.top_n).copy()
    out.insert(0, "prediction_date", args.prediction_date)
    out.insert(1, "snapshot_created_at", datetime.now().isoformat(timespec="seconds"))
    out["source_candidates_file"] = str(src)
    out["require_entry_close_retest"] = bool(args.require_entry_close_retest)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = prediction_path(log_dir, args.prediction_date)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    summary = {
        "prediction_date": args.prediction_date,
        "source": str(src),
        "rows": int(len(out)),
        "require_entry_close_retest": bool(args.require_entry_close_retest),
        "output": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def load_eval_bar(hist_dir: Path, code: str, eval_date: Optional[str]) -> Optional[pd.Series]:
    path = hist_dir / f"{normalize_code(code)}.csv"
    if not path.exists():
        return None
    hist = normalize_hist(pd.read_csv(path))
    if hist.empty:
        return None
    hist = hist.sort_values("日期").reset_index(drop=True)
    if eval_date:
        target = pd.Timestamp(eval_date)
        row = hist[hist["日期"] == target]
        if row.empty:
            return None
        return row.iloc[-1]
    return hist.iloc[-1]


def review(args: argparse.Namespace) -> Dict[str, object]:
    log_dir = Path(args.log_dir)
    src = Path(args.predictions_file) if args.predictions_file else prediction_path(log_dir, args.prediction_date)
    if not src.exists():
        raise FileNotFoundError(f"predictions file not found: {src}")
    preds = pd.read_csv(src, dtype={"code": str})
    hist_dir = Path(args.hist_dir)
    rows: List[Dict[str, object]] = []
    eval_date_seen = ""
    for row in preds.itertuples(index=False):
        code = normalize_code(getattr(row, "code"))
        bar = load_eval_bar(hist_dir, code, args.eval_date or None)
        if bar is None:
            rows.append({"code": code, "name": getattr(row, "name", ""), "ok": False, "error": "missing_eval_bar"})
            continue
        eval_date = bar["日期"].strftime("%Y-%m-%d")
        eval_date_seen = eval_date_seen or eval_date
        pred_close = float(getattr(row, "close", np.nan))
        eval_open = float(bar["开盘"])
        eval_close = float(bar["收盘"])
        day_pct = float(bar["涨跌幅"]) if "涨跌幅" in bar and np.isfinite(float(bar["涨跌幅"])) else 0.0
        bullish = bool(eval_close > eval_open)
        from_prediction = eval_close / pred_close - 1 if np.isfinite(pred_close) and pred_close > 0 else np.nan
        rows.append(
            {
                "prediction_date": getattr(row, "prediction_date", args.prediction_date),
                "eval_date": eval_date,
                "rank": int(getattr(row, "rank", 0) or 0),
                "code": code,
                "name": getattr(row, "name", ""),
                "prediction_close": round(pred_close, 3) if np.isfinite(pred_close) else np.nan,
                "eval_open": round(eval_open, 3),
                "eval_close": round(eval_close, 3),
                "eval_high": round(float(bar["最高"]), 3),
                "eval_low": round(float(bar["最低"]), 3),
                "eval_pct_chg": round(day_pct, 2),
                "bullish_candle": bullish,
                "return_from_prediction_close": round(float(from_prediction) * 100, 2) if np.isfinite(from_prediction) else np.nan,
                "amount_text": _format_amount(float(bar["成交额"])),
                "model_prob": getattr(row, "model_prob", np.nan),
                "rule_score": getattr(row, "rule_score", np.nan),
                "final_score": getattr(row, "final_score", np.nan),
                "entry_close_retest_signal": getattr(row, "entry_close_retest_signal", np.nan),
                "hit_positive_day": bool(day_pct > 0 and bullish),
                "hit_strong_day": bool(day_pct >= args.strong_pct and bullish),
                "ok": True,
                "error": "",
            }
        )
    out = pd.DataFrame(rows)
    eval_date_out = args.eval_date or eval_date_seen or datetime.now().strftime("%Y-%m-%d")
    review_dir = Path(args.review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    out_path = review_path(review_dir, args.prediction_date, eval_date_out)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    valid = out[out.get("ok", False) == True].copy()
    summary = {
        "prediction_date": args.prediction_date,
        "eval_date": eval_date_out,
        "predictions": int(len(preds)),
        "reviewed": int(len(valid)),
        "positive_day_hits": int(valid["hit_positive_day"].sum()) if not valid.empty else 0,
        "strong_day_hits": int(valid["hit_strong_day"].sum()) if not valid.empty else 0,
        "avg_eval_pct_chg": round(float(valid["eval_pct_chg"].mean()), 2) if not valid.empty else 0.0,
        "median_eval_pct_chg": round(float(valid["eval_pct_chg"].median()), 2) if not valid.empty else 0.0,
        "avg_return_from_prediction_close": round(float(valid["return_from_prediction_close"].mean()), 2) if not valid.empty else 0.0,
        "output": str(out_path),
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not valid.empty:
        cols = [
            "rank",
            "code",
            "name",
            "eval_date",
            "eval_pct_chg",
            "bullish_candle",
            "return_from_prediction_close",
            "hit_positive_day",
            "hit_strong_day",
        ]
        print(valid.sort_values(["hit_strong_day", "eval_pct_chg"], ascending=[False, False])[cols].head(args.print_top).to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot")
    snap.add_argument("--candidates-file", required=True)
    snap.add_argument("--prediction-date", default=datetime.now().strftime("%Y-%m-%d"))
    snap.add_argument("--top-n", type=int, default=80)
    snap.add_argument("--require-entry-close-retest", action="store_true")
    snap.add_argument("--log-dir", default="data_cache/daily_uptrend_predictions")

    rev = sub.add_parser("review")
    rev.add_argument("--prediction-date", required=True)
    rev.add_argument("--eval-date", default="")
    rev.add_argument("--predictions-file", default="")
    rev.add_argument("--hist-dir", default="data_cache/main_uptrend/hist")
    rev.add_argument("--log-dir", default="data_cache/daily_uptrend_predictions")
    rev.add_argument("--review-dir", default="data_cache/daily_uptrend_reviews")
    rev.add_argument("--strong-pct", type=float, default=3.0)
    rev.add_argument("--print-top", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "snapshot":
        snapshot(args)
    elif args.command == "review":
        review(args)


if __name__ == "__main__":
    main()
