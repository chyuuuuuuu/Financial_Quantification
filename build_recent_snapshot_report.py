#!/usr/bin/env python
"""Build the public recent Top20 report from saved daily prediction snapshots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from main_uptrend_model import is_hs_main_board_code, normalize_hist


def normalize_code(value: object) -> str:
    return str(value).strip().zfill(6)[-6:]


def clean_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    return df.replace({np.nan: None}).to_dict(orient="records")


def attach_forward_returns(row: Dict[str, object], history_dir: Path, forward_days: int) -> Dict[str, object]:
    code = normalize_code(row["code"])
    base = float(row.get("close") or 0)
    latest_date = pd.Timestamp(row["latest_date"])
    path = history_dir / f"{code}.csv"
    if base <= 0 or not path.exists():
        for step in range(1, forward_days + 1):
            row[f"date_{step}d"] = ""
            row[f"ret_{step}d_pct"] = np.nan
        row[f"high_{forward_days}d_pct"] = np.nan
        row[f"low_{forward_days}d_pct"] = np.nan
        row["win_3d"] = False
        return row
    try:
        hist = normalize_hist(pd.read_csv(path)).sort_values("日期")
    except Exception:
        hist = pd.DataFrame()
    if hist.empty:
        return attach_forward_returns({**row, "close": base}, history_dir / "__missing__", forward_days)
    future = hist[pd.to_datetime(hist["日期"]).dt.normalize() > latest_date.normalize()].head(forward_days)
    for step in range(1, forward_days + 1):
        if len(future) >= step:
            item = future.iloc[step - 1]
            row[f"date_{step}d"] = pd.Timestamp(item["日期"]).strftime("%Y-%m-%d")
            row[f"ret_{step}d_pct"] = round((float(item["收盘"]) / base - 1.0) * 100.0, 3)
        else:
            row[f"date_{step}d"] = ""
            row[f"ret_{step}d_pct"] = np.nan
    if not future.empty:
        row[f"high_{forward_days}d_pct"] = round((float(future["最高"].max()) / base - 1.0) * 100.0, 3)
        row[f"low_{forward_days}d_pct"] = round((float(future["最低"].min()) / base - 1.0) * 100.0, 3)
    else:
        row[f"high_{forward_days}d_pct"] = np.nan
        row[f"low_{forward_days}d_pct"] = np.nan
    ret = row.get(f"ret_{forward_days}d_pct", np.nan)
    row["win_3d"] = bool(np.isfinite(ret) and float(ret) > 0)
    return row


def summarize(daily_top: pd.DataFrame, forward_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret_col = f"ret_{forward_days}d_pct"
    daily_rows = []
    for day, group in daily_top.groupby("latest_date", sort=True):
        valid = group.dropna(subset=[ret_col])
        daily_rows.append(
            {
                "date": day,
                "count": int(len(group)),
                "mean_ret_3d_pct": round(float(valid[ret_col].mean()), 3) if not valid.empty else np.nan,
                "median_ret_3d_pct": round(float(valid[ret_col].median()), 3) if not valid.empty else np.nan,
                "win_rate_pct": round(float((valid[ret_col] > 0).mean() * 100.0), 2) if not valid.empty else np.nan,
                "benchmark_mean_3d_pct": np.nan,
                "benchmark_win_rate_pct": np.nan,
            }
        )
    signal_rows = []
    for signal, group in daily_top.groupby("signal_type", sort=True):
        valid = group.dropna(subset=[ret_col])
        signal_rows.append(
            {
                "signal_type": signal,
                "count": int(len(group)),
                "mean_ret_3d_pct": round(float(valid[ret_col].mean()), 3) if not valid.empty else np.nan,
                "median_ret_3d_pct": round(float(valid[ret_col].median()), 3) if not valid.empty else np.nan,
                "win_rate_pct": round(float((valid[ret_col] > 0).mean() * 100.0), 2) if not valid.empty else np.nan,
            }
        )
    return pd.DataFrame(daily_rows), pd.DataFrame(signal_rows)


def run(args: argparse.Namespace) -> Dict[str, object]:
    snapshot_dir = Path(args.snapshot_dir)
    history_dir = Path(args.history_dir)
    rows: List[Dict[str, object]] = []
    for path in sorted(snapshot_dir.glob("predictions_*.csv")):
        date_raw = path.stem.replace("predictions_", "")
        snapshot_date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        if snapshot_date < args.start_date:
            continue
        df = pd.read_csv(path, dtype={"code": str})
        if df.empty:
            continue
        df["code"] = df["code"].map(normalize_code)
        df["name"] = df.get("name", df["code"]).fillna("").astype(str)
        df = df[~df["name"].str.contains("ST|退|N|C", case=False, na=False)].copy()
        df = df[df["code"].map(is_hs_main_board_code)].copy()
        df = df.sort_values(["rank", "final_score"], ascending=[True, False]).head(args.top_n).copy()
        for rank, item in enumerate(df.to_dict(orient="records"), start=1):
            latest_date = str(item.get("latest_date") or snapshot_date)[:10]
            row = {
                "signal_priority": int(item.get("signal_priority") or 1),
                "signal_type": item.get("signal_type") or "",
                "code": normalize_code(item.get("code")),
                "name": item.get("name") or "",
                "latest_date": latest_date,
                "entry_date": str(item.get("entry_date") or "")[:10],
                "trigger_date": str(item.get("trigger_date") or "")[:10] if item.get("trigger_date") == item.get("trigger_date") else "",
                "entry_age": item.get("entry_age"),
                "trigger_age": item.get("trigger_age"),
                "close": round(float(item.get("close") or 0), 3),
                "pct_chg": round(float(item.get("pct_chg") or 0), 2),
                "amount": float(item.get("amount") or 0),
                "amount_text": item.get("amount_text") or "",
                "entry_volume_ratio_20": item.get("entry_volume_ratio_20"),
                "entry_amount_ratio_20": item.get("entry_amount_ratio_20"),
                "trigger_volume_vs_entry": item.get("trigger_volume_vs_entry"),
                "volume_10_vs_entry": item.get("volume_10_vs_entry"),
                "amount_10_vs_entry": item.get("amount_10_vs_entry"),
                "entry_close_breakout_gap": item.get("entry_close_breakout_gap"),
                "macd_water_ok": item.get("macd_water_ok"),
                "macd_min_dif": item.get("macd_min_dif"),
                "macd_min_dea": item.get("macd_min_dea"),
                "model_prob": item.get("model_prob"),
                "rule_score": item.get("rule_score"),
                "final_score": round(float(item.get("final_score") or 0), 2),
                "reason": item.get("reason") or "",
                "rank": rank,
            }
            rows.append(attach_forward_returns(row, history_dir, args.forward_days))
    daily_top = pd.DataFrame(rows)
    if not daily_top.empty:
        daily_top = daily_top.sort_values(["latest_date", "rank"]).reset_index(drop=True)
    daily_summary, signal_summary = summarize(daily_top, args.forward_days)
    dates = sorted(daily_top["latest_date"].dropna().unique().tolist()) if not daily_top.empty else []
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start": args.start_date,
        "end": dates[-1] if dates else args.start_date,
        "top_n": args.top_n,
        "forward_days": args.forward_days,
        "score_mode": "snapshot-top20-mainboard",
        "universe_size": None,
        "history_ok": None,
        "trading_days": dates,
        "daily_summary": clean_records(daily_summary),
        "signal_summary": clean_records(signal_summary),
        "top_rows": clean_records(daily_top),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_top.to_csv(out_dir / "recent_snapshot_top20.csv", index=False, encoding="utf-8-sig")
    daily_summary.to_csv(out_dir / "recent_snapshot_daily_summary.csv", index=False, encoding="utf-8-sig")
    signal_summary.to_csv(out_dir / "recent_snapshot_signal_summary.csv", index=False, encoding="utf-8-sig")
    template = Path(args.template)
    if template.exists():
        static_root = output_json.parent.parent
        html = template.read_text(encoding="utf-8")
        (static_root / "index.html").write_text(html, encoding="utf-8")
        (static_root / "june-backtest.html").write_text(html, encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2026-06-25")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--forward-days", type=int, default=3)
    parser.add_argument("--snapshot-dir", default="data_cache/daily_uptrend_predictions")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--output-json", default="static/reports/june_top20_backtest.json")
    parser.add_argument("--output-dir", default="data_cache/recent_snapshot_report")
    parser.add_argument("--template", default="templates/june_backtest.html")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
