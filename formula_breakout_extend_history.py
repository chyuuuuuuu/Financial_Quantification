#!/usr/bin/env python
"""Extend formula breakout daily history cache to a requested start date."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from formula_breakout_pipeline import load_universe, now_text
from main_uptrend_model import StockItem, fetch_hist, normalize_hist


def cached_first_date(path: Path) -> Optional[pd.Timestamp]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["日期"])
        dates = pd.to_datetime(df["日期"], errors="coerce").dropna()
        if dates.empty:
            return None
        return dates.min()
    except Exception:
        try:
            df = normalize_hist(pd.read_csv(path))
        except Exception:
            return None
    if df.empty:
        return None
    return pd.to_datetime(df["日期"], errors="coerce").dropna().min()


def refresh_one(row: object, args: argparse.Namespace) -> Dict[str, object]:
    code = str(getattr(row, "code")).zfill(6)[-6:]
    name = str(getattr(row, "name", ""))
    path = Path(args.history_dir) / f"{code}.csv"
    first = cached_first_date(path)
    target = pd.Timestamp(args.start_date)
    if first is not None and first <= target and not args.force:
        return {
            "code": code,
            "name": name,
            "ok": True,
            "skipped": True,
            "first_date": first.strftime("%Y-%m-%d"),
            "latest_date": "",
            "rows": None,
            "error": "",
        }
    item = StockItem(code=code, name=name)
    _, hist, error = fetch_hist(
        item,
        Path(args.history_dir),
        args.start_date.replace("-", ""),
        args.end_date.replace("-", ""),
        cache_days=-1.0,
        refresh=True,
        retry=args.retry,
    )
    if error or hist.empty:
        return {
            "code": code,
            "name": name,
            "ok": False,
            "skipped": False,
            "first_date": "",
            "latest_date": "",
            "rows": 0,
            "error": error or "empty",
        }
    return {
        "code": code,
        "name": name,
        "ok": True,
        "skipped": False,
        "first_date": pd.to_datetime(hist["日期"]).min().strftime("%Y-%m-%d"),
        "latest_date": pd.to_datetime(hist["日期"]).max().strftime("%Y-%m-%d"),
        "rows": int(len(hist)),
        "error": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2016-07-03")
    parser.add_argument("--end-date", default="2026-07-02")
    parser.add_argument("--universe-file", default="data_cache/volume_contraction_screen_20260701_mainboard_entry_close/refresh_status.csv")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--output", default="data_cache/formula_breakout_backtests/extend_history_status.csv")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    universe = load_universe(Path(args.universe_file))
    rows = []
    total = len(universe)
    ok = 0
    refreshed = 0
    Path(args.history_dir).mkdir(parents=True, exist_ok=True)
    print(f"[{now_text()}] extend histories total={total}, start={args.start_date}, end={args.end_date}, workers={args.workers}")
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(refresh_one, row, args) for row in universe.itertuples(index=False)]
        for done, fut in enumerate(as_completed(futures), start=1):
            item = fut.result()
            rows.append(item)
            ok += int(bool(item["ok"]))
            refreshed += int(bool(item["ok"]) and not bool(item["skipped"]))
            if args.progress_every > 0 and (done % args.progress_every == 0 or done == total):
                print(f"[{now_text()}] extend progress {done}/{total}, ok={ok}, refreshed={refreshed}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["ok", "code"], ascending=[True, True]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[{now_text()}] wrote {out}, ok={ok}, refreshed={refreshed}")


if __name__ == "__main__":
    main()
