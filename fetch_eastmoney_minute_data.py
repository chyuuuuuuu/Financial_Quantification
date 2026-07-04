#!/usr/bin/env python
"""Fetch Eastmoney 1-minute A-share bars into a local cache.

Eastmoney's trends2 endpoint currently returns recent 1-minute bars only
(`ndays`, typically 5 trading days). The historical kline endpoint is not
reliable from this environment, so this script keeps the 1-minute pull explicit
and records coverage in a summary file.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd


BASE_URL = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"


def normalize_code(value: object) -> str:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else ""


def market_prefix(code: str) -> str:
    return "1" if code.startswith("6") else "0"


def collect_codes_from_operations(path: Path, since_date: str) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    codes: set[str] = set()
    for day in data.get("operations", []):
        date_text = str(day.get("date") or "")
        if since_date and date_text < since_date:
            continue
        for item in day.get("items", []):
            code = normalize_code(item.get("code"))
            if code:
                codes.add(code)
    for holding in data.get("open_holdings", []):
        code = normalize_code(holding.get("code"))
        if code:
            codes.add(code)
    return sorted(codes)


def parse_codes(raw_codes: str) -> list[str]:
    codes = [normalize_code(item) for item in raw_codes.replace("，", ",").split(",")]
    return sorted({code for code in codes if code})


def build_url(code: str, ndays: int) -> str:
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "ndays": str(ndays),
        "iscr": "0",
        "secid": f"{market_prefix(code)}.{code}",
    }
    return f"{BASE_URL}?{urlencode(params)}"


def curl_json(url: str, timeout: int, retries: int, sleep_seconds: float) -> dict:
    last_error = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [
                "curl",
                "--http1.1",
                "--silent",
                "--show-error",
                "--max-time",
                str(timeout),
                "-A",
                "Mozilla/5.0",
                "-H",
                "Referer: https://quote.eastmoney.com/",
                url,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        last_error = (result.stderr or result.stdout or f"curl exit {result.returncode}").strip()
        if attempt < retries:
            time.sleep(sleep_seconds)
    raise RuntimeError(last_error or "empty curl response")


def trends_to_frame(code: str, payload: dict) -> pd.DataFrame:
    data = payload.get("data") or {}
    trends = data.get("trends") or []
    rows = []
    for raw in trends:
        parts = str(raw).split(",")
        if len(parts) < 8:
            continue
        rows.append(
            {
                "code": code,
                "time": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
                "amount": parts[6],
                "avg_price": parts[7],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for col in ["open", "close", "high", "low", "volume", "amount", "avg_price"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time")
    frame["date"] = frame["time"].dt.strftime("%Y-%m-%d")
    frame["time"] = frame["time"].dt.strftime("%Y-%m-%d %H:%M")
    return frame[
        [
            "code",
            "date",
            "time",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "avg_price",
        ]
    ]


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", default="", help="Comma-separated stock codes.")
    parser.add_argument("--operations-json", default="data_cache/formula_breakout_backtests/formula_breakout_top3_limit_backtest_10y_full.json")
    parser.add_argument("--since-date", default="2026-06-29", help="Collect codes from operations on or after this date.")
    parser.add_argument("--output-dir", default="data_cache/minute_bars/eastmoney_1m")
    parser.add_argument("--ndays", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of codes to fetch.")
    args = parser.parse_args()

    if args.codes.strip():
        codes = parse_codes(args.codes)
    else:
        codes = collect_codes_from_operations(Path(args.operations_json), args.since_date)
    if args.limit > 0:
        codes = codes[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for idx, code in enumerate(codes, start=1):
        status = {"code": code, "ok": False, "rows": 0, "first_time": "", "last_time": "", "error": ""}
        try:
            payload = curl_json(build_url(code, args.ndays), args.timeout, args.retries, args.sleep)
            frame = trends_to_frame(code, payload)
            if frame.empty:
                raise ValueError("empty minute data")
            frame.to_csv(output_dir / f"{code}.csv", index=False, encoding="utf-8-sig")
            status.update(
                {
                    "ok": True,
                    "rows": int(len(frame)),
                    "first_time": str(frame.iloc[0]["time"]),
                    "last_time": str(frame.iloc[-1]["time"]),
                }
            )
        except Exception as exc:  # noqa: BLE001 - record per-code failures and continue.
            status["error"] = f"{type(exc).__name__}: {exc}"
        summary_rows.append(status)
        print(f"[{idx}/{len(codes)}] {code} rows={status['rows']} ok={status['ok']} {status['error']}")
        time.sleep(args.sleep)

    write_csv(output_dir / "fetch_summary.csv", summary_rows)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "eastmoney_trends2",
        "period": "1m",
        "ndays": args.ndays,
        "since_date_for_code_selection": args.since_date,
        "requested_codes": len(codes),
        "ok_codes": sum(1 for row in summary_rows if row["ok"]),
        "failed_codes": sum(1 for row in summary_rows if not row["ok"]),
        "output_dir": str(output_dir),
        "note": "Eastmoney trends2 returns recent 1-minute bars only, normally about 5 trading days.",
    }
    (output_dir / "fetch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
