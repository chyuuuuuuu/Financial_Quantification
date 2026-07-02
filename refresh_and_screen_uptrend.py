#!/usr/bin/env python
"""Refresh latest daily bars and screen current pre-main-run candidates."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch

from latent_uptrend_model import TemporalAttentionNet
from main_uptrend_model import StockItem, _now_text, is_hs_main_board_code, normalize_hist
from volume_contraction_breakout_model import SEQ_COLS, score_current


def normalize_code(value: object) -> str:
    return str(value).strip().zfill(6)[-6:]


def market_id(code: str) -> str:
    return "1" if code.startswith("6") else "0"


def tencent_symbol(code: str) -> str:
    return ("sh" if code.startswith("6") else "sz") + code


def fetch_tencent_recent(code: str, retry: int = 3) -> Tuple[Optional[pd.DataFrame], str]:
    symbol = tencent_symbol(code)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,20,qfq"
    last_error = ""
    for attempt in range(retry):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            payload = resp.json().get("data") or {}
            item = payload.get(symbol) or {}
            rows_raw = item.get("qfqday") or item.get("day") or []
            if not rows_raw:
                return None, "empty"

            latest_date = ""
            latest_amount = np.nan
            latest_turnover = 0.0
            qt = (item.get("qt") or {}).get(symbol) or []
            if len(qt) > 38:
                ts = str(qt[30])
                if len(ts) >= 8:
                    latest_date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
                try:
                    latest_amount = float(str(qt[35]).split("/")[2])
                except Exception:
                    latest_amount = np.nan
                try:
                    latest_turnover = float(qt[38])
                except Exception:
                    latest_turnover = 0.0

            rows = []
            for vals in rows_raw:
                if len(vals) < 6:
                    continue
                date = str(vals[0])
                open_ = float(vals[1])
                close = float(vals[2])
                high = float(vals[3])
                low = float(vals[4])
                volume = float(vals[5])
                avg_price = np.nanmean([open_, close, high, low])
                amount = volume * 100.0 * avg_price if np.isfinite(avg_price) else 0.0
                turnover = 0.0
                if date == latest_date:
                    if np.isfinite(latest_amount):
                        amount = latest_amount
                    turnover = latest_turnover
                rows.append(
                    {
                        "日期": date,
                        "股票代码": code,
                        "开盘": open_,
                        "收盘": close,
                        "最高": high,
                        "最低": low,
                        "成交量": volume,
                        "成交额": amount,
                        "换手率": turnover,
                    }
                )
            return pd.DataFrame(rows), ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.4 * (attempt + 1))
    return None, last_error


def parse_tencent_quote_line(line: str, target_date: str) -> Optional[Dict[str, object]]:
    if not line or "~" not in line:
        return None
    try:
        payload = line.split('="', 1)[1].rsplit('"', 1)[0]
    except Exception:
        return None
    vals = payload.split("~")
    if len(vals) < 39:
        return None
    code = normalize_code(vals[2])
    ts = str(vals[30])
    if len(ts) < 8:
        return None
    date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
    if date != target_date:
        return None
    try:
        open_ = float(vals[5])
        close = float(vals[3])
        high = float(vals[33])
        low = float(vals[34])
        volume = float(vals[36])
        pct_chg = float(vals[32])
        chg = float(vals[31])
        turnover = float(vals[38])
        amount_text = str(vals[35])
        amount = float(amount_text.split("/")[2]) if "/" in amount_text and len(amount_text.split("/")) >= 3 else float(vals[57]) * 10000.0
        prev_close = float(vals[4])
        amplitude = (high - low) / prev_close * 100.0 if prev_close > 0 else 0.0
    except Exception:
        return None
    return {
        "日期": date,
        "股票代码": code,
        "开盘": open_,
        "收盘": close,
        "最高": high,
        "最低": low,
        "成交量": volume,
        "成交额": amount,
        "振幅": amplitude,
        "涨跌幅": pct_chg,
        "涨跌额": chg,
        "换手率": turnover,
    }


def fetch_tencent_quote_batch(codes: List[str], target_date: str, retry: int = 3) -> Tuple[List[Dict[str, object]], str]:
    symbols = [tencent_symbol(code) for code in codes]
    url = "https://qt.gtimg.cn/q=" + ",".join(symbols)
    last_error = ""
    for attempt in range(retry):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            text = resp.content.decode("gbk", errors="ignore")
            rows = []
            for line in text.splitlines():
                row = parse_tencent_quote_line(line, target_date)
                if row is not None:
                    rows.append(row)
            return rows, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.4 * (attempt + 1))
    return [], last_error


def fetch_eastmoney_recent(code: str, begin: str, end: str, retry: int = 3) -> Tuple[Optional[pd.DataFrame], str]:
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={market_id(code)}.{code}&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        f"&klt=101&fqt=1&beg={begin}&end={end}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Referer": "https://quote.eastmoney.com/",
    }
    last_error = ""
    for attempt in range(retry):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            payload = resp.json().get("data") or {}
            rows = []
            for item in payload.get("klines") or []:
                vals = item.split(",")
                if len(vals) < 11:
                    continue
                rows.append(
                    {
                        "日期": vals[0],
                        "股票代码": code,
                        "开盘": float(vals[1]),
                        "收盘": float(vals[2]),
                        "最高": float(vals[3]),
                        "最低": float(vals[4]),
                        "成交量": float(vals[5]),
                        "成交额": float(vals[6]),
                        "振幅": float(vals[7]),
                        "涨跌幅": float(vals[8]),
                        "涨跌额": float(vals[9]),
                        "换手率": float(vals[10]),
                    }
                )
            if not rows:
                return None, "empty"
            return pd.DataFrame(rows), ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.4 * (attempt + 1))
    return None, last_error


def merge_history(path: Path, latest: pd.DataFrame) -> Tuple[str, int]:
    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, latest], ignore_index=True)
    else:
        merged = latest
    hist = normalize_hist(merged)
    hist["股票代码"] = path.stem
    prev_close = hist["收盘"].shift(1).replace(0, np.nan)
    hist["涨跌额"] = (hist["收盘"] - prev_close).fillna(0.0)
    hist["涨跌幅"] = ((hist["收盘"] / prev_close - 1.0) * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    hist["振幅"] = (((hist["最高"] - hist["最低"]) / prev_close) * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    before = len(normalize_hist(pd.read_csv(path))) if path.exists() else 0
    hist.to_csv(path, index=False, encoding="utf-8-sig")
    latest_date = hist["日期"].max().strftime("%Y-%m-%d") if not hist.empty else ""
    return latest_date, int(len(hist) - before)


def load_universe(path: Path, exclude_chinext: bool, main_board_only: bool) -> List[StockItem]:
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].map(normalize_code)
    if "name" in df.columns:
        df = df[~df["name"].fillna("").astype(str).str.contains("ST|退|N|C", case=False, na=False)]
    if main_board_only:
        df = df[df["code"].map(is_hs_main_board_code)]
    if exclude_chinext:
        df = df[~df["code"].str.startswith(("300", "301"))]
    items = []
    for row in df.drop_duplicates("code").itertuples(index=False):
        items.append(
            StockItem(
                code=str(row.code),
                name=str(getattr(row, "name", row.code)),
                amount=float(getattr(row, "amount", 0) or 0),
                turnover=float(getattr(row, "turnover", 0) or 0),
                pct_chg=float(getattr(row, "pct_chg", 0) or 0),
                close=float(getattr(row, "close", 0) or 0),
            )
        )
    return items


def refresh_histories(items: List[StockItem], args: argparse.Namespace) -> pd.DataFrame:
    hist_dir = Path(args.history_dir)
    hist_dir.mkdir(parents=True, exist_ok=True)
    statuses = []

    def work(item: StockItem) -> Dict[str, object]:
        path = hist_dir / f"{item.code}.csv"
        if args.skip_current and path.exists():
            try:
                cached = normalize_hist(pd.read_csv(path))
                cached_latest = cached["日期"].max().strftime("%Y-%m-%d") if not cached.empty else ""
                if cached_latest == args.target_date:
                    return {"code": item.code, "name": item.name, "ok": True, "latest_date": cached_latest, "added_rows": 0, "error": "skip_current"}
            except Exception:
                pass
        if args.eastmoney_only:
            latest, error = fetch_eastmoney_recent(item.code, args.begin, args.end, retry=args.retry)
        else:
            latest, error = fetch_tencent_recent(item.code, retry=args.retry)
            if latest is None or latest.empty:
                latest, error = fetch_eastmoney_recent(item.code, args.begin, args.end, retry=args.retry)
        if latest is None or latest.empty:
            return {"code": item.code, "name": item.name, "ok": False, "latest_date": "", "added_rows": 0, "error": error or "empty"}
        try:
            latest_date, added_rows = merge_history(path, latest)
            return {"code": item.code, "name": item.name, "ok": True, "latest_date": latest_date, "added_rows": added_rows, "error": ""}
        except Exception as exc:
            return {"code": item.code, "name": item.name, "ok": False, "latest_date": "", "added_rows": 0, "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(work, item): item for item in items}
        for i, fut in enumerate(as_completed(future_map), start=1):
            statuses.append(fut.result())
            if i % args.progress_every == 0 or i == len(items):
                ok = sum(1 for row in statuses if row["ok"])
                latest_616 = sum(1 for row in statuses if row["latest_date"] == args.target_date)
                print(f"[{_now_text()}] refresh progress {i}/{len(items)}, ok={ok}, latest_{args.target_date}={latest_616}", flush=True)
    return pd.DataFrame(statuses)


def refresh_histories_by_quote(items: List[StockItem], args: argparse.Namespace) -> pd.DataFrame:
    hist_dir = Path(args.history_dir)
    hist_dir.mkdir(parents=True, exist_ok=True)
    statuses: Dict[str, Dict[str, object]] = {
        item.code: {"code": item.code, "name": item.name, "ok": False, "latest_date": "", "added_rows": 0, "error": "not_requested"}
        for item in items
    }
    codes = [item.code for item in items]
    code_to_name = {item.code: item.name for item in items}
    batches = [codes[i : i + args.quote_batch_size] for i in range(0, len(codes), args.quote_batch_size)]
    done = 0
    for batch_no, batch in enumerate(batches, start=1):
        rows, error = fetch_tencent_quote_batch(batch, args.target_date, retry=args.retry)
        returned = {str(row["股票代码"]): row for row in rows}
        for code in batch:
            path = hist_dir / f"{code}.csv"
            if code in returned:
                try:
                    latest_date, added_rows = merge_history(path, pd.DataFrame([returned[code]]))
                    statuses[code] = {
                        "code": code,
                        "name": code_to_name.get(code, code),
                        "ok": True,
                        "latest_date": latest_date,
                        "added_rows": added_rows,
                        "error": "",
                    }
                except Exception as exc:
                    statuses[code] = {
                        "code": code,
                        "name": code_to_name.get(code, code),
                        "ok": False,
                        "latest_date": "",
                        "added_rows": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            else:
                statuses[code] = {
                    "code": code,
                    "name": code_to_name.get(code, code),
                    "ok": False,
                    "latest_date": "",
                    "added_rows": 0,
                    "error": error or "no_target_quote",
                }
        done += len(batch)
        if batch_no % max(1, args.progress_every // max(1, args.quote_batch_size)) == 0 or done >= len(codes):
            values = list(statuses.values())
            ok = sum(1 for row in values if row["ok"])
            latest_616 = sum(1 for row in values if row["latest_date"] == args.target_date)
            print(f"[{_now_text()}] quote refresh progress {done}/{len(codes)}, ok={ok}, latest_{args.target_date}={latest_616}", flush=True)
        time.sleep(args.quote_sleep)
    return pd.DataFrame(list(statuses.values()))


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args", {})
    ctx_cols = checkpoint.get("context_cols")
    model = TemporalAttentionNet(
        len(checkpoint.get("seq_cols", SEQ_COLS)),
        len(ctx_cols),
        d_model=int(model_args.get("d_model", 96)),
        dropout=float(model_args.get("dropout", 0.18)),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def load_histories(items: List[StockItem], history_dir: Path) -> Dict[str, Tuple[StockItem, pd.DataFrame]]:
    out: Dict[str, Tuple[StockItem, pd.DataFrame]] = {}
    for item in items:
        path = history_dir / f"{item.code}.csv"
        if not path.exists():
            continue
        try:
            df = normalize_hist(pd.read_csv(path))
            if not df.empty:
                out[item.code] = (item, df)
        except Exception:
            continue
    return out


def parse_args() -> argparse.Namespace:
    today = datetime.now().strftime("%Y%m%d")
    begin = (datetime.now() - timedelta(days=25)).strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-file", default="data_cache/volume_contraction_manual_v4/universe.csv")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--model-path", default="model_cache/volume_contraction_breakout_5y_tcn.pt")
    parser.add_argument("--output-dir", default="data_cache/volume_contraction_screen_20260616")
    parser.add_argument("--begin", default=begin)
    parser.add_argument("--end", default=today)
    parser.add_argument("--target-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--exclude-chinext", action="store_true", default=True)
    parser.add_argument("--main-board-only", action="store_true", default=True)
    parser.add_argument("--include-non-main-board", dest="main_board_only", action="store_false")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--candidate-min-amount", type=float, default=50_000_000)
    parser.add_argument("--skip-current", action="store_true", default=True)
    parser.add_argument("--quote-only", action="store_true", default=False)
    parser.add_argument("--quote-batch-size", type=int, default=60)
    parser.add_argument("--quote-sleep", type=float, default=0.15)
    parser.add_argument("--eastmoney-only", action="store_true", default=False)
    parser.add_argument("--refresh-only", action="store_true", default=False)
    parser.add_argument("--require-entry-close-retest", action="store_true", default=False)
    parser.add_argument("--disable-entry-close-breakout-signal", action="store_true", default=False)
    parser.add_argument("--entry-close-breakout-recent-days", type=int, default=12)
    parser.add_argument("--entry-close-breakout-min-body", type=float, default=0.006)
    parser.add_argument("--entry-close-breakout-min-ret", type=float, default=0.0)
    parser.add_argument("--entry-close-breakout-close-buffer", type=float, default=0.0)
    parser.add_argument("--entry-close-breakout-prior-tolerance", type=float, default=0.01)
    parser.add_argument("--entry-close-breakout-max-trigger-volume-vs-entry", type=float, default=0.95)
    parser.add_argument("--entry-close-breakout-max-trigger-amount-vs-entry", type=float, default=1.20)
    parser.add_argument("--entry-close-breakout-max-volume10-vs-entry", type=float, default=0.62)
    parser.add_argument("--entry-close-breakout-max-latest-volume-vs-entry", type=float, default=0.70)
    parser.add_argument("--entry-close-breakout-max-amount10-vs-entry", type=float, default=0.72)
    parser.add_argument("--entry-close-breakout-max-pre-breakout-ret", type=float, default=0.55)
    parser.add_argument("--entry-close-breakout-max-ret-since-entry", type=float, default=0.75)
    parser.add_argument("--entry-close-breakout-require-macd-to-latest", action="store_true", default=True)
    parser.add_argument("--disable-macd-water-filter", action="store_true", default=False)
    parser.add_argument("--macd-waterline-min", type=float, default=0.0)
    parser.add_argument("--macd-require-dea-above-water", action="store_true", default=True)
    parser.add_argument("--disable-breakout-hold-signal", action="store_true", default=False)
    parser.add_argument("--breakout-hold-lookback-days", type=int, default=45)
    parser.add_argument("--breakout-hold-min-days", type=int, default=1)
    parser.add_argument("--snapshot-top-n", type=int, default=80)
    parser.add_argument("--prediction-log-dir", default="data_cache/daily_uptrend_predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_universe(
        Path(args.universe_file),
        exclude_chinext=args.exclude_chinext,
        main_board_only=args.main_board_only,
    )
    print(f"[{_now_text()}] universe={len(items)}, begin={args.begin}, end={args.end}", flush=True)

    statuses = refresh_histories_by_quote(items, args) if args.quote_only else refresh_histories(items, args)
    statuses.to_csv(out_dir / "refresh_status.csv", index=False, encoding="utf-8-sig")
    if args.refresh_only:
        latest_counts = statuses["latest_date"].value_counts().head(10).to_dict() if not statuses.empty else {}
        summary = {
            "run_time": datetime.now().isoformat(),
            "universe": len(items),
            "refresh_ok": int(statuses["ok"].sum()) if not statuses.empty else 0,
            "target_date": args.target_date,
            "target_date_count": int((statuses["latest_date"] == args.target_date).sum()) if not statuses.empty else 0,
            "latest_date_counts": latest_counts,
            "history_dir": str(args.history_dir),
            "output_dir": str(out_dir),
            "eastmoney_only": bool(args.eastmoney_only),
        }
        (out_dir / "refresh_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    model, checkpoint = load_checkpoint(Path(args.model_path))
    model_args = argparse.Namespace(**checkpoint["args"])
    model_args.candidate_min_amount = args.candidate_min_amount
    model_args.progress_every = args.progress_every
    model_args.require_entry_close_retest = args.require_entry_close_retest
    model_args.main_board_only = args.main_board_only
    model_args.disable_entry_close_breakout_signal = args.disable_entry_close_breakout_signal
    model_args.entry_close_breakout_recent_days = args.entry_close_breakout_recent_days
    model_args.entry_close_breakout_min_body = args.entry_close_breakout_min_body
    model_args.entry_close_breakout_min_ret = args.entry_close_breakout_min_ret
    model_args.entry_close_breakout_close_buffer = args.entry_close_breakout_close_buffer
    model_args.entry_close_breakout_prior_tolerance = args.entry_close_breakout_prior_tolerance
    model_args.entry_close_breakout_max_trigger_volume_vs_entry = args.entry_close_breakout_max_trigger_volume_vs_entry
    model_args.entry_close_breakout_max_trigger_amount_vs_entry = args.entry_close_breakout_max_trigger_amount_vs_entry
    model_args.entry_close_breakout_max_volume10_vs_entry = args.entry_close_breakout_max_volume10_vs_entry
    model_args.entry_close_breakout_max_latest_volume_vs_entry = args.entry_close_breakout_max_latest_volume_vs_entry
    model_args.entry_close_breakout_max_amount10_vs_entry = args.entry_close_breakout_max_amount10_vs_entry
    model_args.entry_close_breakout_max_pre_breakout_ret = args.entry_close_breakout_max_pre_breakout_ret
    model_args.entry_close_breakout_max_ret_since_entry = args.entry_close_breakout_max_ret_since_entry
    model_args.entry_close_breakout_require_macd_to_latest = args.entry_close_breakout_require_macd_to_latest
    model_args.disable_macd_water_filter = args.disable_macd_water_filter
    model_args.macd_waterline_min = args.macd_waterline_min
    model_args.macd_require_dea_above_water = args.macd_require_dea_above_water
    model_args.enable_breakout_hold_signal = not args.disable_breakout_hold_signal
    model_args.breakout_hold_lookback_days = args.breakout_hold_lookback_days
    model_args.breakout_hold_min_days = args.breakout_hold_min_days
    histories = load_histories(items, Path(args.history_dir))
    min_date = pd.Timestamp(datetime.now() - timedelta(days=int(getattr(model_args, "strategy_days", 1825))))
    norm = (
        np.asarray(checkpoint["seq_mean"]),
        np.asarray(checkpoint["seq_std"]),
        np.asarray(checkpoint["ctx_mean"]).reshape(1, -1),
        np.asarray(checkpoint["ctx_std"]).reshape(1, -1),
    )
    print(f"[{_now_text()}] scoring histories={len(histories)}", flush=True)
    candidates = score_current(histories, model, norm, model_args, min_date)
    candidates.to_csv(out_dir / "contraction_current_candidates.csv", index=False, encoding="utf-8-sig")
    snapshot_path = ""
    if args.snapshot_top_n > 0 and not candidates.empty:
        log_dir = Path(args.prediction_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        snapshot = candidates.head(args.snapshot_top_n).copy()
        snapshot.insert(0, "prediction_date", args.target_date)
        snapshot.insert(1, "snapshot_created_at", datetime.now().isoformat(timespec="seconds"))
        snapshot["source_output_dir"] = str(out_dir)
        snapshot["require_entry_close_retest"] = bool(args.require_entry_close_retest)
        snapshot["breakout_hold_signal_enabled"] = bool(model_args.enable_breakout_hold_signal)
        snapshot_path = str(log_dir / f"predictions_{args.target_date.replace('-', '')}.csv")
        snapshot.to_csv(snapshot_path, index=False, encoding="utf-8-sig")

    latest_counts = statuses["latest_date"].value_counts().head(10).to_dict() if not statuses.empty else {}
    summary = {
        "run_time": datetime.now().isoformat(),
        "universe": len(items),
        "refresh_ok": int(statuses["ok"].sum()) if not statuses.empty else 0,
        "target_date": args.target_date,
        "target_date_count": int((statuses["latest_date"] == args.target_date).sum()) if not statuses.empty else 0,
        "latest_date_counts": latest_counts,
        "histories": len(histories),
        "candidate_rows": int(len(candidates)),
        "require_entry_close_retest": bool(args.require_entry_close_retest),
        "main_board_only": bool(args.main_board_only),
        "entry_close_breakout_signal_enabled": not bool(args.disable_entry_close_breakout_signal),
        "macd_water_filter_enabled": not bool(args.disable_macd_water_filter),
        "breakout_hold_signal_enabled": bool(model_args.enable_breakout_hold_signal),
        "prediction_snapshot": snapshot_path,
        "model_path": str(args.model_path),
        "output_dir": str(out_dir),
    }
    (out_dir / "screen_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if candidates.empty:
        print("(empty)")
    else:
        cols = [
            "rank",
            "signal_type",
            "code",
            "name",
            "latest_date",
            "entry_date",
            "trigger_date",
            "entry_age",
            "trigger_age",
            "pct_chg",
            "amount_text",
            "entry_volume_ratio_20",
            "trigger_volume_vs_entry",
            "volume_10_vs_entry",
            "amount_10_vs_entry",
            "ret_since_entry",
            "entry_close_breakout_gap",
            "macd_water_ok",
            "max_ret_since_entry",
            "post_entry_min_ret",
            "entry_close_retest_gap",
            "entry_close_retest_signal",
            "breakout_hold_signal",
            "pullback_low_date",
            "pullback_low_vs_entry_low",
            "breakout_ret_since_entry",
            "model_prob",
            "rule_score",
            "final_score",
            "reason",
        ]
        cols = [col for col in cols if col in candidates.columns]
        print(candidates[cols].head(80).to_string(index=False))


if __name__ == "__main__":
    main()
