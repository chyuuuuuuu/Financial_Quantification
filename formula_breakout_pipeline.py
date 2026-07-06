#!/usr/bin/env python
"""Daily selector for the supplied TDX triple-volume breakout formula."""

from __future__ import annotations

import argparse
import builtins
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from main_uptrend_model import normalize_hist

print = partial(builtins.print, flush=True)


BASE_COLUMNS = [
    "rank",
    "code",
    "name",
    "latest_date",
    "close",
    "open",
    "high",
    "low",
    "pct_chg",
    "amount",
    "volume",
    "formula_score",
    "base_score",
    "contraction_score",
    "triple_score",
    "timing_score",
    "macd_score",
    "breakout_score",
    "heat_penalty",
    "triple_date",
    "triple_close",
    "triple_volume",
    "bars_since_triple",
    "volume_vs_prev",
    "vol_ma5_vs_triple",
    "vol_ma10_vs_triple",
    "max_close_vs_triple",
    "breakout_gap_pct",
    "prior_close_gap_pct",
    "washed_below_ma20",
    "macd_dif",
    "macd_dea",
    "reason",
]


def normalize_code(value: object) -> str:
    return str(value).strip().zfill(6)[-6:]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def clean_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    if df.empty:
        return []
    return df.replace({np.nan: None}).to_dict(orient="records")


def ymd(run_time: datetime) -> str:
    return run_time.strftime("%Y-%m-%d")


def stamp(run_time: datetime) -> str:
    return run_time.strftime("%Y%m%d_%H%M")


def compact_date(value: object) -> str:
    try:
        return pd.Timestamp(str(value)[:10]).strftime("%Y%m%d")
    except Exception:
        return ""


def next_run_time(hour: int, minute: int, now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def load_universe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str, "股票代码": str})
    code_col = "code" if "code" in df.columns else "股票代码"
    name_col = "name" if "name" in df.columns else "股票名称" if "股票名称" in df.columns else None
    out = pd.DataFrame()
    out["code"] = df[code_col].map(normalize_code)
    out["name"] = df[name_col].astype(str) if name_col else ""
    main = out["code"].str.startswith(("00", "60"))
    non_st = ~out["name"].str.contains(r"\*?ST", case=False, na=False)
    return out[main & non_st].drop_duplicates("code").reset_index(drop=True)


def run_refresh(args: argparse.Namespace, run_time: datetime) -> None:
    output_dir = Path(args.run_dir) / f"{stamp(run_time)}_refresh"
    cmd = [
        sys.executable,
        "refresh_and_screen_uptrend.py",
        "--universe-file",
        args.universe_file,
        "--history-dir",
        args.history_dir,
        "--output-dir",
        str(output_dir),
        "--target-date",
        ymd(run_time),
        "--workers",
        str(args.workers),
        "--retry",
        str(args.retry),
        "--progress-every",
        str(args.progress_every),
        "--refresh-only",
        "--quote-only",
        "--quote-source",
        args.quote_source,
        "--no-skip-current",
    ]
    print(f"[{now_text()}] refresh current bars: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def macd(close: pd.Series) -> Tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False, min_periods=1).mean()
    return dif, dea


def evaluate_formula(
    code: str,
    name: str,
    hist: pd.DataFrame,
    target_date: str,
    require_target_date: bool,
    normalized: bool = False,
) -> Optional[Dict[str, object]]:
    if not normalized:
        hist = normalize_hist(hist)
    if hist.empty:
        return None
    target_ts = pd.Timestamp(target_date)
    hist = hist[pd.to_datetime(hist["日期"]).dt.normalize() <= target_ts.normalize()].copy()
    if len(hist) < 35:
        return None
    latest = pd.Timestamp(hist.iloc[-1]["日期"]).normalize()
    if require_target_date and latest != target_ts.normalize():
        return None

    open_ = hist["开盘"].astype(float).reset_index(drop=True)
    close = hist["收盘"].astype(float).reset_index(drop=True)
    high = hist["最高"].astype(float).reset_index(drop=True)
    low = hist["最低"].astype(float).reset_index(drop=True)
    volume = hist["成交量"].astype(float).fillna(0).reset_index(drop=True)
    amount = hist["成交额"].astype(float).fillna(0).reset_index(drop=True)
    pct_chg = hist["涨跌幅"].astype(float).fillna(0).reset_index(drop=True)
    date = pd.to_datetime(hist["日期"]).reset_index(drop=True)

    i = len(hist) - 1
    triple_yang = (volume > 3.0 * volume.shift(1)) & (close > open_)
    triple_idx = np.flatnonzero(triple_yang.fillna(False).to_numpy())
    if len(triple_idx) == 0:
        return None
    j = int(triple_idx[-1])
    n = i - j
    if n < 2 or n > 120:
        return None

    triple_close = safe_float(close.iloc[j])
    triple_volume = safe_float(volume.iloc[j])
    if triple_close <= 0 or triple_volume <= 0:
        return None

    vol_ma5 = safe_float(volume.rolling(5, min_periods=5).mean().iloc[i])
    vol_ma10 = safe_float(volume.rolling(10, min_periods=10).mean().iloc[i])
    overall_contraction = vol_ma5 < triple_volume and vol_ma10 < triple_volume
    if not overall_contraction:
        return None

    ma20 = close.rolling(20, min_periods=20).mean()
    pre_close_window = close.iloc[j:i]
    max_pre_close = safe_float(pre_close_window.max(), triple_close)
    not_too_hot = max_pre_close < triple_close * 1.15
    washed = bool((close.iloc[j + 1 : i] < ma20.iloc[j + 1 : i]).fillna(False).any())
    if not (not_too_hot and washed):
        return None

    dif, dea = macd(close)
    macd_strong = (
        safe_float(dif.iloc[i]) > safe_float(dea.iloc[i])
        and safe_float(dif.iloc[i]) > safe_float(dif.iloc[i - 1])
        and safe_float(dea.iloc[i]) > safe_float(dea.iloc[i - 1])
    )
    if not macd_strong:
        return None

    breakout = safe_float(close.iloc[i - 1]) < triple_close and safe_float(close.iloc[i]) >= triple_close and safe_float(close.iloc[i]) > safe_float(open_.iloc[i])
    if not breakout:
        return None

    volume_vs_prev = safe_float(volume.iloc[j] / volume.iloc[j - 1], 0.0) if j > 0 and safe_float(volume.iloc[j - 1]) > 0 else 0.0
    vol_ma5_vs_triple = vol_ma5 / triple_volume
    vol_ma10_vs_triple = vol_ma10 / triple_volume
    breakout_gap = (safe_float(close.iloc[i]) / triple_close - 1.0) * 100.0
    prior_gap = (safe_float(close.iloc[i - 1]) / triple_close - 1.0) * 100.0
    max_close_vs_triple = (max_pre_close / triple_close - 1.0) * 100.0

    contraction_score = max(0.0, min(25.0, (1.0 - min(vol_ma10_vs_triple, 1.0)) * 35.0))
    triple_score = max(0.0, min(15.0, (volume_vs_prev - 3.0) * 4.0))
    timing_score = max(0.0, min(12.0, 12.0 * (1.0 - (n - 2) / 118.0)))
    macd_score = max(0.0, min(12.0, (safe_float(dif.iloc[i]) - safe_float(dea.iloc[i])) / triple_close * 1200.0))
    breakout_score = max(0.0, min(10.0, breakout_gap * 2.0))
    heat_penalty = max(0.0, min(8.0, max_close_vs_triple - 8.0))
    base_score = 45.0
    score = round(base_score + contraction_score + triple_score + timing_score + macd_score + breakout_score - heat_penalty, 3)

    return {
        "code": code,
        "name": name,
        "latest_date": latest.strftime("%Y-%m-%d"),
        "close": round(safe_float(close.iloc[i]), 4),
        "open": round(safe_float(open_.iloc[i]), 4),
        "high": round(safe_float(high.iloc[i]), 4),
        "low": round(safe_float(low.iloc[i]), 4),
        "pct_chg": round(safe_float(pct_chg.iloc[i]), 4),
        "amount": round(safe_float(amount.iloc[i]), 2),
        "volume": round(safe_float(volume.iloc[i]), 2),
        "formula_score": score,
        "base_score": round(base_score, 3),
        "contraction_score": round(contraction_score, 3),
        "triple_score": round(triple_score, 3),
        "timing_score": round(timing_score, 3),
        "macd_score": round(macd_score, 3),
        "breakout_score": round(breakout_score, 3),
        "heat_penalty": round(heat_penalty, 3),
        "triple_date": pd.Timestamp(date.iloc[j]).strftime("%Y-%m-%d"),
        "triple_close": round(triple_close, 4),
        "triple_volume": round(triple_volume, 2),
        "bars_since_triple": int(n),
        "volume_vs_prev": round(volume_vs_prev, 4),
        "vol_ma5_vs_triple": round(vol_ma5_vs_triple, 4),
        "vol_ma10_vs_triple": round(vol_ma10_vs_triple, 4),
        "max_close_vs_triple": round(max_close_vs_triple, 4),
        "breakout_gap_pct": round(breakout_gap, 4),
        "prior_close_gap_pct": round(prior_gap, 4),
        "washed_below_ma20": True,
        "macd_dif": round(safe_float(dif.iloc[i]), 6),
        "macd_dea": round(safe_float(dea.iloc[i]), 6),
        "reason": f"三倍阳{n}日后首次收盘突破；MA5/MA10量能={vol_ma5_vs_triple:.2f}/{vol_ma10_vs_triple:.2f}倍；MACD开口上行",
    }


def load_histories(universe: pd.DataFrame, history_dir: Path) -> Dict[str, pd.DataFrame]:
    histories: Dict[str, pd.DataFrame] = {}
    for item in universe.itertuples(index=False):
        path = history_dir / f"{item.code}.csv"
        if not path.exists():
            continue
        try:
            hist = normalize_hist(pd.read_csv(path))
        except Exception:
            continue
        if not hist.empty:
            histories[item.code] = hist
    return histories


def screen_candidates(args: argparse.Namespace, target_date: str, histories: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    universe = load_universe(Path(args.universe_file))
    names = dict(zip(universe["code"], universe["name"]))
    rows: List[Dict[str, object]] = []
    for idx, code in enumerate(universe["code"], start=1):
        if histories is not None:
            hist = histories.get(code)
            if hist is None:
                continue
        else:
            path = Path(args.history_dir) / f"{code}.csv"
            if not path.exists():
                continue
            try:
                hist = pd.read_csv(path)
            except Exception:
                continue
        item = evaluate_formula(code, names.get(code, ""), hist, target_date, args.require_target_date, normalized=histories is not None)
        if item is not None:
            rows.append(item)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"[{now_text()}] formula screen {idx}/{len(universe)}, selected={len(rows)}")
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)
    df = df.sort_values(["formula_score", "amount"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def save_snapshot(candidates: pd.DataFrame, args: argparse.Namespace, run_time: datetime, run_dir: Path) -> pd.DataFrame:
    snapshots_dir = Path(args.snapshot_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = candidates.copy()
    if out.empty:
        out = pd.DataFrame(columns=BASE_COLUMNS)
    out.insert(0, "snapshot_time", run_time.isoformat(timespec="seconds"))
    out.insert(0, "snapshot_date", ymd(run_time))

    snapshot_path = snapshots_dir / f"candidates_{stamp(run_time)}.csv"
    latest_path = snapshots_dir / "latest_selected.csv"
    out.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
    out.to_csv(latest_path, index=False, encoding="utf-8-sig")
    out.to_csv(run_dir / "selected.csv", index=False, encoding="utf-8-sig")
    return out


def future_returns(row: pd.Series, history_dir: Path) -> Optional[Dict[str, object]]:
    code = normalize_code(row.get("code"))
    base = safe_float(row.get("close"), 0.0)
    if base <= 0:
        return None
    latest = pd.Timestamp(str(row.get("latest_date") or row.get("snapshot_date"))[:10])
    path = history_dir / f"{code}.csv"
    if not path.exists():
        return None
    try:
        hist = normalize_hist(pd.read_csv(path))
    except Exception:
        return None
    future = hist[pd.to_datetime(hist["日期"]).dt.normalize() > latest.normalize()].head(5)
    if future.empty:
        return None
    out: Dict[str, object] = {
        "next_date": pd.Timestamp(future.iloc[0]["日期"]).strftime("%Y-%m-%d"),
        "next_ret_pct": round((safe_float(future.iloc[0]["收盘"]) / base - 1.0) * 100.0, 4),
        "next_high_pct": round((safe_float(future.iloc[0]["最高"]) / base - 1.0) * 100.0, 4),
        "next_low_pct": round((safe_float(future.iloc[0]["最低"]) / base - 1.0) * 100.0, 4),
    }
    for horizon in (3, 5):
        if len(future) >= horizon:
            window = future.head(horizon)
            out[f"ret_{horizon}d_pct"] = round((safe_float(window.iloc[-1]["收盘"]) / base - 1.0) * 100.0, 4)
            out[f"max_{horizon}d_pct"] = round((safe_float(window["最高"].max()) / base - 1.0) * 100.0, 4)
            out[f"min_{horizon}d_pct"] = round((safe_float(window["最低"].min()) / base - 1.0) * 100.0, 4)
    return out


def update_feedback(args: argparse.Namespace) -> pd.DataFrame:
    snapshots_dir = Path(args.snapshot_dir)
    rows: List[Dict[str, object]] = []
    for path in sorted(snapshots_dir.glob("candidates_*.csv")):
        try:
            df = pd.read_csv(path, dtype={"code": str})
        except Exception:
            continue
        for item in df.to_dict(orient="records"):
            ret = future_returns(pd.Series(item), Path(args.history_dir))
            if ret is not None:
                rows.append({**item, **ret, "snapshot_file": path.name})
    feedback = pd.DataFrame(rows)
    feedback_path = Path(args.feedback_file)
    if not feedback.empty:
        feedback = feedback.drop_duplicates(["snapshot_time", "code"], keep="last")
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback.to_csv(feedback_path, index=False, encoding="utf-8-sig")
    return feedback


def daily_feedback(feedback: pd.DataFrame) -> List[Dict[str, object]]:
    if feedback.empty:
        return []
    df = feedback.copy()
    df["snapshot_date"] = df["snapshot_date"].astype(str)
    df["next_ret_pct"] = pd.to_numeric(df["next_ret_pct"], errors="coerce")
    grouped = df.groupby("snapshot_date", as_index=False).agg(
        rows=("code", "count"),
        avg_next_ret_pct=("next_ret_pct", "mean"),
        win_rate_pct=("next_ret_pct", lambda x: (x > 0).mean() * 100.0),
    )
    grouped["avg_next_ret_pct"] = grouped["avg_next_ret_pct"].round(4)
    grouped["win_rate_pct"] = grouped["win_rate_pct"].round(2)
    return clean_records(grouped.sort_values("snapshot_date"))


def refresh_static_index(static_dir: Path) -> None:
    reports_dir = static_dir / "reports"
    rows: List[Dict[str, object]] = []
    for path in sorted(reports_dir.glob("formula_breakout_*.json")):
        if path.name == "formula_breakout_index.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        snapshot_date = str(payload.get("snapshot_date") or "")
        if not snapshot_date:
            continue
        rows.append(
            {
                "date": snapshot_date,
                "file": path.name,
                "selected_count": int(payload.get("selected_count") or 0),
                "avg_formula_score": payload.get("avg_formula_score"),
                "generated_at": payload.get("generated_at"),
            }
        )
    rows = sorted({item["date"]: item for item in rows}.values(), key=lambda item: item["date"], reverse=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "formula_breakout_index.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), "dates": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_analysis(args: argparse.Namespace, run_time: datetime, selected: pd.DataFrame, feedback: pd.DataFrame) -> Dict[str, object]:
    recent_feedback = feedback.sort_values("snapshot_time").tail(args.feedback_window) if not feedback.empty else pd.DataFrame()
    selected_display = selected.head(args.display_limit).copy()
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_date": ymd(run_time),
        "snapshot_time": run_time.isoformat(timespec="seconds"),
        "port": "formula_breakout",
        "formula_name": "三倍阳缩量回调突破",
        "selected_count": int(len(selected)),
        "display_limit": int(args.display_limit),
        "avg_formula_score": round(float(pd.to_numeric(selected["formula_score"], errors="coerce").mean()), 4) if not selected.empty else None,
        "avg_breakout_gap_pct": round(float(pd.to_numeric(selected["breakout_gap_pct"], errors="coerce").mean()), 4) if not selected.empty else None,
        "selected_rows": clean_records(selected_display),
        "recent_feedback": {
            "rows": int(len(recent_feedback)),
            "avg_next_ret_pct": round(float(recent_feedback["next_ret_pct"].mean()), 4) if not recent_feedback.empty else None,
            "win_rate_pct": round(float((recent_feedback["next_ret_pct"] > 0).mean() * 100.0), 2) if not recent_feedback.empty else None,
        },
        "daily_feedback": daily_feedback(feedback),
        "feedback_rows": clean_records(recent_feedback.tail(args.feedback_display_limit)) if not recent_feedback.empty else [],
        "rules": {
            "main_non_st": "00/60 主板，剔除 ST/*ST",
            "triple_yang": "V > 3 * REF(V,1) 且 C > O",
            "window": "BARSLAST(三倍阳) 在 2 到 120 日",
            "contraction": "MA(V,5) 与 MA(V,10) 均小于三倍阳成交量",
            "wash": "三倍阳至昨日最高收盘小于三倍阳收盘 1.15 倍，且期间至少一次收盘跌破 MA20",
            "macd": "DIF > DEA，DIF/DEA 同步上行",
            "breakout": "昨日收盘低于三倍阳收盘，今日阳线收盘突破",
        },
    }


def write_static_report(analysis: Dict[str, object], args: argparse.Namespace, update_latest: bool = True) -> None:
    static_dir = Path(args.static_dir)
    reports_dir = static_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(analysis, ensure_ascii=False, indent=2)
    if update_latest:
        (reports_dir / "formula_breakout.json").write_text(payload, encoding="utf-8")
    date_key = compact_date(analysis.get("snapshot_date"))
    if date_key:
        (reports_dir / f"formula_breakout_{date_key}.json").write_text(payload, encoding="utf-8")
    refresh_static_index(static_dir)
    template = Path(args.template)
    if template.exists():
        html = template.read_text(encoding="utf-8")
        (static_dir / "formula-breakout.html").write_text(html, encoding="utf-8")


def run_once(args: argparse.Namespace, run_time: Optional[datetime] = None) -> Dict[str, object]:
    run_time = run_time or datetime.now().replace(second=0, microsecond=0)
    if args.refresh:
        run_refresh(args, run_time)
    run_dir = Path(args.run_dir) / stamp(run_time)
    selected = screen_candidates(args, ymd(run_time))
    selected = save_snapshot(selected, args, run_time, run_dir)
    feedback = update_feedback(args)
    analysis = build_analysis(args, run_time, selected, feedback)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.archive_only:
        Path(args.snapshot_dir, "latest_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    write_static_report(analysis, args, update_latest=not args.archive_only)
    print(f"[{now_text()}] formula selected={len(selected)} saved under {run_dir}")
    return analysis


def replay(args: argparse.Namespace) -> None:
    if not args.replay_start or not args.replay_end:
        raise ValueError("--replay-start and --replay-end are required")
    universe = load_universe(Path(args.universe_file))
    names = dict(zip(universe["code"], universe["name"]))
    start = pd.Timestamp(args.replay_start)
    end = pd.Timestamp(args.replay_end)
    dates = [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(start, end, freq="D")
        if d.weekday() < 5
    ]
    rows_by_date: Dict[str, List[Dict[str, object]]] = {d: [] for d in dates}
    print(f"[{now_text()}] replay dates={dates}")
    for idx, code in enumerate(universe["code"], start=1):
        if args.progress_every > 0 and idx % args.progress_every == 0:
            selected = sum(len(v) for v in rows_by_date.values())
            print(f"[{now_text()}] formula replay {idx}/{len(universe)}, selected_total={selected}")
        path = Path(args.history_dir) / f"{code}.csv"
        if not path.exists():
            continue
        try:
            hist = normalize_hist(pd.read_csv(path))
        except Exception:
            continue
        if hist.empty:
            continue
        for d in dates:
            item = evaluate_formula(code, names.get(code, ""), hist, d, True, normalized=True)
            if item is not None:
                rows_by_date[d].append(item)

    selected_by_date: Dict[str, pd.DataFrame] = {}
    for d in dates:
        target = datetime.strptime(d, "%Y-%m-%d").replace(hour=args.schedule_hour, minute=args.schedule_minute)
        selected = pd.DataFrame(rows_by_date[d])
        if selected.empty:
            selected = pd.DataFrame(columns=BASE_COLUMNS)
        else:
            selected = selected.sort_values(["formula_score", "amount"], ascending=False).reset_index(drop=True)
            selected.insert(0, "rank", np.arange(1, len(selected) + 1))
        selected_by_date[d] = save_snapshot(selected, args, target, Path(args.run_dir) / stamp(target))
        print(f"[{now_text()}] replay {d}: selected={len(selected)}")
    feedback = update_feedback(args)
    for d in dates:
        target = datetime.strptime(d, "%Y-%m-%d").replace(hour=args.schedule_hour, minute=args.schedule_minute)
        analysis = build_analysis(args, target, selected_by_date.get(d, pd.DataFrame(columns=BASE_COLUMNS)), feedback)
        update_latest = (not args.archive_only) and d == dates[-1]
        if update_latest:
            Path(args.snapshot_dir, "latest_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        write_static_report(analysis, args, update_latest=update_latest)


def daemon(args: argparse.Namespace) -> None:
    print(f"[{now_text()}] formula scheduler started, target={args.schedule_hour:02d}:{args.schedule_minute:02d}")
    while True:
        target = next_run_time(args.schedule_hour, args.schedule_minute)
        seconds = max(1, int((target - datetime.now()).total_seconds()))
        print(f"[{now_text()}] next formula run at {target:%Y-%m-%d %H:%M:%S}, sleep={seconds}s")
        time.sleep(seconds)
        try:
            run_once(args, target)
        except Exception as exc:
            print(f"[{now_text()}] formula run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            time.sleep(60)


def print_cron(args: argparse.Namespace) -> None:
    print(
        f"30 15 * * 1-5 cd {Path.cwd()} && {sys.executable} formula_breakout_pipeline.py --run-once "
        f"--target-date $(date +\\%F) >> data_cache/formula_breakout_snapshots/cron.log 2>&1"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--print-cron", action="store_true")
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--target-date", default="")
    parser.add_argument("--replay-start", default="")
    parser.add_argument("--replay-end", default="")
    parser.add_argument("--replay-min-coverage", type=int, default=500)
    parser.add_argument("--schedule-hour", type=int, default=15)
    parser.add_argument("--schedule-minute", type=int, default=30)
    parser.add_argument("--universe-file", default="data_cache/volume_contraction_screen_20260701_mainboard_entry_close/refresh_status.csv")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--run-dir", default="data_cache/formula_breakout_runs")
    parser.add_argument("--snapshot-dir", default="data_cache/formula_breakout_snapshots")
    parser.add_argument("--feedback-file", default="data_cache/formula_breakout_snapshots/feedback.csv")
    parser.add_argument("--static-dir", default="static")
    parser.add_argument("--template", default="templates/formula_breakout.html")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--quote-source", choices=["tencent", "pytdx", "auto"], default="auto")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=600)
    parser.add_argument("--require-target-date", action="store_true", default=True)
    parser.add_argument("--allow-stale-target", dest="require_target_date", action="store_false")
    parser.add_argument("--display-limit", type=int, default=200)
    parser.add_argument("--feedback-window", type=int, default=200)
    parser.add_argument("--feedback-display-limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_date:
        run_time = datetime.strptime(args.target_date, "%Y-%m-%d").replace(hour=args.schedule_hour, minute=args.schedule_minute)
    else:
        run_time = datetime.now().replace(second=0, microsecond=0)
    if args.print_cron:
        print_cron(args)
    elif args.replay:
        replay(args)
    elif args.daemon:
        daemon(args)
    elif args.run_once:
        run_once(args, run_time)
    else:
        raise SystemExit("choose one of --run-once, --replay, --daemon, --print-cron")


if __name__ == "__main__":
    main()
