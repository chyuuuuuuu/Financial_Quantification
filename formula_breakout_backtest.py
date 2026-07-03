#!/usr/bin/env python
"""One-year backtest for the formula breakout selector."""

from __future__ import annotations

import argparse
import builtins
import json
import math
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from formula_breakout_pipeline import (
    clean_records,
    load_universe,
    macd,
    normalize_code,
    now_text,
)
from main_uptrend_model import normalize_hist

print = partial(builtins.print, flush=True)


HORIZONS = (1, 3, 5)


def round_or_none(value: object, digits: int = 4) -> Optional[float]:
    try:
        out = float(value)
        if math.isfinite(out):
            return round(out, digits)
    except Exception:
        pass
    return None


def infer_latest_date(universe: pd.DataFrame, history_dir: Path) -> str:
    latest: Optional[pd.Timestamp] = None
    for code in universe["code"]:
        path = history_dir / f"{code}.csv"
        if not path.exists():
            continue
        try:
            dates = pd.read_csv(path, usecols=["日期"])
        except Exception:
            continue
        if dates.empty:
            continue
        value = pd.to_datetime(dates["日期"], errors="coerce").dropna()
        if value.empty:
            continue
        item = pd.Timestamp(value.iloc[-1]).normalize()
        latest = item if latest is None or item > latest else latest
    if latest is None:
        raise ValueError(f"no history dates found under {history_dir}")
    return latest.strftime("%Y-%m-%d")


def resolve_dates(args: argparse.Namespace, universe: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    history_dir = Path(args.history_dir)
    end = pd.Timestamp(args.end_date or infer_latest_date(universe, history_dir)).normalize()
    if args.start_date:
        start = pd.Timestamp(args.start_date).normalize()
    else:
        start = (end - pd.DateOffset(years=1) + pd.Timedelta(days=1)).normalize()
    if start > end:
        raise ValueError("--start-date cannot be after --end-date")
    return start, end


def segment_state(close: np.ndarray, ma20: np.ndarray, triple_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_pre_close = np.full(len(close), np.nan, dtype=float)
    washed = np.zeros(len(close), dtype=bool)
    triple_indices = np.flatnonzero(triple_mask)
    for pos, start in enumerate(triple_indices):
        stop = int(triple_indices[pos + 1]) if pos + 1 < len(triple_indices) else len(close)
        running_max = float(close[start])
        seen_wash = False
        for idx in range(start + 1, stop):
            max_pre_close[idx] = running_max
            washed[idx] = seen_wash
            if np.isfinite(ma20[idx]) and close[idx] < ma20[idx]:
                seen_wash = True
            if close[idx] > running_max:
                running_max = float(close[idx])
    return max_pre_close, washed


def future_metrics(
    idx: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    date_text: Sequence[str],
    horizons: Iterable[int],
) -> Dict[str, object]:
    base = float(close[idx])
    out: Dict[str, object] = {}
    if base <= 0:
        return out
    if idx + 1 < len(close):
        out["next_date"] = date_text[idx + 1]
        out["next_ret_pct"] = round((float(close[idx + 1]) / base - 1.0) * 100.0, 4)
        out["next_high_pct"] = round((float(high[idx + 1]) / base - 1.0) * 100.0, 4)
        out["next_low_pct"] = round((float(low[idx + 1]) / base - 1.0) * 100.0, 4)
    for horizon in horizons:
        if idx + horizon < len(close):
            window = slice(idx + 1, idx + horizon + 1)
            out[f"ret_{horizon}d_pct"] = round((float(close[idx + horizon]) / base - 1.0) * 100.0, 4)
            out[f"max_{horizon}d_pct"] = round((float(np.nanmax(high[window])) / base - 1.0) * 100.0, 4)
            out[f"min_{horizon}d_pct"] = round((float(np.nanmin(low[window])) / base - 1.0) * 100.0, 4)
    return out


def evaluate_history(
    code: str,
    name: str,
    hist: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[List[Dict[str, object]], List[str]]:
    hist = normalize_hist(hist)
    if hist.empty or len(hist) < 35:
        return [], []
    hist = hist.copy()
    hist["日期_dt"] = pd.to_datetime(hist["日期"], errors="coerce").dt.normalize()
    hist = hist.dropna(subset=["日期_dt"]).sort_values("日期_dt").drop_duplicates("日期_dt", keep="last")
    if hist.empty:
        return [], []

    date_series = hist["日期_dt"]
    in_window = (date_series >= start) & (date_series <= end)
    market_dates = date_series.loc[in_window].dt.strftime("%Y-%m-%d").tolist()
    if not market_dates:
        return [], []

    open_ = pd.to_numeric(hist["开盘"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(hist["收盘"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(hist["最高"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(hist["最低"], errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(hist["成交量"], errors="coerce").fillna(0).to_numpy(dtype=float)
    amount = pd.to_numeric(hist["成交额"], errors="coerce").fillna(0).to_numpy(dtype=float)
    pct_chg = pd.to_numeric(hist["涨跌幅"], errors="coerce").fillna(0).to_numpy(dtype=float)
    date_text = date_series.dt.strftime("%Y-%m-%d").tolist()

    valid_price = np.isfinite(open_) & np.isfinite(close) & np.isfinite(high) & np.isfinite(low)
    prev_volume = np.r_[np.nan, volume[:-1]]
    triple_mask = (volume > 3.0 * prev_volume) & (close > open_) & valid_price
    row_idx = np.arange(len(hist))
    last_triple = np.maximum.accumulate(np.where(triple_mask, row_idx, -1))
    bars_since = row_idx - last_triple
    has_triple = last_triple >= 0
    clipped_last = np.maximum(last_triple, 0)

    triple_close = close[clipped_last]
    triple_volume = volume[clipped_last]
    prev_at_triple = np.full(len(hist), np.nan, dtype=float)
    prev_ok = last_triple > 0
    prev_at_triple[prev_ok] = volume[last_triple[prev_ok] - 1]
    volume_vs_prev = np.divide(triple_volume, prev_at_triple, out=np.zeros(len(hist), dtype=float), where=prev_at_triple > 0)

    vol_ma5 = pd.Series(volume).rolling(5, min_periods=5).mean().to_numpy(dtype=float)
    vol_ma10 = pd.Series(volume).rolling(10, min_periods=10).mean().to_numpy(dtype=float)
    ma20 = pd.Series(close).rolling(20, min_periods=20).mean().to_numpy(dtype=float)
    dif, dea = macd(pd.Series(close))
    dif_arr = dif.to_numpy(dtype=float)
    dea_arr = dea.to_numpy(dtype=float)
    prior_close = np.r_[np.nan, close[:-1]]
    prior_dif = np.r_[np.nan, dif_arr[:-1]]
    prior_dea = np.r_[np.nan, dea_arr[:-1]]

    max_pre_close, washed = segment_state(close, ma20, triple_mask)
    vol_ma5_vs_triple = np.divide(vol_ma5, triple_volume, out=np.full(len(hist), np.nan), where=triple_volume > 0)
    vol_ma10_vs_triple = np.divide(vol_ma10, triple_volume, out=np.full(len(hist), np.nan), where=triple_volume > 0)
    breakout_gap = (close / triple_close - 1.0) * 100.0
    prior_gap = (prior_close / triple_close - 1.0) * 100.0
    max_close_vs_triple = (max_pre_close / triple_close - 1.0) * 100.0

    target_mask = in_window.to_numpy(dtype=bool)
    contraction = (vol_ma5 < triple_volume) & (vol_ma10 < triple_volume)
    not_too_hot = max_pre_close < triple_close * 1.15
    macd_strong = (dif_arr > dea_arr) & (dif_arr > prior_dif) & (dea_arr > prior_dea)
    breakout = (prior_close < triple_close) & (close >= triple_close) & (close > open_)
    signal_mask = (
        target_mask
        & has_triple
        & (bars_since >= 2)
        & (bars_since <= 120)
        & valid_price
        & (triple_close > 0)
        & (triple_volume > 0)
        & contraction
        & not_too_hot
        & washed
        & macd_strong
        & breakout
    )

    signal_indices = np.flatnonzero(signal_mask)
    rows: List[Dict[str, object]] = []
    for idx in signal_indices:
        contraction_score = max(0.0, min(25.0, (1.0 - min(float(vol_ma10_vs_triple[idx]), 1.0)) * 35.0))
        triple_score = max(0.0, min(15.0, (float(volume_vs_prev[idx]) - 3.0) * 4.0))
        timing_score = max(0.0, min(12.0, 12.0 * (1.0 - (float(bars_since[idx]) - 2.0) / 118.0)))
        macd_score = max(0.0, min(12.0, (float(dif_arr[idx]) - float(dea_arr[idx])) / float(triple_close[idx]) * 1200.0))
        breakout_score = max(0.0, min(10.0, float(breakout_gap[idx]) * 2.0))
        heat_penalty = max(0.0, min(8.0, float(max_close_vs_triple[idx]) - 8.0))
        base_score = 45.0
        score = round(base_score + contraction_score + triple_score + timing_score + macd_score + breakout_score - heat_penalty, 3)
        triple_idx = int(last_triple[idx])
        row: Dict[str, object] = {
            "snapshot_date": date_text[idx],
            "code": normalize_code(code),
            "name": name,
            "latest_date": date_text[idx],
            "close": round(float(close[idx]), 4),
            "open": round(float(open_[idx]), 4),
            "high": round(float(high[idx]), 4),
            "low": round(float(low[idx]), 4),
            "pct_chg": round(float(pct_chg[idx]), 4),
            "amount": round(float(amount[idx]), 2),
            "volume": round(float(volume[idx]), 2),
            "formula_score": score,
            "base_score": round(base_score, 3),
            "contraction_score": round(contraction_score, 3),
            "triple_score": round(triple_score, 3),
            "timing_score": round(timing_score, 3),
            "macd_score": round(macd_score, 3),
            "breakout_score": round(breakout_score, 3),
            "heat_penalty": round(heat_penalty, 3),
            "triple_date": date_text[triple_idx],
            "triple_close": round(float(triple_close[idx]), 4),
            "triple_volume": round(float(triple_volume[idx]), 2),
            "bars_since_triple": int(bars_since[idx]),
            "volume_vs_prev": round(float(volume_vs_prev[idx]), 4),
            "vol_ma5_vs_triple": round(float(vol_ma5_vs_triple[idx]), 4),
            "vol_ma10_vs_triple": round(float(vol_ma10_vs_triple[idx]), 4),
            "max_close_vs_triple": round(float(max_close_vs_triple[idx]), 4),
            "breakout_gap_pct": round(float(breakout_gap[idx]), 4),
            "prior_close_gap_pct": round(float(prior_gap[idx]), 4),
            "washed_below_ma20": True,
            "macd_dif": round(float(dif_arr[idx]), 6),
            "macd_dea": round(float(dea_arr[idx]), 6),
            "reason": (
                f"三倍阳{int(bars_since[idx])}日后首次收盘突破；"
                f"MA5/MA10量能={float(vol_ma5_vs_triple[idx]):.2f}/{float(vol_ma10_vs_triple[idx]):.2f}倍；MACD开口上行"
            ),
        }
        row.update(future_metrics(idx, close, high, low, date_text, HORIZONS))
        rows.append(row)
    return rows, market_dates


def rank_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    out = signals.sort_values(["snapshot_date", "formula_score", "amount"], ascending=[True, False, False]).copy()
    out["rank"] = out.groupby("snapshot_date").cumcount() + 1
    cols = ["rank"] + [col for col in out.columns if col != "rank"]
    return out[cols].reset_index(drop=True)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return round(float(drawdown.min() * 100.0), 4)


def build_portfolio(
    signals: pd.DataFrame,
    market_dates: Sequence[str],
    max_positions: int,
    label: str,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        empty_daily = pd.DataFrame({"snapshot_date": list(market_dates), "portfolio_ret_pct": 0.0, "equity": 1.0})
        return {
            "label": label,
            "max_positions": max_positions or None,
            "signal_days": 0,
            "measured_signal_days": 0,
            "total_signals": 0,
            "measured_signals": 0,
            "avg_candidates_per_signal_day": None,
            "max_candidates": 0,
            "avg_next_ret_pct": None,
            "win_rate_pct": None,
            "compound_ret_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }, empty_daily, pd.DataFrame()

    work = signals.copy()
    if max_positions > 0:
        work = work[work["rank"] <= max_positions].copy()
    signal_counts = work.groupby("snapshot_date", as_index=False).agg(selected_count=("code", "count"))
    trade = work[pd.to_numeric(work["next_ret_pct"], errors="coerce").notna()].copy()
    for col in ["next_ret_pct", "ret_3d_pct", "ret_5d_pct", "next_high_pct", "next_low_pct", "formula_score"]:
        if col in trade:
            trade[col] = pd.to_numeric(trade[col], errors="coerce")

    if trade.empty:
        daily_stats = pd.DataFrame(columns=["snapshot_date", "traded_count", "avg_next_ret_pct"])
    else:
        daily_stats = trade.groupby("snapshot_date", as_index=False).agg(
            traded_count=("code", "count"),
            avg_formula_score=("formula_score", "mean"),
            avg_next_ret_pct=("next_ret_pct", "mean"),
            median_next_ret_pct=("next_ret_pct", "median"),
            win_rate_pct=("next_ret_pct", lambda x: (x > 0).mean() * 100.0),
            avg_next_high_pct=("next_high_pct", "mean"),
            avg_next_low_pct=("next_low_pct", "mean"),
            avg_3d_ret_pct=("ret_3d_pct", "mean"),
            avg_5d_ret_pct=("ret_5d_pct", "mean"),
        )

    calendar = pd.DataFrame({"snapshot_date": list(market_dates)})
    daily = calendar.merge(signal_counts, on="snapshot_date", how="left").merge(daily_stats, on="snapshot_date", how="left")
    daily["selected_count"] = daily["selected_count"].fillna(0).astype(int)
    daily["traded_count"] = daily["traded_count"].fillna(0).astype(int)
    daily["portfolio_ret_pct"] = daily["avg_next_ret_pct"].fillna(0.0)
    daily["equity"] = (1.0 + daily["portfolio_ret_pct"] / 100.0).cumprod()
    daily["month"] = daily["snapshot_date"].str.slice(0, 7)

    monthly = daily.groupby("month", as_index=False).agg(
        trading_days=("snapshot_date", "count"),
        signal_days=("selected_count", lambda x: int((x > 0).sum())),
        measured_signal_days=("traded_count", lambda x: int((x > 0).sum())),
        total_signals=("selected_count", "sum"),
        avg_daily_ret_pct=("portfolio_ret_pct", "mean"),
        win_days_pct=("portfolio_ret_pct", lambda x: float((x > 0).mean() * 100.0)),
    )
    monthly["compound_ret_pct"] = daily.groupby("month")["portfolio_ret_pct"].apply(lambda x: ((1.0 + x / 100.0).prod() - 1.0) * 100.0).values

    ret = pd.to_numeric(trade["next_ret_pct"], errors="coerce") if not trade.empty else pd.Series(dtype=float)
    pos_sum = float(ret[ret > 0].sum()) if not ret.empty else 0.0
    neg_sum = float(ret[ret < 0].sum()) if not ret.empty else 0.0
    trading_days = int(len(daily))
    final_equity = float(daily["equity"].iloc[-1]) if not daily.empty else 1.0
    annual_ret = (final_equity ** (244.0 / trading_days) - 1.0) * 100.0 if trading_days > 0 and final_equity > 0 else None
    daily_ret = pd.to_numeric(daily["portfolio_ret_pct"], errors="coerce").fillna(0.0)
    sharpe = None
    if len(daily_ret) > 1 and float(daily_ret.std(ddof=1)) > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std(ddof=1) * math.sqrt(244.0))

    summary = {
        "label": label,
        "max_positions": max_positions or None,
        "trading_days": trading_days,
        "signal_days": int((daily["selected_count"] > 0).sum()),
        "measured_signal_days": int((daily["traded_count"] > 0).sum()),
        "total_signals": int(daily["selected_count"].sum()),
        "measured_signals": int(len(trade)),
        "unmeasured_signals": int(daily["selected_count"].sum() - len(trade)),
        "avg_candidates_per_signal_day": round_or_none(signal_counts["selected_count"].mean() if not signal_counts.empty else None, 3),
        "max_candidates": int(signal_counts["selected_count"].max()) if not signal_counts.empty else 0,
        "avg_next_ret_pct": round_or_none(ret.mean() if not ret.empty else None, 4),
        "median_next_ret_pct": round_or_none(ret.median() if not ret.empty else None, 4),
        "win_rate_pct": round_or_none((ret > 0).mean() * 100.0 if not ret.empty else None, 2),
        "avg_3d_ret_pct": round_or_none(pd.to_numeric(trade.get("ret_3d_pct"), errors="coerce").mean() if not trade.empty else None, 4),
        "avg_5d_ret_pct": round_or_none(pd.to_numeric(trade.get("ret_5d_pct"), errors="coerce").mean() if not trade.empty else None, 4),
        "compound_ret_pct": round_or_none((final_equity - 1.0) * 100.0, 4),
        "annualized_ret_pct": round_or_none(annual_ret, 4),
        "max_drawdown_pct": max_drawdown(daily["equity"]),
        "sharpe_like": round_or_none(sharpe, 4),
        "best_daily_ret_pct": round_or_none(daily_ret.max() if not daily_ret.empty else None, 4),
        "worst_daily_ret_pct": round_or_none(daily_ret.min() if not daily_ret.empty else None, 4),
        "profit_factor": round_or_none(pos_sum / abs(neg_sum) if neg_sum < 0 else None, 4),
    }
    numeric_cols = [col for col in daily.columns if col.endswith("_pct") or col in {"equity", "avg_formula_score"}]
    for col in numeric_cols:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").round(4)
    for col in [col for col in monthly.columns if col.endswith("_pct")]:
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce").round(4)
    return summary, daily, monthly


def build_report(args: argparse.Namespace) -> Dict[str, object]:
    universe = load_universe(Path(args.universe_file))
    start, end = resolve_dates(args, universe)
    histories_dir = Path(args.history_dir)
    rows: List[Dict[str, object]] = []
    market_dates: set[str] = set()
    for idx, item in enumerate(universe.itertuples(index=False), start=1):
        path = histories_dir / f"{item.code}.csv"
        if not path.exists():
            continue
        try:
            hist = pd.read_csv(path)
        except Exception:
            continue
        stock_rows, stock_dates = evaluate_history(item.code, item.name, hist, start, end)
        rows.extend(stock_rows)
        market_dates.update(stock_dates)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"[{now_text()}] backtest {idx}/{len(universe)}, signals={len(rows)}")

    signals = rank_signals(pd.DataFrame(rows))
    dates = sorted(market_dates)
    all_summary, all_daily, all_monthly = build_portfolio(signals, dates, 0, "all_formula_signals")
    top_summary, top_daily, top_monthly = build_portfolio(signals, dates, args.top_n, f"top{args.top_n}")
    signals_display = signals.sort_values(["snapshot_date", "rank"], ascending=[False, True]).head(args.signal_display_limit) if not signals.empty else pd.DataFrame()

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "port": "formula_breakout",
        "strategy_name": "三倍阳缩量回调突破",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "universe_count": int(len(universe)),
        "history_file_count": int(sum(1 for p in histories_dir.glob("*.csv"))),
        "assumption": "信号按收盘后确认；回测收益采用信号日收盘价到后续交易日收盘价，组合按每日入选等权。",
        "rules": {
            "main_non_st": "00/60 主板，剔除 ST/*ST",
            "triple_yang": "V > 3 * REF(V,1) 且 C > O",
            "window": "BARSLAST(三倍阳) 在 2 到 120 日",
            "contraction": "MA(V,5) 与 MA(V,10) 均小于三倍阳成交量",
            "wash": "三倍阳至昨日最高收盘小于三倍阳收盘 1.15 倍，且期间至少一次收盘跌破 MA20",
            "macd": "DIF > DEA，DIF/DEA 同步上行",
            "breakout": "昨日收盘低于三倍阳收盘，今日阳线收盘突破",
        },
        "all_summary": all_summary,
        "top_summary": top_summary,
        "all_daily": clean_records(all_daily),
        f"top{args.top_n}_daily": clean_records(top_daily),
        "all_monthly": clean_records(all_monthly),
        f"top{args.top_n}_monthly": clean_records(top_monthly),
        "recent_signals": clean_records(signals_display),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[{now_text()}] wrote {output}, signals={len(signals)}, days={len(dates)}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--universe-file", default="data_cache/volume_contraction_screen_20260701_mainboard_entry_close/refresh_status.csv")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--output", default="static/reports/formula_breakout_backtest_1y.json")
    parser.add_argument("--csv-output", default="data_cache/formula_breakout_backtests/formula_breakout_backtest_1y_signals.csv")
    parser.add_argument("--signal-display-limit", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    build_report(parse_args())


if __name__ == "__main__":
    main()
