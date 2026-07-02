#!/usr/bin/env python
"""Five-year contraction-before-main-run model.

Pattern:
1. In the last five years, a stock prints a 3x-volume bullish candle.
2. After that candle, volume and amount contract for a while.
3. Before any obvious main run, contracted quiet days are used as positive
   samples if a main run appears later in the observation window.
4. A main run starts with a fast price surge or a second heavy-volume bullish
   candle, and then reaches a configurable multiple within about one trading
   month from that start day.
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from main_uptrend_model import (
    FEATURE_COLS,
    StockItem,
    _format_amount,
    _now_text,
    add_features,
    is_hs_main_board_code,
    load_spot_universe,
    make_universe,
)
from latent_uptrend_model import TemporalAttentionNet, choose_device, ensure_torch, load_histories

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except ImportError as exc:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

print = partial(builtins.print, flush=True)


SEQ_COLS = FEATURE_COLS
SHORT_CONTEXT_COLS = [
    "entry_age",
    "entry_day_ret",
    "entry_body_pct",
    "entry_volume_ratio_20",
    "entry_amount_ratio_20",
    "entry_volume_ratio_20_prev",
    "entry_amount_ratio_20_prev",
    "entry_turnover",
    "entry_close_to_high_60",
    "volume_5_vs_entry",
    "volume_10_vs_entry",
    "amount_5_vs_entry",
    "amount_10_vs_entry",
    "latest_volume_vs_entry",
    "latest_amount_vs_entry",
    "contraction_score",
    "ret_since_entry",
    "max_ret_since_entry",
    "drawdown_from_post_entry_high",
    "days_since_post_entry_high",
    "latest_ret_20",
    "latest_amount_ratio_20",
    "latest_volume_ratio_20",
    "latest_close_to_high_60",
    "latest_turnover",
    "above_ma20",
    "above_ma60",
]
LONG_CONTEXT_COLS = [
    "long_history_days",
    "long_ret_120",
    "long_ret_250",
    "long_ret_500",
    "long_ret_1000",
    "long_close_to_high_250",
    "long_close_to_high_500",
    "long_close_to_high_1000",
    "long_close_to_low_250",
    "long_close_to_low_500",
    "long_close_to_low_1000",
    "long_range_250",
    "long_range_500",
    "long_range_1000",
    "long_volume_20_vs_120",
    "long_volume_20_vs_250",
    "long_volume_20_vs_500",
    "long_amount_20_vs_120",
    "long_amount_20_vs_250",
    "long_amount_20_vs_500",
    "long_turnover_20_vs_120",
    "long_turnover_20_vs_250",
    "long_volume_dryup_rank_250",
    "long_amount_dryup_rank_250",
    "long_volatility_60",
    "long_volatility_120",
    "long_trend_slope_120",
    "long_trend_slope_250",
    "entry_position_250",
    "entry_position_500",
    "entry_to_sample_volume_dryup",
    "entry_to_sample_amount_dryup",
]
CONTEXT_COLS = SHORT_CONTEXT_COLS + LONG_CONTEXT_COLS


def safe_ratio(value: float, base: float) -> float:
    return float(value / base) if np.isfinite(value) and np.isfinite(base) and base > 0 else 0.0


def safe_window(series: pd.Series, end_idx: int, window: int) -> pd.Series:
    start = max(0, end_idx - window + 1)
    return series.iloc[start : end_idx + 1]


def safe_return(close: pd.Series, sample_idx: int, window: int) -> float:
    base_idx = sample_idx - window
    if base_idx < 0:
        return 0.0
    base = float(close.iloc[base_idx])
    latest = float(close.iloc[sample_idx])
    return latest / base - 1 if base > 0 else 0.0


def close_position(close_value: float, low_value: float, high_value: float) -> float:
    if not np.isfinite(close_value) or not np.isfinite(low_value) or not np.isfinite(high_value) or high_value <= low_value:
        return 0.0
    return float((close_value - low_value) / (high_value - low_value))


def mean_ratio(short_series: pd.Series, long_series: pd.Series) -> float:
    short_mean = float(short_series.mean()) if len(short_series) else np.nan
    long_mean = float(long_series.mean()) if len(long_series) else np.nan
    return safe_ratio(short_mean, long_mean)


def rolling_mean_rank(series: pd.Series, sample_idx: int, short_window: int, long_window: int) -> float:
    rolled = series.rolling(short_window, min_periods=max(3, short_window // 2)).mean()
    current = float(rolled.iloc[sample_idx])
    hist = safe_window(rolled, sample_idx, long_window).dropna()
    if not np.isfinite(current) or hist.empty:
        return 0.0
    return float((hist <= current).mean())


def post_entry_retest_features(feat: pd.DataFrame, entry_idx: int, sample_idx: int) -> Dict[str, object]:
    """Describe the post-entry path: first dip, then retest the entry close."""
    close = feat["close"].astype(float).reset_index(drop=True)
    entry_close = float(close.iloc[entry_idx]) if 0 <= entry_idx < len(close) else np.nan
    sample_close = float(close.iloc[sample_idx]) if 0 <= sample_idx < len(close) else np.nan
    empty = {
        "entry_close": entry_close if np.isfinite(entry_close) else 0.0,
        "post_entry_min_ret": 0.0,
        "post_entry_min_date": "",
        "days_to_post_entry_min": 0,
        "days_from_post_entry_min": 0,
        "entry_close_retest_gap": 0.0,
        "recovered_from_post_entry_low": 0.0,
        "entry_close_retest_signal": False,
    }
    if sample_idx <= entry_idx or not np.isfinite(entry_close) or entry_close <= 0 or not np.isfinite(sample_close):
        return empty

    post = close.iloc[entry_idx + 1 : sample_idx + 1]
    if post.empty:
        return empty

    trough_idx = int(post.idxmin())
    trough_close = float(close.iloc[trough_idx])
    trough_ret = trough_close / entry_close - 1 if trough_close > 0 else 0.0
    retest_gap = sample_close / entry_close - 1
    recovered = sample_close / trough_close - 1 if trough_close > 0 else 0.0
    days_to_min = trough_idx - entry_idx
    days_from_min = sample_idx - trough_idx

    # 先跌至少约 5%，再从低点回升，并且当前收盘回到爆量日收盘价附近。
    retest_signal = bool(
        trough_ret <= -0.05
        and days_to_min >= 2
        and days_from_min >= 1
        and recovered >= 0.04
        and -0.06 <= retest_gap <= 0.06
    )
    date_value = feat.iloc[trough_idx]["日期"]
    min_date = date_value.strftime("%Y-%m-%d") if hasattr(date_value, "strftime") else str(date_value)
    return {
        "entry_close": entry_close,
        "post_entry_min_ret": float(trough_ret),
        "post_entry_min_date": min_date,
        "days_to_post_entry_min": int(days_to_min),
        "days_from_post_entry_min": int(days_from_min),
        "entry_close_retest_gap": float(retest_gap),
        "recovered_from_post_entry_low": float(recovered),
        "entry_close_retest_signal": retest_signal,
    }


def breakout_pullback_hold_features(feat: pd.DataFrame, entry_idx: int, latest_idx: int) -> Dict[str, object]:
    """Describe a volume breakout that pulls back without breaking the breakout day's low."""
    close = feat["close"].astype(float).reset_index(drop=True)
    high = feat["最高"].astype(float).reset_index(drop=True)
    low = feat["最低"].astype(float).reset_index(drop=True)
    entry = feat.iloc[entry_idx]
    latest = feat.iloc[latest_idx]
    entry_close = float(close.iloc[entry_idx])
    entry_high = float(high.iloc[entry_idx])
    entry_low = float(low.iloc[entry_idx])
    latest_close = float(close.iloc[latest_idx])
    empty = {
        "entry_close": entry_close if np.isfinite(entry_close) else 0.0,
        "entry_high": entry_high if np.isfinite(entry_high) else 0.0,
        "entry_low": entry_low if np.isfinite(entry_low) else 0.0,
        "breakout_entry_score": 0.0,
        "pullback_low_date": "",
        "pullback_low_vs_entry_low": 0.0,
        "pullback_close_vs_entry_close": 0.0,
        "breakout_ret_since_entry": 0.0,
        "breakout_recovery_from_pullback": 0.0,
        "breakout_hold_signal": False,
    }
    if latest_idx <= entry_idx or entry_low <= 0 or entry_close <= 0 or latest_close <= 0:
        return empty

    post = feat.iloc[entry_idx + 1 : latest_idx + 1].copy()
    if post.empty:
        return empty

    low_idx = int(post["最低"].astype(float).idxmin())
    low_price = float(feat.loc[low_idx, "最低"])
    low_close = float(feat.loc[low_idx, "close"])
    low_date_value = feat.loc[low_idx, "日期"]
    low_date = low_date_value.strftime("%Y-%m-%d") if hasattr(low_date_value, "strftime") else str(low_date_value)

    pullback_low_vs_entry_low = low_price / entry_low - 1
    pullback_close_vs_entry_close = low_close / entry_close - 1
    ret_since_entry = latest_close / entry_close - 1
    recovery = latest_close / max(low_price, 1e-6) - 1
    entry_volume_ratio = max(
        float(entry.get("volume_ratio_20", 0) or 0),
        float(entry.get("volume_ratio_20_prev", entry.get("volume_ratio_20", 0)) or 0),
    )
    entry_amount_ratio = max(
        float(entry.get("amount_ratio_20", 0) or 0),
        float(entry.get("amount_ratio_20_prev", entry.get("amount_ratio_20", 0)) or 0),
    )
    entry_day_ret = float(entry.get("day_ret", 0) or 0)
    entry_body = float(entry.get("body_pct", 0) or 0)
    close_to_high = float(entry.get("close_to_high_60", 0) or 0)
    latest_close_to_high = float(latest.get("close_to_high_60", 0) or 0)

    breakout_entry_score = 0.0
    breakout_entry_score += 20.0 if entry_day_ret >= 0.04 and entry_body >= 0.012 else 0.0
    breakout_entry_score += 18.0 * max(0.0, min(1.0, (entry_volume_ratio - 1.6) / 2.0))
    breakout_entry_score += 18.0 * max(0.0, min(1.0, (entry_amount_ratio - 1.8) / 2.2))
    breakout_entry_score += 14.0 * max(0.0, min(1.0, (close_to_high + 0.08) / 0.10))
    breakout_entry_score += 10.0 if float(entry.get("above_ma20", 0)) == 1 and float(entry.get("above_ma60", 0)) == 1 else 0.0

    hold_signal = bool(
        breakout_entry_score >= 42.0
        and entry_volume_ratio >= 1.6
        and entry_amount_ratio >= 1.8
        and pullback_low_vs_entry_low >= -0.025
        and ret_since_entry >= 0.035
        and ret_since_entry <= 0.75
        and recovery >= 0.055
        and latest_close_to_high >= -0.12
        and float(latest.get("above_ma20", 0)) == 1
        and float(latest.get("above_ma60", 0)) == 1
    )
    return {
        "entry_close": entry_close,
        "entry_high": entry_high,
        "entry_low": entry_low,
        "breakout_entry_score": float(breakout_entry_score),
        "pullback_low_date": low_date,
        "pullback_low_vs_entry_low": float(pullback_low_vs_entry_low),
        "pullback_close_vs_entry_close": float(pullback_close_vs_entry_close),
        "breakout_ret_since_entry": float(ret_since_entry),
        "breakout_recovery_from_pullback": float(recovery),
        "breakout_hold_signal": hold_signal,
    }


def breakout_pullback_hold_rule_score(feat: pd.DataFrame, entry_idx: int, latest_idx: int) -> float:
    info = breakout_pullback_hold_features(feat, entry_idx, latest_idx)
    latest = feat.iloc[latest_idx]

    def clip01(x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        return max(0.0, min(1.0, float(x)))

    age = latest_idx - entry_idx
    ret_since_entry = float(info["breakout_ret_since_entry"])
    score = 0.0
    score += 14 * math.exp(-((age - 14) / 18) ** 2)
    score += 0.45 * float(info["breakout_entry_score"])
    score += 12 * clip01((0.025 + float(info["pullback_low_vs_entry_low"])) / 0.045)
    score += 14 * clip01((float(info["breakout_recovery_from_pullback"]) - 0.055) / 0.30)
    score += 14 * math.exp(-((ret_since_entry - 0.26) / 0.24) ** 2)
    score += 8 * clip01((float(latest.get("latest_close_to_high_60", latest.get("close_to_high_60", 0))) + 0.12) / 0.14)
    score += 6 if float(latest.get("above_ma20", 0)) == 1 and float(latest.get("above_ma60", 0)) == 1 else 0
    if bool(info["breakout_hold_signal"]):
        score += 6
    score -= 18 * clip01((ret_since_entry - 0.55) / 0.25)
    return round(min(100.0, score), 2)


def find_breakout_pullback_hold_entry(feat: pd.DataFrame, latest_idx: int, args: argparse.Namespace, min_date: pd.Timestamp) -> Optional[int]:
    lookback = int(getattr(args, "breakout_hold_lookback_days", 45))
    min_age = int(getattr(args, "breakout_hold_min_days", 1))
    entry_min = max(getattr(args, "seq_len", 240), latest_idx - lookback)
    entry_max = latest_idx - min_age
    if entry_max < entry_min:
        return None
    dates = pd.to_datetime(feat["日期"])
    best_idx = None
    best_score = -1.0
    for entry_idx in range(entry_min, entry_max + 1):
        if dates.iloc[entry_idx] < min_date:
            continue
        info = breakout_pullback_hold_features(feat, entry_idx, latest_idx)
        if not info["breakout_hold_signal"]:
            continue
        score = breakout_pullback_hold_rule_score(feat, entry_idx, latest_idx)
        if score > best_score:
            best_score = score
            best_idx = entry_idx
    return best_idx


@dataclass
class ContractionSample:
    code: str
    name: str
    label: int
    sample_idx: int
    entry_idx: int
    run_idx: int
    event_id: str
    seq: np.ndarray
    context: np.ndarray
    meta: Dict[str, object]


def sanitize(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def make_sequence(feat: pd.DataFrame, idx: int, seq_len: int) -> Optional[np.ndarray]:
    start = idx - seq_len + 1
    if start < 0:
        return None
    seq = feat.iloc[start : idx + 1][SEQ_COLS].to_numpy(dtype=np.float32)
    if seq.shape != (seq_len, len(SEQ_COLS)):
        return None
    return sanitize(seq)


def future_run_multiple(close: pd.Series, horizon: int) -> np.ndarray:
    future_max = close.iloc[::-1].rolling(horizon + 1, min_periods=1).max().iloc[::-1]
    return (future_max / close.replace(0, np.nan)).to_numpy(dtype=float)


def triple_volume_bull_mask(feat: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    volume_ratio = np.fmax(
        feat["volume_ratio_20"].to_numpy(dtype=float),
        feat.get("volume_ratio_20_prev", feat["volume_ratio_20"]).to_numpy(dtype=float),
    )
    amount_ratio = np.fmax(
        feat["amount_ratio_20"].to_numpy(dtype=float),
        feat.get("amount_ratio_20_prev", feat["amount_ratio_20"]).to_numpy(dtype=float),
    )
    day_ret = feat["day_ret"].to_numpy(dtype=float)
    body_pct = feat["body_pct"].to_numpy(dtype=float)
    gap_ret = feat["gap_ret"].to_numpy(dtype=float)
    body_ok = (body_pct >= args.entry_min_body) | (
        (day_ret >= args.entry_gap_min_ret) & (body_pct >= args.entry_gap_min_body)
    ) | (
        (gap_ret >= args.entry_gap_min_gap) & (body_pct >= args.entry_gap_min_body)
    )
    return (
        (volume_ratio >= args.entry_volume_ratio)
        & (amount_ratio >= args.entry_amount_ratio)
        & (day_ret >= args.entry_min_ret)
        & body_ok
        & (feat["turnover"].to_numpy(dtype=float) >= args.entry_min_turnover)
        & (feat["ret_20"].to_numpy(dtype=float) <= args.entry_max_ret20)
    )


def main_start_mask(feat: pd.DataFrame, future_mult_20: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    day_ret = feat["day_ret"].to_numpy(dtype=float)
    body = feat["body_pct"].to_numpy(dtype=float)
    ret3 = feat["ret_3"].to_numpy(dtype=float)
    amount_ratio = feat["amount_ratio_20"].to_numpy(dtype=float)
    volume_ratio = feat["volume_ratio_20"].to_numpy(dtype=float)
    price_surge = (
        (day_ret >= args.run_start_min_ret)
        & (body >= args.run_start_min_body)
        & (amount_ratio >= args.run_start_amount_ratio)
        & (volume_ratio >= args.run_start_volume_ratio)
    ) | (
        (ret3 >= args.run_start_ret3)
        & (amount_ratio >= args.run_start_amount_ratio)
    )
    second_volume_bull = (
        (volume_ratio >= args.second_volume_ratio)
        & (amount_ratio >= args.second_amount_ratio)
        & (day_ret >= args.second_min_ret)
        & (body >= args.second_min_body)
    )
    confirmed_start = price_surge | second_volume_bull
    if getattr(args, "allow_confirmed_run_start", False):
        return confirmed_start
    return (future_mult_20 >= args.month_multiple) & confirmed_start


def long_context_features(feat: pd.DataFrame, entry_idx: int, sample_idx: int) -> np.ndarray:
    close = feat["close"].astype(float).reset_index(drop=True)
    high = feat["最高"].astype(float).reset_index(drop=True)
    low = feat["最低"].astype(float).reset_index(drop=True)
    volume = feat["volume"].replace(0, np.nan).astype(float).reset_index(drop=True)
    amount = feat["amount"].replace(0, np.nan).astype(float).reset_index(drop=True)
    turnover = feat["turnover"].replace(0, np.nan).astype(float).reset_index(drop=True)
    ret = close.pct_change()

    latest_close = float(close.iloc[sample_idx])
    entry_close = float(close.iloc[entry_idx])
    latest20_volume = safe_window(volume, sample_idx, 20)
    latest20_amount = safe_window(amount, sample_idx, 20)
    latest20_turnover = safe_window(turnover, sample_idx, 20)
    entry20_volume = safe_window(volume, entry_idx, 20)
    entry20_amount = safe_window(amount, entry_idx, 20)

    values: List[float] = [float(sample_idx + 1)]
    for win in [120, 250, 500, 1000]:
        values.append(safe_return(close, sample_idx, win))
    for win in [250, 500, 1000]:
        high_win = safe_window(high, sample_idx, win)
        high_value = float(high_win.max()) if len(high_win) else np.nan
        values.append(latest_close / high_value - 1 if np.isfinite(high_value) and high_value > 0 else 0.0)
    for win in [250, 500, 1000]:
        low_win = safe_window(low, sample_idx, win)
        low_value = float(low_win.min()) if len(low_win) else np.nan
        values.append(latest_close / low_value - 1 if np.isfinite(low_value) and low_value > 0 else 0.0)
    for win in [250, 500, 1000]:
        high_win = safe_window(high, sample_idx, win)
        low_win = safe_window(low, sample_idx, win)
        high_value = float(high_win.max()) if len(high_win) else np.nan
        low_value = float(low_win.min()) if len(low_win) else np.nan
        values.append(high_value / low_value - 1 if np.isfinite(high_value) and np.isfinite(low_value) and low_value > 0 else 0.0)

    for win in [120, 250, 500]:
        values.append(mean_ratio(latest20_volume, safe_window(volume, sample_idx, win)))
    for win in [120, 250, 500]:
        values.append(mean_ratio(latest20_amount, safe_window(amount, sample_idx, win)))
    for win in [120, 250]:
        values.append(mean_ratio(latest20_turnover, safe_window(turnover, sample_idx, win)))
    values.append(rolling_mean_rank(volume, sample_idx, 20, 250))
    values.append(rolling_mean_rank(amount, sample_idx, 20, 250))
    values.append(float(ret.iloc[max(0, sample_idx - 59) : sample_idx + 1].std()) if sample_idx >= 5 else 0.0)
    values.append(float(ret.iloc[max(0, sample_idx - 119) : sample_idx + 1].std()) if sample_idx >= 5 else 0.0)
    values.append(safe_return(close.rolling(20).mean(), sample_idx, 120))
    values.append(safe_return(close.rolling(60).mean(), sample_idx, 250))

    for win in [250, 500]:
        high_win = safe_window(high, sample_idx, win)
        low_win = safe_window(low, sample_idx, win)
        high_value = float(high_win.max()) if len(high_win) else np.nan
        low_value = float(low_win.min()) if len(low_win) else np.nan
        values.append(close_position(entry_close, low_value, high_value))

    values.append(mean_ratio(latest20_volume, entry20_volume))
    values.append(mean_ratio(latest20_amount, entry20_amount))
    return sanitize(np.array(values, dtype=np.float32))


def contraction_context(feat: pd.DataFrame, entry_idx: int, sample_idx: int) -> np.ndarray:
    entry = feat.iloc[entry_idx]
    sample = feat.iloc[sample_idx]
    close = feat["close"].reset_index(drop=True)
    volume = feat["volume"].replace(0, np.nan).reset_index(drop=True)
    amount = feat["amount"].replace(0, np.nan).reset_index(drop=True)
    entry_close = float(close.iloc[entry_idx])
    sample_close = float(close.iloc[sample_idx])
    entry_volume = float(volume.iloc[entry_idx]) if np.isfinite(volume.iloc[entry_idx]) else np.nan
    entry_amount = float(amount.iloc[entry_idx]) if np.isfinite(amount.iloc[entry_idx]) else np.nan

    post_close = close.iloc[entry_idx : sample_idx + 1]
    high = float(post_close.max()) if len(post_close) else sample_close
    high_idx = int(post_close.idxmax()) if len(post_close) else sample_idx
    volume5 = float(volume.iloc[max(0, sample_idx - 4) : sample_idx + 1].mean())
    volume10 = float(volume.iloc[max(0, sample_idx - 9) : sample_idx + 1].mean())
    amount5 = float(amount.iloc[max(0, sample_idx - 4) : sample_idx + 1].mean())
    amount10 = float(amount.iloc[max(0, sample_idx - 9) : sample_idx + 1].mean())

    vol5 = safe_ratio(volume5, entry_volume)
    vol10 = safe_ratio(volume10, entry_volume)
    amt5 = safe_ratio(amount5, entry_amount)
    amt10 = safe_ratio(amount10, entry_amount)
    latest_vol = safe_ratio(float(volume.iloc[sample_idx]), entry_volume)
    latest_amt = safe_ratio(float(amount.iloc[sample_idx]), entry_amount)
    contraction = 1.0 - np.nanmean([vol5, vol10, amt5, amt10, latest_vol, latest_amt])
    ret_since_entry = sample_close / entry_close - 1 if entry_close > 0 else 0.0
    max_ret = high / entry_close - 1 if entry_close > 0 else 0.0
    drawdown = sample_close / high - 1 if high > 0 else 0.0

    short_values = np.array(
        [
            sample_idx - entry_idx,
            float(entry.get("day_ret", 0)),
            float(entry.get("body_pct", 0)),
            float(entry.get("volume_ratio_20", 0)),
            float(entry.get("amount_ratio_20", 0)),
            float(entry.get("volume_ratio_20_prev", entry.get("volume_ratio_20", 0))),
            float(entry.get("amount_ratio_20_prev", entry.get("amount_ratio_20", 0))),
            float(entry.get("turnover", 0)),
            float(entry.get("close_to_high_60", 0)),
            vol5,
            vol10,
            amt5,
            amt10,
            latest_vol,
            latest_amt,
            contraction,
            ret_since_entry,
            max_ret,
            drawdown,
            sample_idx - high_idx,
            float(sample.get("ret_20", 0)),
            float(sample.get("amount_ratio_20", 0)),
            float(sample.get("volume_ratio_20", 0)),
            float(sample.get("close_to_high_60", 0)),
            float(sample.get("turnover", 0)),
            float(sample.get("above_ma20", 0)),
            float(sample.get("above_ma60", 0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([sanitize(short_values), long_context_features(feat, entry_idx, sample_idx)])


def is_contracted_not_running(feat: pd.DataFrame, entry_idx: int, sample_idx: int, args: argparse.Namespace) -> bool:
    if sample_idx - entry_idx < args.min_contract_days:
        return False
    ctx = dict(zip(CONTEXT_COLS, contraction_context(feat, entry_idx, sample_idx)))
    sample = feat.iloc[sample_idx]
    if ctx["volume_5_vs_entry"] > args.max_volume5_vs_entry:
        return False
    if ctx["volume_10_vs_entry"] > args.max_volume10_vs_entry:
        return False
    if ctx["amount_10_vs_entry"] > args.max_amount10_vs_entry:
        return False
    if ctx["latest_volume_vs_entry"] > args.max_latest_volume_vs_entry:
        return False
    if ctx["max_ret_since_entry"] > args.max_pre_run_ret:
        return False
    if ctx["ret_since_entry"] > args.max_current_ret_since_entry:
        return False
    if float(sample.get("ret_20", 0)) > args.max_sample_ret20:
        return False
    if float(sample.get("day_ret", 0)) >= args.latest_started_ret and float(sample.get("body_pct", 0)) >= args.latest_started_body:
        return False
    return True


def make_sample(
    item: StockItem,
    feat: pd.DataFrame,
    sample_idx: int,
    entry_idx: int,
    run_idx: int,
    label: int,
    event_id: str,
    month_multiple: float,
    seq_len: int,
) -> Optional[ContractionSample]:
    seq = make_sequence(feat, sample_idx, seq_len)
    if seq is None:
        return None
    ctx = contraction_context(feat, entry_idx, sample_idx)
    sample = feat.iloc[sample_idx]
    entry = feat.iloc[entry_idx]
    run_date = feat.iloc[run_idx]["日期"].strftime("%Y-%m-%d") if 0 <= run_idx < len(feat) else ""
    meta = {
        "code": item.code,
        "name": item.name,
        "sample_date": sample["日期"].strftime("%Y-%m-%d"),
        "entry_date": entry["日期"].strftime("%Y-%m-%d"),
        "run_date": run_date,
        "entry_age": sample_idx - entry_idx,
        "lead_days_to_run": run_idx - sample_idx if run_idx >= 0 else np.nan,
        "label": label,
        "close": float(sample["close"]),
        "pct_chg": float(sample["pct_chg"]),
        "amount": float(sample["amount"]),
        "entry_amount": float(entry["amount"]),
        "entry_pct_chg": float(entry["pct_chg"]),
        "month_run_multiple": round(float(month_multiple), 4) if np.isfinite(month_multiple) else np.nan,
        "event_id": event_id,
    }
    for name, value in zip(CONTEXT_COLS, ctx):
        meta[name] = float(value)
    return ContractionSample(item.code, item.name, label, sample_idx, entry_idx, run_idx, event_id, seq, ctx, meta)


def detect_for_stock(item: StockItem, hist: pd.DataFrame, args: argparse.Namespace, min_date: pd.Timestamp) -> Tuple[List[ContractionSample], List[ContractionSample], List[Dict[str, object]]]:
    feat = add_features(hist)
    if feat.empty or len(feat) < args.seq_len + args.month_days + 30:
        return [], [], []
    work = feat.reset_index(drop=True)
    close = work["close"].reset_index(drop=True)
    future_mult = future_run_multiple(close, args.month_days)
    entry_mask = triple_volume_bull_mask(work, args)
    run_mask = main_start_mask(work, future_mult, args)
    dates = pd.to_datetime(work["日期"])

    positives: List[ContractionSample] = []
    negatives: List[ContractionSample] = []
    events: List[Dict[str, object]] = []
    rng = random.Random(args.random_state + int(item.code))

    latest_complete_idx = len(work) - args.month_days - 1
    entry_indices = np.flatnonzero(entry_mask)
    for entry_idx in entry_indices:
        if entry_idx < args.seq_len or entry_idx >= latest_complete_idx:
            continue
        if dates.iloc[entry_idx] < min_date:
            continue
        run_search_start = entry_idx + args.min_contract_days
        run_search_end = min(entry_idx + args.max_entry_to_run_days, latest_complete_idx)
        if run_search_start >= run_search_end:
            continue
        run_candidates = np.flatnonzero(run_mask[run_search_start : run_search_end + 1]) + run_search_start
        run_candidates = [int(x) for x in run_candidates if is_contracted_not_running(work, entry_idx, max(entry_idx + args.min_contract_days, x - args.lead_max), args)]
        run_idx = run_candidates[0] if run_candidates else -1
        if run_idx >= 0:
            event_id = f"{item.code}_{work.iloc[entry_idx]['日期'].strftime('%Y%m%d')}_{run_idx}"
            events.append(
                {
                    "code": item.code,
                    "name": item.name,
                    "entry_date": work.iloc[entry_idx]["日期"].strftime("%Y-%m-%d"),
                    "run_date": work.iloc[run_idx]["日期"].strftime("%Y-%m-%d"),
                    "entry_to_run_days": run_idx - entry_idx,
                    "entry_pct_chg": float(work.iloc[entry_idx]["pct_chg"]),
                    "entry_amount": float(work.iloc[entry_idx]["amount"]),
                    "entry_volume_ratio_20": float(work.iloc[entry_idx]["volume_ratio_20"]),
                    "entry_amount_ratio_20": float(work.iloc[entry_idx]["amount_ratio_20"]),
                    "month_run_multiple": round(float(future_mult[run_idx]), 4),
                }
            )
            candidate_indices = []
            for sample_idx in range(entry_idx + args.min_contract_days, run_idx - args.lead_min + 1, args.positive_stride):
                if sample_idx < args.seq_len:
                    continue
                if not is_contracted_not_running(work, entry_idx, sample_idx, args):
                    continue
                candidate_indices.append(sample_idx)
            # Always include the final pre-run window, but do not only train on
            # it; earlier contracted days are what current inference often sees.
            for sample_idx in range(max(entry_idx + args.min_contract_days, run_idx - args.lead_max), run_idx - args.lead_min + 1):
                if sample_idx < args.seq_len:
                    continue
                if is_contracted_not_running(work, entry_idx, sample_idx, args):
                    candidate_indices.append(sample_idx)
            candidate_indices = sorted(set(candidate_indices))
            if len(candidate_indices) > args.max_positive_samples_per_event:
                rng.shuffle(candidate_indices)
                candidate_indices = sorted(candidate_indices[: args.max_positive_samples_per_event])
            for sample_idx in candidate_indices:
                sample = make_sample(item, work, sample_idx, entry_idx, run_idx, 1, event_id, future_mult[run_idx], args.seq_len)
                if sample is not None:
                    positives.append(sample)
        else:
            if rng.random() > args.negative_entry_sample_rate:
                continue
            pool_start = entry_idx + args.min_contract_days
            pool_end = min(entry_idx + args.max_entry_to_run_days, latest_complete_idx)
            pool = list(range(pool_start, pool_end + 1, args.negative_stride))
            rng.shuffle(pool)
            picked = 0
            for sample_idx in pool:
                if picked >= args.max_negatives_per_entry:
                    break
                if not is_contracted_not_running(work, entry_idx, sample_idx, args):
                    continue
                if run_mask[sample_idx + args.lead_min : min(sample_idx + args.lead_max + 1, len(run_mask))].any():
                    continue
                sample = make_sample(item, work, sample_idx, entry_idx, -1, 0, f"{item.code}_{work.iloc[entry_idx]['日期'].strftime('%Y%m%d')}_neg", future_mult[sample_idx], args.seq_len)
                if sample is not None:
                    negatives.append(sample)
                    picked += 1
    if len(negatives) > args.max_negatives_per_stock:
        negatives = rng.sample(negatives, args.max_negatives_per_stock)
    return positives, negatives, events


def load_manual_label_events(args: argparse.Namespace) -> List[Dict[str, object]]:
    if getattr(args, "disable_manual_labels", False):
        return []
    label_path = Path(getattr(args, "manual_label_file", "") or "")
    if not label_path.exists():
        return []
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    events = payload.get("events", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise RuntimeError(f"manual label file must contain a list of events: {label_path}")
    out: List[Dict[str, object]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        code = str(event.get("code", "")).strip().zfill(6)[-6:]
        entry_date = event.get("entry_date")
        run_date = event.get("run_date")
        if not code or not entry_date or not run_date:
            continue
        item = dict(event)
        item["code"] = code
        item["entry_date"] = str(entry_date)
        item["run_date"] = str(run_date)
        item.setdefault("name", code)
        out.append(item)
    return out


def add_manual_items(items: List[StockItem], manual_events: List[Dict[str, object]]) -> List[StockItem]:
    by_code = {item.code: item for item in items}
    for event in manual_events:
        code = str(event["code"])
        if code not in by_code:
            by_code[code] = StockItem(code=code, name=str(event.get("name") or code))
    return list(by_code.values())


def _date_to_index(work: pd.DataFrame, value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    date = pd.Timestamp(value)
    matches = np.flatnonzero(pd.to_datetime(work["日期"]).to_numpy() == np.datetime64(date))
    if len(matches) == 0:
        return None
    return int(matches[0])


def _date_to_index_at_or_before(work: pd.DataFrame, value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    date = pd.Timestamp(value)
    dates = pd.to_datetime(work["日期"])
    matches = np.flatnonzero((dates <= date).to_numpy())
    if len(matches) == 0:
        return None
    return int(matches[-1])


def _manual_positive_indices(work: pd.DataFrame, entry_idx: int, run_idx: int, event: Dict[str, object], args: argparse.Namespace) -> List[int]:
    indices: List[int] = []
    for value in event.get("positive_dates") or []:
        idx = _date_to_index(work, value)
        if idx is not None:
            indices.append(idx)

    start_idx = _date_to_index(work, event.get("positive_start"))
    end_idx = _date_to_index(work, event.get("positive_end"))
    if start_idx is not None and end_idx is not None:
        indices.extend(range(start_idx, end_idx + 1))

    if not indices:
        left = max(entry_idx + args.min_contract_days, run_idx - args.lead_max)
        right = run_idx - args.lead_min
        indices.extend(range(left, right + 1))

    if bool(event.get("include_run_date", False)):
        indices.append(run_idx)

    return sorted({idx for idx in indices if args.seq_len <= idx <= run_idx})


def manual_label_samples_for_stock(
    item: StockItem,
    hist: pd.DataFrame,
    manual_events: List[Dict[str, object]],
    args: argparse.Namespace,
    min_date: pd.Timestamp,
) -> Tuple[List[ContractionSample], List[Dict[str, object]]]:
    if not manual_events:
        return [], []
    feat = add_features(hist)
    if feat.empty or len(feat) < args.seq_len + 30:
        return [], []
    work = feat.reset_index(drop=True)
    future_mult = future_run_multiple(work["close"].reset_index(drop=True), args.month_days)
    positives: List[ContractionSample] = []
    events: List[Dict[str, object]] = []
    seen_samples = set()

    for event in manual_events:
        entry_idx = _date_to_index(work, event.get("entry_date"))
        run_idx = _date_to_index(work, event.get("run_date"))
        if run_idx is None:
            run_idx = _date_to_index_at_or_before(work, event.get("run_date"))
        if entry_idx is None or run_idx is None or run_idx <= entry_idx:
            continue
        if pd.Timestamp(work.iloc[entry_idx]["日期"]) < min_date:
            continue
        event_id = f"{item.code}_{pd.Timestamp(work.iloc[entry_idx]['日期']).strftime('%Y%m%d')}_{run_idx}_manual"
        entry = work.iloc[entry_idx]
        run = work.iloc[run_idx]
        event_row = {
            "code": item.code,
            "name": item.name,
            "entry_date": entry["日期"].strftime("%Y-%m-%d"),
            "run_date": run["日期"].strftime("%Y-%m-%d"),
            "confirm_date": str(event.get("confirm_date") or ""),
            "entry_to_run_days": run_idx - entry_idx,
            "entry_pct_chg": float(entry["pct_chg"]),
            "entry_amount": float(entry["amount"]),
            "entry_volume_ratio_20": float(entry.get("volume_ratio_20", 0)),
            "entry_amount_ratio_20": float(entry.get("amount_ratio_20", 0)),
            "entry_volume_ratio_20_prev": float(entry.get("volume_ratio_20_prev", entry.get("volume_ratio_20", 0))),
            "entry_amount_ratio_20_prev": float(entry.get("amount_ratio_20_prev", entry.get("amount_ratio_20", 0))),
            "month_run_multiple": round(float(future_mult[run_idx]), 4) if np.isfinite(future_mult[run_idx]) else np.nan,
            "label_source": "manual",
            "label_note": str(event.get("note") or ""),
        }
        events.append(event_row)

        for sample_idx in _manual_positive_indices(work, entry_idx, run_idx, event, args):
            key = (item.code, entry_idx, sample_idx, run_idx)
            if key in seen_samples:
                continue
            sample = make_sample(item, work, sample_idx, entry_idx, run_idx, 1, event_id, future_mult[run_idx], args.seq_len)
            if sample is None:
                continue
            sample.meta["label_source"] = "manual"
            sample.meta["sample_weight"] = float(event.get("sample_weight", args.manual_loss_weight))
            sample.meta["confirm_date"] = str(event.get("confirm_date") or "")
            sample.meta["manual_note"] = str(event.get("note") or "")
            positives.append(sample)
            seen_samples.add(key)
    return positives, events


def build_arrays(samples: List[ContractionSample]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    return (
        np.stack([s.seq for s in samples]).astype(np.float32),
        np.stack([s.context for s in samples]).astype(np.float32),
        np.array([s.label for s in samples], dtype=np.float32),
        pd.DataFrame([s.meta for s in samples]),
    )


def split_indices(meta: pd.DataFrame, y: np.ndarray, val_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    label_source = meta["label_source"] if "label_source" in meta.columns else pd.Series("", index=meta.index)
    manual_mask = label_source.fillna("").astype(str).eq("manual").to_numpy()
    manual_idx = np.flatnonzero(manual_mask)
    regular_idx = np.flatnonzero(~manual_mask)
    if len(regular_idx) == 0:
        idx = np.arange(len(y))
        train_idx, val_idx = train_test_split(idx, test_size=val_size, stratify=y, random_state=random_state)
        return np.asarray(train_idx), np.asarray(val_idx)

    regular_meta = meta.iloc[regular_idx]
    regular_y = y[regular_idx]
    dates = pd.to_datetime(regular_meta["sample_date"], errors="coerce")
    order = regular_idx[np.argsort(dates.to_numpy())]
    split = int(len(order) * (1 - val_size))
    train_idx = np.asarray(order[:split])
    val_idx = np.asarray(order[split:])
    if y[train_idx].sum() >= 5 and y[val_idx].sum() >= 3 and (y[val_idx] == 0).sum() >= 10:
        if len(manual_idx):
            train_idx = np.concatenate([train_idx, manual_idx])
        return train_idx, val_idx
    train_rel, val_rel = train_test_split(
        np.arange(len(regular_idx)),
        test_size=val_size,
        stratify=regular_y if len(np.unique(regular_y)) > 1 else None,
        random_state=random_state,
    )
    train_idx = regular_idx[train_rel]
    val_idx = regular_idx[val_rel]
    if len(manual_idx):
        train_idx = np.concatenate([train_idx, manual_idx])
    return np.asarray(train_idx), np.asarray(val_idx)


def train_model(samples: List[ContractionSample], args: argparse.Namespace) -> Tuple[TemporalAttentionNet, Dict[str, object], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    ensure_torch()
    X_seq, X_ctx, y, meta = build_arrays(samples)
    train_idx, val_idx = split_indices(meta, y, args.val_size, args.random_state)
    if "sample_weight" in meta.columns:
        sample_weight = pd.to_numeric(meta["sample_weight"], errors="coerce").fillna(1.0).clip(lower=0.1).to_numpy(dtype=np.float32)
    else:
        sample_weight = np.ones(len(meta), dtype=np.float32)
    seq_mean = X_seq[train_idx].mean(axis=(0, 1), keepdims=True)
    seq_std = X_seq[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
    ctx_mean = X_ctx[train_idx].mean(axis=0, keepdims=True)
    ctx_std = X_ctx[train_idx].std(axis=0, keepdims=True) + 1e-6
    X_seq = sanitize((X_seq - seq_mean) / seq_std)
    X_ctx = sanitize((X_ctx - ctx_mean) / ctx_std)
    device = choose_device(args)
    model = TemporalAttentionNet(len(SEQ_COLS), len(CONTEXT_COLS), d_model=args.d_model, dropout=args.dropout).to(device)

    y_train = y[train_idx]
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    weights = np.where(y_train > 0, max(1.0, neg / max(pos, 1.0)), 1.0).astype(np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True)
    train_ds = TensorDataset(
        torch.from_numpy(X_seq[train_idx]).float(),
        torch.from_numpy(X_ctx[train_idx]).float(),
        torch.from_numpy(y[train_idx]).float(),
        torch.from_numpy(sample_weight[train_idx]).float(),
    )
    val_ds = TensorDataset(torch.from_numpy(X_seq[val_idx]).float(), torch.from_numpy(X_ctx[val_idx]).float(), torch.from_numpy(y[val_idx]).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, pin_memory=device.type == "cuda")

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = -1.0
    best_state = None
    best_metrics: Dict[str, object] = {}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for seq_b, ctx_b, y_b, w_b in train_loader:
            seq_b = seq_b.to(device, non_blocking=True)
            ctx_b = ctx_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            w_b = w_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(seq_b, ctx_b)
            loss_vec = loss_fn(logits, y_b)
            loss = (loss_vec * w_b).sum() / w_b.sum().clamp_min(1e-6)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for seq_b, ctx_b, y_b in val_loader:
                logits = model(seq_b.to(device), ctx_b.to(device))
                probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
                ys.extend(y_b.numpy().tolist())
        y_val = np.asarray(ys, dtype=np.float32)
        p_val = np.nan_to_num(np.asarray(probs, dtype=np.float32), nan=0.5, posinf=1.0, neginf=0.0)
        auc = float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else 0.5
        ap = float(average_precision_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else 0.0
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"[{_now_text()}] epoch {epoch:03d} loss={np.mean(losses) if losses else float('nan'):.4f} val_auc={auc:.4f} val_ap={ap:.4f}")
        if auc > best_auc and all(torch.isfinite(p.detach()).all().item() for p in model.parameters()):
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = {"val_auc": round(auc, 4), "val_ap": round(ap, 4), "epoch": epoch}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(f"[{_now_text()}] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(model(torch.from_numpy(X_seq[val_idx]).float().to(device), torch.from_numpy(X_ctx[val_idx]).float().to(device))).detach().cpu().numpy()
    p_val = np.nan_to_num(p_val, nan=0.5, posinf=1.0, neginf=0.0)
    y_val = y[val_idx]
    pred = (p_val >= args.threshold).astype(int)
    best_metrics.update(
        {
            "samples": int(len(samples)),
            "positives": int(y.sum()),
            "negatives": int((y == 0).sum()),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "val_positives": int(y_val.sum()),
            "val_negatives": int((y_val == 0).sum()),
            "classification_report": classification_report(y_val, pred, output_dict=True, zero_division=0),
            "device": str(device),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    )
    model_path = Path("model_cache") / "volume_contraction_breakout_5y_tcn.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "seq_cols": SEQ_COLS,
            "context_cols": CONTEXT_COLS,
            "seq_mean": seq_mean,
            "seq_std": seq_std,
            "ctx_mean": ctx_mean,
            "ctx_std": ctx_std,
            "args": vars(args),
            "metrics": best_metrics,
        },
        model_path,
    )
    best_metrics["model_path"] = str(model_path)
    return model, best_metrics, (seq_mean, seq_std, ctx_mean, ctx_std)


def contraction_rule_score(feat: pd.DataFrame, entry_idx: int, latest_idx: int) -> float:
    ctx = dict(zip(CONTEXT_COLS, contraction_context(feat, entry_idx, latest_idx)))
    retest = post_entry_retest_features(feat, entry_idx, latest_idx)

    def clip01(x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        return max(0.0, min(1.0, float(x)))

    age = ctx["entry_age"]
    breakout_score = 18 * math.exp(-((age - 45) / 28) ** 2)
    breakout_score += 16 * clip01((ctx["entry_volume_ratio_20"] - 3.0) / 4.5)
    breakout_score += 14 * clip01((ctx["entry_amount_ratio_20"] - 2.5) / 5.0)
    breakout_score += 18 * clip01((0.62 - ctx["volume_10_vs_entry"]) / 0.62)
    breakout_score += 12 * clip01((0.70 - ctx["amount_10_vs_entry"]) / 0.70)
    breakout_score += 10 * clip01((ctx["latest_close_to_high_60"] + 0.15) / 0.15)
    breakout_score += 6 * clip01((ctx["above_ma20"] + ctx["above_ma60"]) / 2.0)
    breakout_score += 6 * clip01((0.65 - ctx["max_ret_since_entry"]) / 0.65)

    low_position_score = 18 * math.exp(-((age - 110) / 75) ** 2)
    low_position_score += 22 * clip01((0.62 - ctx["volume_10_vs_entry"]) / 0.62)
    low_position_score += 18 * clip01((0.68 - ctx["amount_10_vs_entry"]) / 0.68)
    low_position_score += 14 * clip01((0.45 - ctx["max_ret_since_entry"]) / 0.45)
    low_position_score += 10 * clip01((0.55 - abs(ctx["ret_since_entry"])) / 0.55)
    low_position_score += 8 * clip01((0.08 - ctx["latest_ret_20"]) / 0.30)
    low_position_score += 8 * clip01((1.15 - ctx["latest_volume_ratio_20"]) / 1.15)
    low_position_score += 8 * clip01((ctx["entry_volume_ratio_20"] - 2.5) / 3.5)
    low_position_score += 6 * clip01((ctx["entry_amount_ratio_20"] - 2.5) / 3.5)

    retest_score = 0.0
    retest_score += 8 * clip01((-0.05 - float(retest["post_entry_min_ret"])) / 0.25)
    retest_score += 10 * clip01((0.08 - abs(float(retest["entry_close_retest_gap"]))) / 0.08)
    retest_score += 8 * clip01((float(retest["recovered_from_post_entry_low"]) - 0.04) / 0.28)
    if retest["entry_close_retest_signal"]:
        retest_score += 8

    return round(min(100.0, max(breakout_score, low_position_score) + 0.45 * retest_score), 2)


def reason_text(feat: pd.DataFrame, entry_idx: int, latest_idx: int) -> str:
    ctx = dict(zip(CONTEXT_COLS, contraction_context(feat, entry_idx, latest_idx)))
    retest = post_entry_retest_features(feat, entry_idx, latest_idx)
    reasons = [
        f"{int(ctx['entry_age'])}天前三倍爆量阳",
        f"进场量{ctx['entry_volume_ratio_20']:.1f}倍",
        f"10日量缩至{ctx['volume_10_vs_entry']:.2f}",
    ]
    if ctx["amount_10_vs_entry"] <= 0.7:
        reasons.append(f"10日额缩至{ctx['amount_10_vs_entry']:.2f}")
    if ctx["latest_close_to_high_60"] >= -0.08:
        reasons.append("接近60日高点")
    if ctx["above_ma20"] == 1 and ctx["above_ma60"] == 1:
        reasons.append("站上20/60日线")
    if retest["entry_close_retest_signal"]:
        reasons.append("先跌后回测爆量收盘")
    if ctx["max_ret_since_entry"] <= 0.55:
        reasons.append("尚未主升")
    return "、".join(reasons[:7])


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def macd_water_status(feat: pd.DataFrame, start_idx: int, end_idx: int, args: argparse.Namespace) -> Dict[str, object]:
    if "macd_dif" not in feat.columns or "macd_dea" not in feat.columns:
        return {"ok": False, "min_dif": 0.0, "min_dea": 0.0}
    start_idx = max(0, int(start_idx))
    end_idx = min(len(feat) - 1, int(end_idx))
    if end_idx < start_idx:
        return {"ok": False, "min_dif": 0.0, "min_dea": 0.0}
    window = feat.iloc[start_idx : end_idx + 1]
    dif = pd.to_numeric(window["macd_dif"], errors="coerce")
    dea = pd.to_numeric(window["macd_dea"], errors="coerce")
    if dif.isna().any() or dea.isna().any():
        return {"ok": False, "min_dif": 0.0, "min_dea": 0.0}
    min_dif = float(dif.min())
    min_dea = float(dea.min())
    line = float(getattr(args, "macd_waterline_min", 0.0))
    require_dea = bool(getattr(args, "macd_require_dea_above_water", True))
    ok = bool(min_dif >= line and (min_dea >= line if require_dea else True))
    return {"ok": ok, "min_dif": min_dif, "min_dea": min_dea}


def is_pre_breakout_contracted(feat: pd.DataFrame, entry_idx: int, check_idx: int, args: argparse.Namespace) -> bool:
    if check_idx - entry_idx < max(1, int(getattr(args, "min_contract_days", 5))):
        return False
    ctx = dict(zip(CONTEXT_COLS, contraction_context(feat, entry_idx, check_idx)))
    max_volume10 = float(getattr(args, "entry_close_breakout_max_volume10_vs_entry", args.max_volume10_vs_entry))
    max_latest_volume = float(
        getattr(args, "entry_close_breakout_max_latest_volume_vs_entry", args.max_latest_volume_vs_entry)
    )
    max_amount10 = float(getattr(args, "entry_close_breakout_max_amount10_vs_entry", args.max_amount10_vs_entry))
    max_pre_ret = float(getattr(args, "entry_close_breakout_max_pre_breakout_ret", args.max_pre_run_ret))
    return bool(
        ctx["volume_5_vs_entry"] <= args.max_volume5_vs_entry
        and ctx["volume_10_vs_entry"] <= max_volume10
        and ctx["amount_10_vs_entry"] <= max_amount10
        and ctx["latest_volume_vs_entry"] <= max_latest_volume
        and ctx["max_ret_since_entry"] <= max_pre_ret
        and ctx["ret_since_entry"] <= args.max_current_ret_since_entry
    )


def entry_close_breakout_features(
    feat: pd.DataFrame,
    entry_idx: int,
    trigger_idx: int,
    latest_idx: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    close = feat["close"].astype(float).reset_index(drop=True)
    volume = feat["volume"].replace(0, np.nan).astype(float).reset_index(drop=True)
    amount = feat["amount"].replace(0, np.nan).astype(float).reset_index(drop=True)
    entry = feat.iloc[entry_idx]
    trigger = feat.iloc[trigger_idx]
    latest = feat.iloc[latest_idx]
    entry_close = float(close.iloc[entry_idx])
    trigger_close = float(close.iloc[trigger_idx])
    latest_close = float(close.iloc[latest_idx])
    trigger_body = float(trigger.get("body_pct", 0) or 0)
    trigger_day_ret = float(trigger.get("day_ret", 0) or 0)
    prior_close = close.iloc[entry_idx + 1 : trigger_idx]
    prior_max_close = float(prior_close.max()) if len(prior_close) else 0.0
    pre_idx = trigger_idx - 1
    ctx = dict(zip(CONTEXT_COLS, contraction_context(feat, entry_idx, pre_idx))) if pre_idx > entry_idx else {}
    entry_volume = float(volume.iloc[entry_idx]) if np.isfinite(volume.iloc[entry_idx]) else np.nan
    trigger_volume = float(volume.iloc[trigger_idx]) if np.isfinite(volume.iloc[trigger_idx]) else np.nan
    entry_amount = float(amount.iloc[entry_idx]) if np.isfinite(amount.iloc[entry_idx]) else np.nan
    trigger_amount = float(amount.iloc[trigger_idx]) if np.isfinite(amount.iloc[trigger_idx]) else np.nan
    macd_end = latest_idx if bool(getattr(args, "entry_close_breakout_require_macd_to_latest", True)) else trigger_idx
    macd = macd_water_status(feat, entry_idx, macd_end, args)

    min_body = float(getattr(args, "entry_close_breakout_min_body", 0.006))
    min_ret = float(getattr(args, "entry_close_breakout_min_ret", 0.0))
    close_buffer = float(getattr(args, "entry_close_breakout_close_buffer", 0.0))
    prior_tolerance = float(getattr(args, "entry_close_breakout_prior_tolerance", 0.01))
    max_trigger_volume = float(getattr(args, "entry_close_breakout_max_trigger_volume_vs_entry", 0.95))
    max_trigger_amount = float(getattr(args, "entry_close_breakout_max_trigger_amount_vs_entry", 1.20))
    max_ret_since_entry = float(getattr(args, "entry_close_breakout_max_ret_since_entry", 0.75))

    volume_vs_entry = safe_ratio(trigger_volume, entry_volume)
    amount_vs_entry = safe_ratio(trigger_amount, entry_amount)
    pre_contracted = is_pre_breakout_contracted(feat, entry_idx, pre_idx, args) if pre_idx > entry_idx else False
    macd_required = not bool(getattr(args, "disable_macd_water_filter", False))
    signal = bool(
        np.isfinite(entry_close)
        and entry_close > 0
        and trigger_close > entry_close * (1.0 + close_buffer)
        and trigger_body >= min_body
        and trigger_day_ret >= min_ret
        and prior_max_close <= entry_close * (1.0 + prior_tolerance)
        and volume_vs_entry <= max_trigger_volume
        and amount_vs_entry <= max_trigger_amount
        and latest_close / entry_close - 1 <= max_ret_since_entry
        and pre_contracted
        and (bool(macd["ok"]) if macd_required else True)
    )

    return {
        "signal": signal,
        "entry_close": entry_close,
        "trigger_date": trigger["日期"].strftime("%Y-%m-%d"),
        "trigger_close": trigger_close,
        "trigger_pct_chg": float(trigger.get("pct_chg", 0)),
        "trigger_body_pct": trigger_body,
        "trigger_volume_vs_entry": volume_vs_entry,
        "trigger_amount_vs_entry": amount_vs_entry,
        "trigger_volume_ratio_20": float(trigger.get("volume_ratio_20", 0)),
        "trigger_amount_ratio_20": float(trigger.get("amount_ratio_20", 0)),
        "prior_max_close": prior_max_close,
        "entry_close_breakout_gap": trigger_close / entry_close - 1 if entry_close > 0 else 0.0,
        "latest_ret_since_entry": latest_close / entry_close - 1 if entry_close > 0 else 0.0,
        "latest_ret_since_trigger": latest_close / trigger_close - 1 if trigger_close > 0 else 0.0,
        "pre_volume_10_vs_entry": float(ctx.get("volume_10_vs_entry", 0.0)),
        "pre_amount_10_vs_entry": float(ctx.get("amount_10_vs_entry", 0.0)),
        "pre_latest_volume_vs_entry": float(ctx.get("latest_volume_vs_entry", 0.0)),
        "macd_water_ok": bool(macd["ok"]),
        "macd_min_dif": float(macd["min_dif"]),
        "macd_min_dea": float(macd["min_dea"]),
        "latest_above_ma20": float(latest.get("above_ma20", 0)),
        "latest_above_ma60": float(latest.get("above_ma60", 0)),
    }


def entry_close_breakout_rule_score(
    feat: pd.DataFrame,
    entry_idx: int,
    trigger_idx: int,
    latest_idx: int,
    info: Dict[str, object],
) -> float:
    entry = feat.iloc[entry_idx]
    score = 42.0
    score += 10.0 * _clip01((float(entry.get("volume_ratio_20", 0)) - 3.0) / 3.5)
    score += 8.0 * _clip01((float(entry.get("amount_ratio_20", 0)) - 2.5) / 4.0)
    score += 14.0 * _clip01((0.62 - float(info["pre_volume_10_vs_entry"])) / 0.62)
    score += 10.0 * _clip01((0.70 - float(info["pre_amount_10_vs_entry"])) / 0.70)
    score += 10.0 * _clip01((float(info["entry_close_breakout_gap"]) - 0.002) / 0.08)
    score += 8.0 * _clip01((0.95 - float(info["trigger_volume_vs_entry"])) / 0.95)
    score += 8.0 * math.exp(-((latest_idx - trigger_idx) / 8.0) ** 2)
    score += 5.0 * _clip01(float(info["macd_min_dif"]) / 0.18)
    score += 5.0 * _clip01(float(info["macd_min_dea"]) / 0.12)
    score += 4.0 * _clip01((float(info["latest_above_ma20"]) + float(info["latest_above_ma60"])) / 2.0)
    score -= 12.0 * _clip01((float(info["latest_ret_since_entry"]) - 0.65) / 0.35)
    return round(max(0.0, min(100.0, score)), 2)


def entry_close_breakout_reason_text(
    feat: pd.DataFrame,
    entry_idx: int,
    trigger_idx: int,
    latest_idx: int,
    info: Dict[str, object],
) -> str:
    entry = feat.iloc[entry_idx]
    reasons = [
        f"{latest_idx - entry_idx}天前三倍爆量阳",
        f"进场量{float(entry.get('volume_ratio_20', 0)):.1f}倍",
        f"缩量至{float(info['pre_volume_10_vs_entry']):.2f}",
        f"{latest_idx - trigger_idx}天前阳线突破爆量收盘",
        "MACD水上",
    ]
    if float(info["trigger_volume_vs_entry"]) <= 0.70:
        reasons.append(f"突破量低于进场量{float(info['trigger_volume_vs_entry']):.2f}")
    if float(info["latest_above_ma20"]) == 1 and float(info["latest_above_ma60"]) == 1:
        reasons.append("站上20/60日线")
    return "、".join(reasons[:7])


def find_entry_close_breakout_signal(
    feat: pd.DataFrame,
    latest_idx: int,
    args: argparse.Namespace,
    min_date: pd.Timestamp,
) -> Optional[Tuple[int, int, Dict[str, object], float]]:
    if bool(getattr(args, "disable_entry_close_breakout_signal", False)):
        return None
    dates = pd.to_datetime(feat["日期"])
    entry_mask = triple_volume_bull_mask(feat, args)
    entry_min = max(0, latest_idx - int(getattr(args, "max_entry_to_run_days", 250)))
    entry_max = latest_idx - max(1, int(getattr(args, "min_contract_days", 5)))
    if entry_max < entry_min:
        return None
    recent_days = int(getattr(args, "entry_close_breakout_recent_days", 12))
    best: Optional[Tuple[int, int, Dict[str, object], float]] = None
    for entry_idx in np.flatnonzero(entry_mask[entry_min : entry_max + 1]) + entry_min:
        entry_idx = int(entry_idx)
        if dates.iloc[entry_idx] < min_date:
            continue
        trigger_min = entry_idx + max(1, int(getattr(args, "min_contract_days", 5)))
        trigger_max = min(latest_idx, entry_idx + int(getattr(args, "max_entry_to_run_days", 250)))
        for trigger_idx in range(trigger_min, trigger_max + 1):
            if latest_idx - trigger_idx > recent_days:
                continue
            info = entry_close_breakout_features(feat, entry_idx, trigger_idx, latest_idx, args)
            if not bool(info["signal"]):
                continue
            score = entry_close_breakout_rule_score(feat, entry_idx, trigger_idx, latest_idx, info)
            if best is None or score > best[3]:
                best = (entry_idx, trigger_idx, info, score)
    return best


def score_current(histories: Dict[str, Tuple[StockItem, pd.DataFrame]], model: TemporalAttentionNet, norm: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], args: argparse.Namespace, min_date: pd.Timestamp) -> pd.DataFrame:
    ensure_torch()
    seq_mean, seq_std, ctx_mean, ctx_std = norm
    device = choose_device(args)
    model = model.to(device)
    model.eval()
    breakout_enabled = bool(
        getattr(args, "enable_breakout_hold_signal", not getattr(args, "disable_breakout_hold_signal", False))
    )
    rows = []
    for i, (code, (item, hist)) in enumerate(histories.items(), start=1):
        feat = add_features(hist)
        if feat.empty or len(feat) < args.seq_len + args.min_contract_days + 2:
            continue
        work = feat.reset_index(drop=True)
        latest_idx = len(work) - 1
        latest = work.iloc[latest_idx]
        if float(latest.get("amount", 0)) < args.candidate_min_amount:
            continue
        dates = pd.to_datetime(work["日期"])
        stock_rows = []

        entry_close_breakout = find_entry_close_breakout_signal(work, latest_idx, args, min_date)
        if entry_close_breakout is not None:
            best_entry, trigger_idx, info, rule = entry_close_breakout
            entry = work.iloc[best_entry]
            trigger = work.iloc[trigger_idx]
            stock_rows.append(
                {
                    "signal_priority": 2,
                    "signal_type": "entry_close_breakout",
                    "code": item.code,
                    "name": item.name,
                    "latest_date": latest["日期"].strftime("%Y-%m-%d"),
                    "entry_date": entry["日期"].strftime("%Y-%m-%d"),
                    "trigger_date": info["trigger_date"],
                    "entry_age": latest_idx - best_entry,
                    "trigger_age": latest_idx - trigger_idx,
                    "entry_close": round(float(info["entry_close"]), 3),
                    "trigger_close": round(float(info["trigger_close"]), 3),
                    "close": round(float(latest["close"]), 3),
                    "pct_chg": round(float(latest["pct_chg"]), 2),
                    "amount": float(latest["amount"]),
                    "amount_text": _format_amount(float(latest["amount"])),
                    "entry_pct_chg": round(float(entry["pct_chg"]), 2),
                    "entry_amount_text": _format_amount(float(entry["amount"])),
                    "entry_volume_ratio_20": round(float(entry.get("volume_ratio_20", 0)), 2),
                    "entry_amount_ratio_20": round(float(entry.get("amount_ratio_20", 0)), 2),
                    "trigger_pct_chg": round(float(info["trigger_pct_chg"]), 2),
                    "trigger_body_pct": round(float(info["trigger_body_pct"]) * 100, 2),
                    "trigger_volume_vs_entry": round(float(info["trigger_volume_vs_entry"]), 3),
                    "trigger_amount_vs_entry": round(float(info["trigger_amount_vs_entry"]), 3),
                    "trigger_volume_ratio_20": round(float(info["trigger_volume_ratio_20"]), 2),
                    "volume_10_vs_entry": round(float(info["pre_volume_10_vs_entry"]), 3),
                    "amount_10_vs_entry": round(float(info["pre_amount_10_vs_entry"]), 3),
                    "ret_since_entry": round(float(info["latest_ret_since_entry"]) * 100, 2),
                    "max_ret_since_entry": np.nan,
                    "post_entry_min_ret": np.nan,
                    "post_entry_min_date": "",
                    "entry_close_retest_gap": np.nan,
                    "recovered_from_post_entry_low": round(float(info["latest_ret_since_trigger"]) * 100, 2),
                    "entry_close_retest_signal": False,
                    "entry_close_breakout_gap": round(float(info["entry_close_breakout_gap"]) * 100, 2),
                    "macd_water_ok": bool(info["macd_water_ok"]),
                    "macd_min_dif": round(float(info["macd_min_dif"]), 4),
                    "macd_min_dea": round(float(info["macd_min_dea"]), 4),
                    "breakout_hold_signal": False,
                    "close_to_high_60": round(float(latest.get("close_to_high_60", 0)) * 100, 2),
                    "model_prob": np.nan,
                    "rule_score": rule,
                    "final_score": rule,
                    "reason": entry_close_breakout_reason_text(work, best_entry, trigger_idx, latest_idx, info),
                }
            )

        entry_mask = triple_volume_bull_mask(work, args)
        entry_min = max(args.seq_len, latest_idx - args.max_entry_to_run_days)
        entry_max = latest_idx - args.min_contract_days
        entries = []
        if entry_max >= entry_min:
            entries = np.flatnonzero(entry_mask[entry_min : entry_max + 1]) + entry_min
            entries = [
                int(x)
                for x in entries
                if dates.iloc[x] >= min_date and is_contracted_not_running(work, int(x), latest_idx, args)
            ]
        if entries:
            best_entry = max(
                entries,
                key=lambda x: float(work.iloc[x].get("volume_ratio_20", 0))
                + float(work.iloc[x].get("amount_ratio_20", 0)),
            )
            seq = make_sequence(work, latest_idx, args.seq_len)
            if seq is not None:
                ctx = contraction_context(work, best_entry, latest_idx)
                retest = post_entry_retest_features(work, best_entry, latest_idx)
                if not (getattr(args, "require_entry_close_retest", False) and not retest["entry_close_retest_signal"]):
                    seq_n = sanitize((seq[None, :, :] - seq_mean) / seq_std)
                    ctx_n = sanitize((ctx[None, :] - ctx_mean) / ctx_std)
                    with torch.no_grad():
                        prob = torch.sigmoid(
                            model(
                                torch.from_numpy(seq_n).float().to(device),
                                torch.from_numpy(ctx_n).float().to(device),
                            )
                        ).detach().cpu().numpy()[0]
                    if not np.isfinite(prob):
                        prob = 0.0
                    rule = contraction_rule_score(work, best_entry, latest_idx)
                    final = 0.68 * float(prob) * 100 + 0.32 * rule
                    entry = work.iloc[best_entry]
                    ctxd = dict(zip(CONTEXT_COLS, ctx))
                    stock_rows.append(
                        {
                            "signal_priority": 1,
                            "signal_type": "contraction_retest",
                            "code": item.code,
                            "name": item.name,
                            "latest_date": latest["日期"].strftime("%Y-%m-%d"),
                            "entry_date": entry["日期"].strftime("%Y-%m-%d"),
                            "entry_age": latest_idx - best_entry,
                            "entry_close": round(float(retest["entry_close"]), 3),
                            "close": round(float(latest["close"]), 3),
                            "pct_chg": round(float(latest["pct_chg"]), 2),
                            "amount": float(latest["amount"]),
                            "amount_text": _format_amount(float(latest["amount"])),
                            "entry_pct_chg": round(float(entry["pct_chg"]), 2),
                            "entry_amount_text": _format_amount(float(entry["amount"])),
                            "entry_volume_ratio_20": round(float(entry.get("volume_ratio_20", 0)), 2),
                            "entry_amount_ratio_20": round(float(entry.get("amount_ratio_20", 0)), 2),
                            "volume_10_vs_entry": round(float(ctxd["volume_10_vs_entry"]), 3),
                            "amount_10_vs_entry": round(float(ctxd["amount_10_vs_entry"]), 3),
                            "ret_since_entry": round(float(ctxd["ret_since_entry"]) * 100, 2),
                            "max_ret_since_entry": round(float(ctxd["max_ret_since_entry"]) * 100, 2),
                            "post_entry_min_ret": round(float(retest["post_entry_min_ret"]) * 100, 2),
                            "post_entry_min_date": retest["post_entry_min_date"],
                            "entry_close_retest_gap": round(float(retest["entry_close_retest_gap"]) * 100, 2),
                            "recovered_from_post_entry_low": round(
                                float(retest["recovered_from_post_entry_low"]) * 100, 2
                            ),
                            "entry_close_retest_signal": bool(retest["entry_close_retest_signal"]),
                            "breakout_hold_signal": False,
                            "close_to_high_60": round(float(ctxd["latest_close_to_high_60"]) * 100, 2),
                            "model_prob": round(float(prob) * 100, 2),
                            "rule_score": rule,
                            "final_score": round(final, 2),
                            "reason": reason_text(work, best_entry, latest_idx),
                        }
                    )

        if breakout_enabled:
            breakout_entry = find_breakout_pullback_hold_entry(work, latest_idx, args, min_date)
            if breakout_entry is not None:
                entry = work.iloc[breakout_entry]
                info = breakout_pullback_hold_features(work, breakout_entry, latest_idx)
                rule = breakout_pullback_hold_rule_score(work, breakout_entry, latest_idx)
                reasons = [
                    f"{latest_idx - breakout_entry}天前爆量突破",
                    f"突破量{float(entry.get('volume_ratio_20', 0)):.1f}倍",
                    "回踩不破爆量低点",
                    f"自回踩低点修复{float(info['breakout_recovery_from_pullback']) * 100:.1f}%",
                ]
                if float(latest.get("above_ma20", 0)) == 1 and float(latest.get("above_ma60", 0)) == 1:
                    reasons.append("站上20/60日线")
                if float(latest.get("close_to_high_60", 0)) >= -0.08:
                    reasons.append("接近60日高点")
                stock_rows.append(
                    {
                        "signal_priority": 1,
                        "signal_type": "breakout_hold",
                        "code": item.code,
                        "name": item.name,
                        "latest_date": latest["日期"].strftime("%Y-%m-%d"),
                        "entry_date": entry["日期"].strftime("%Y-%m-%d"),
                        "entry_age": latest_idx - breakout_entry,
                        "entry_close": round(float(info["entry_close"]), 3),
                        "entry_low": round(float(info["entry_low"]), 3),
                        "close": round(float(latest["close"]), 3),
                        "pct_chg": round(float(latest["pct_chg"]), 2),
                        "amount": float(latest["amount"]),
                        "amount_text": _format_amount(float(latest["amount"])),
                        "entry_pct_chg": round(float(entry["pct_chg"]), 2),
                        "entry_amount_text": _format_amount(float(entry["amount"])),
                        "entry_volume_ratio_20": round(float(entry.get("volume_ratio_20", 0)), 2),
                        "entry_amount_ratio_20": round(float(entry.get("amount_ratio_20", 0)), 2),
                        "volume_10_vs_entry": np.nan,
                        "amount_10_vs_entry": np.nan,
                        "ret_since_entry": round(float(info["breakout_ret_since_entry"]) * 100, 2),
                        "max_ret_since_entry": np.nan,
                        "post_entry_min_ret": np.nan,
                        "post_entry_min_date": "",
                        "entry_close_retest_gap": np.nan,
                        "recovered_from_post_entry_low": round(
                            float(info["breakout_recovery_from_pullback"]) * 100, 2
                        ),
                        "entry_close_retest_signal": False,
                        "breakout_entry_score": round(float(info["breakout_entry_score"]), 2),
                        "breakout_hold_signal": bool(info["breakout_hold_signal"]),
                        "pullback_low_date": info["pullback_low_date"],
                        "pullback_low_vs_entry_low": round(float(info["pullback_low_vs_entry_low"]) * 100, 2),
                        "pullback_close_vs_entry_close": round(
                            float(info["pullback_close_vs_entry_close"]) * 100, 2
                        ),
                        "breakout_ret_since_entry": round(float(info["breakout_ret_since_entry"]) * 100, 2),
                        "breakout_recovery_from_pullback": round(
                            float(info["breakout_recovery_from_pullback"]) * 100, 2
                        ),
                        "close_to_high_60": round(float(latest.get("close_to_high_60", 0)) * 100, 2),
                        "model_prob": np.nan,
                        "rule_score": rule,
                        "final_score": rule,
                        "reason": "、".join(reasons[:7]),
                    }
                )

        if stock_rows:
            rows.append(
                max(
                    stock_rows,
                    key=lambda row: (
                        int(row.get("signal_priority", 0) or 0),
                        float(row.get("final_score", 0) or 0),
                        float(row.get("amount", 0) or 0),
                    ),
                )
            )
        if i % args.progress_every == 0:
            print(f"[{_now_text()}] scoring progress {i}/{len(histories)}, candidates={len(rows)}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["final_score", "amount"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def run(args: argparse.Namespace) -> Dict[str, object]:
    ensure_torch()
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    min_date = pd.Timestamp(datetime.now() - timedelta(days=args.strategy_days))
    manual_events = load_manual_label_events(args)

    print(f"[{_now_text()}] loading spot universe")
    spot = load_spot_universe(Path(args.spot_cache_dir), cache_hours=args.spot_cache_hours, refresh=args.refresh_spot)
    items = make_universe(
        spot,
        min_amount=args.min_amount,
        max_stocks=args.max_stocks,
        include_bj=args.include_bj,
        main_board_only=args.main_board_only,
    )
    if args.exclude_chinext:
        items = [item for item in items if not item.code.startswith(("300", "301"))]
    if manual_events:
        items = add_manual_items(items, manual_events)
    pd.DataFrame([item.__dict__ for item in items]).to_csv(out_dir / "universe.csv", index=False, encoding="utf-8-sig")
    print(f"[{_now_text()}] universe after filters: {len(items)} stocks")
    if manual_events:
        print(f"[{_now_text()}] loaded manual label events: {len(manual_events)}")
    histories = load_histories(items, args, out_dir)

    positives: List[ContractionSample] = []
    negatives: List[ContractionSample] = []
    events: List[Dict[str, object]] = []
    print(f"[{_now_text()}] building five-year contraction samples since {min_date.date()}")
    for i, (code, (item, hist)) in enumerate(histories.items(), start=1):
        ps, ns, es = detect_for_stock(item, hist, args, min_date)
        positives.extend(ps)
        negatives.extend(ns)
        events.extend(es)
        if i % args.progress_every == 0 or i == len(histories):
            print(f"[{_now_text()}] sample progress {i}/{len(histories)}, positives={len(positives)}, negatives={len(negatives)}, events={len(events)}")
    manual_positive_count = 0
    manual_event_count = 0
    if manual_events:
        manual_by_code: Dict[str, List[Dict[str, object]]] = {}
        for event in manual_events:
            manual_by_code.setdefault(str(event["code"]), []).append(event)
        for code, event_list in manual_by_code.items():
            if code not in histories:
                print(f"[{_now_text()}] manual label skipped, no history: {code}")
                continue
            item, hist = histories[code]
            ps, es = manual_label_samples_for_stock(item, hist, event_list, args, min_date)
            positives.extend(ps)
            events.extend(es)
            manual_positive_count += len(ps)
            manual_event_count += len(es)
        if manual_positive_count > 0 and args.manual_label_repeat > 1:
            manual_samples = positives[-manual_positive_count:]
            positives.extend(manual_samples * (args.manual_label_repeat - 1))
        print(f"[{_now_text()}] manual labels added: positives={manual_positive_count}, events={manual_event_count}")
    if not positives:
        raise RuntimeError("No positive samples found; relax month multiple or contraction thresholds")
    rng = random.Random(args.random_state)
    target_neg = min(len(negatives), max(args.min_negative_samples, len(positives) * args.negative_ratio))
    if len(negatives) > target_neg:
        negatives = rng.sample(negatives, target_neg)
    samples = positives + negatives
    rng.shuffle(samples)
    _, _, _, meta = build_arrays(samples)
    meta.to_csv(out_dir / "contraction_training_samples.csv", index=False, encoding="utf-8-sig")
    events_df = pd.DataFrame(events)
    events_df.to_csv(out_dir / "contraction_entry_run_events.csv", index=False, encoding="utf-8-sig")
    print(f"[{_now_text()}] training model: samples={len(samples)}, positives={len(positives)}, negatives={len(negatives)}")
    model, metrics, norm = train_model(samples, args)
    print(f"[{_now_text()}] scoring current candidates")
    candidates = score_current(histories, model, norm, args, min_date)
    candidates.to_csv(out_dir / "contraction_current_candidates.csv", index=False, encoding="utf-8-sig")
    summary = {
        "run_time": datetime.now().isoformat(),
        "definition": {
            "strategy_days": args.strategy_days,
            "lookback_years": args.lookback_years,
            "raw_sequence_days": args.seq_len,
            "long_context": "compressed multi-year price/volume/amount position and dry-up features",
            "entry": "3x volume bullish candle",
            "entry_ratio_basis": "max(current-day rolling ratio, current volume divided by previous rolling average)",
            "post_entry": "volume/amount contraction",
            "run_search_days": args.max_entry_to_run_days,
            "target": f"{args.month_multiple}x within {args.month_days} trading days from main-run start",
            "confirmed_run_start": bool(args.allow_confirmed_run_start),
            "main_board_only": bool(args.main_board_only),
            "manual_labels": str(args.manual_label_file),
            "positive_samples": "contracted pre-run days plus user-confirmed watch/trigger samples after entry",
            "entry_close_breakout": "after contraction, MACD stays above zero and a bullish candle closes above entry close",
        },
        "universe_size": len(items),
        "history_ok": len(histories),
        "events": len(events),
        "positive_samples": len(positives),
        "manual_events": manual_event_count,
        "manual_positive_samples": manual_positive_count,
        "manual_label_repeat": args.manual_label_repeat,
        "negative_samples": len(negatives),
        "training_samples": len(samples),
        "candidate_rows": len(candidates),
        "metrics": metrics,
        "args": vars(args),
    }
    (out_dir / "contraction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 五年内：三倍爆量阳 -> 缩量 -> 一月主升事件 Top ===")
    if events_df.empty:
        print("(empty)")
    else:
        cols = ["code", "name", "entry_date", "run_date", "entry_to_run_days", "entry_pct_chg", "entry_volume_ratio_20", "entry_amount_ratio_20", "month_run_multiple"]
        print(events_df.sort_values(["entry_date", "month_run_multiple"], ascending=[False, False])[cols].head(args.print_top).to_string(index=False))
    print("\n=== 当前可能迎来主升候选 Top ===")
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
            "entry_age",
            "pct_chg",
            "amount_text",
            "entry_volume_ratio_20",
            "volume_10_vs_entry",
            "amount_10_vs_entry",
            "ret_since_entry",
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
        print(candidates[cols].head(args.print_top).to_string(index=False))
    print(f"\n[{_now_text()}] outputs written under {out_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data_cache/volume_contraction_5y")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--spot-cache-dir", default="data_cache/main_uptrend")
    parser.add_argument("--strategy-days", type=int, default=1825)
    parser.add_argument("--lookback-years", type=int, default=6)
    parser.add_argument("--min-amount", type=float, default=0)
    parser.add_argument("--candidate-min-amount", type=float, default=50_000_000)
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--include-bj", action="store_true")
    parser.add_argument("--main-board-only", action="store_true", default=True)
    parser.add_argument("--include-non-main-board", dest="main_board_only", action="store_false")
    parser.add_argument("--exclude-chinext", action="store_true")
    parser.add_argument("--spot-cache-hours", type=float, default=2.0)
    parser.add_argument("--refresh-spot", action="store_true")
    parser.add_argument("--refresh-history", action="store_true")
    parser.add_argument("--cache-days", type=float, default=1.0)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=400)

    parser.add_argument("--entry-volume-ratio", type=float, default=3.0)
    parser.add_argument("--entry-amount-ratio", type=float, default=2.5)
    parser.add_argument("--entry-min-ret", type=float, default=0.03)
    parser.add_argument("--entry-min-body", type=float, default=0.018)
    parser.add_argument("--entry-gap-min-ret", type=float, default=0.04)
    parser.add_argument("--entry-gap-min-gap", type=float, default=0.025)
    parser.add_argument("--entry-gap-min-body", type=float, default=0.006)
    parser.add_argument("--entry-min-turnover", type=float, default=2.5)
    parser.add_argument("--entry-max-ret20", type=float, default=0.45)
    parser.add_argument("--month-days", type=int, default=20)
    parser.add_argument("--month-multiple", type=float, default=2.0)
    parser.add_argument("--run-start-min-ret", type=float, default=0.035)
    parser.add_argument("--run-start-min-body", type=float, default=0.015)
    parser.add_argument("--run-start-ret3", type=float, default=0.12)
    parser.add_argument("--run-start-volume-ratio", type=float, default=1.2)
    parser.add_argument("--run-start-amount-ratio", type=float, default=1.2)
    parser.add_argument("--second-volume-ratio", type=float, default=3.0)
    parser.add_argument("--second-amount-ratio", type=float, default=2.5)
    parser.add_argument("--second-min-ret", type=float, default=0.03)
    parser.add_argument("--second-min-body", type=float, default=0.015)
    parser.add_argument("--allow-confirmed-run-start", action="store_true")
    parser.add_argument("--manual-label-file", default="data_cache/manual_uptrend_labels.json")
    parser.add_argument("--manual-label-repeat", type=int, default=1)
    parser.add_argument("--manual-loss-weight", type=float, default=8.0)
    parser.add_argument("--disable-manual-labels", action="store_true")
    parser.add_argument("--min-contract-days", type=int, default=5)
    parser.add_argument("--max-entry-to-run-days", type=int, default=250)
    parser.add_argument("--lead-min", type=int, default=1)
    parser.add_argument("--lead-max", type=int, default=10)
    parser.add_argument("--positive-stride", type=int, default=5)
    parser.add_argument("--max-positive-samples-per-event", type=int, default=35)
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--max-volume5-vs-entry", type=float, default=0.62)
    parser.add_argument("--max-volume10-vs-entry", type=float, default=0.55)
    parser.add_argument("--max-amount10-vs-entry", type=float, default=0.68)
    parser.add_argument("--max-latest-volume-vs-entry", type=float, default=0.70)
    parser.add_argument("--max-pre-run-ret", type=float, default=0.70)
    parser.add_argument("--max-current-ret-since-entry", type=float, default=0.55)
    parser.add_argument("--max-sample-ret20", type=float, default=0.35)
    parser.add_argument("--latest-started-ret", type=float, default=0.055)
    parser.add_argument("--latest-started-body", type=float, default=0.035)
    parser.add_argument("--require-entry-close-retest", action="store_true")
    parser.add_argument("--disable-entry-close-breakout-signal", action="store_true")
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
    parser.add_argument("--disable-macd-water-filter", action="store_true")
    parser.add_argument("--macd-waterline-min", type=float, default=0.0)
    parser.add_argument("--macd-require-dea-above-water", action="store_true", default=True)
    parser.add_argument("--disable-breakout-hold-signal", action="store_true")
    parser.add_argument("--breakout-hold-lookback-days", type=int, default=45)
    parser.add_argument("--breakout-hold-min-days", type=int, default=1)
    parser.add_argument("--negative-entry-sample-rate", type=float, default=0.65)
    parser.add_argument("--negative-stride", type=int, default=5)
    parser.add_argument("--max-negatives-per-entry", type=int, default=2)
    parser.add_argument("--max-negatives-per-stock", type=int, default=12)
    parser.add_argument("--negative-ratio", type=int, default=6)
    parser.add_argument("--min-negative-samples", type=int, default=300)

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--print-top", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
