#!/usr/bin/env python
"""Backtest daily Top20 candidates for a historical month.

The script replays each trading day with bars truncated at that day, ranks the
current model candidates, and then attaches forward 1/2/3-trading-day returns.
It writes CSV files plus a compact JSON payload consumed by the web dashboard.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from latent_uptrend_model import TemporalAttentionNet
from main_uptrend_model import StockItem, _format_amount, add_features, is_hs_main_board_code, normalize_hist
from volume_contraction_breakout_model import (
    CONTEXT_COLS,
    SEQ_COLS,
    breakout_pullback_hold_features,
    breakout_pullback_hold_rule_score,
    contraction_context,
    contraction_rule_score,
    entry_close_breakout_reason_text,
    find_breakout_pullback_hold_entry,
    find_entry_close_breakout_signal,
    is_contracted_not_running,
    make_sequence,
    post_entry_retest_features,
    reason_text,
    sanitize,
    triple_volume_bull_mask,
)

try:
    import torch
except ImportError as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def ensure_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for model inference") from TORCH_IMPORT_ERROR


def normalize_code(value: object) -> str:
    return str(value).strip().zfill(6)[-6:]


def date_text(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return 0.0
    return float(numerator / denominator)


def clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


@dataclass
class FastFeature:
    item: StockItem
    feat: pd.DataFrame
    dates: np.ndarray
    date_texts: np.ndarray
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    amount: np.ndarray
    volume5: np.ndarray
    volume10: np.ndarray
    amount10: np.ndarray
    day_ret: np.ndarray
    body_pct: np.ndarray
    pct_chg: np.ndarray
    turnover: np.ndarray
    ret20: np.ndarray
    volume_ratio20: np.ndarray
    amount_ratio20: np.ndarray
    macd_dif: np.ndarray
    macd_dea: np.ndarray
    above_ma20: np.ndarray
    above_ma60: np.ndarray
    close_to_high60: np.ndarray
    triple_mask: np.ndarray


def load_universe(path: Path, main_board_only: bool = True) -> List[StockItem]:
    df = pd.read_csv(path, dtype={"code": str, "代码": str})
    if "code" not in df.columns and "代码" in df.columns:
        df = df.rename(columns={"代码": "code", "名称": "name"})
    if "name" not in df.columns:
        df["name"] = df["code"]
    df["code"] = df["code"].map(normalize_code)
    df["name"] = df["name"].fillna("").astype(str)
    df = df[~df["name"].str.contains("ST|退|N|C", case=False, na=False)].copy()
    if main_board_only:
        df = df[df["code"].map(is_hs_main_board_code)].copy()
    df = df.drop_duplicates("code")
    return [StockItem(code=row.code, name=row.name) for row in df.itertuples(index=False)]


def add_missing_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "entry_volume_ratio": 3.0,
        "entry_amount_ratio": 2.5,
        "entry_min_ret": 0.03,
        "entry_min_body": 0.018,
        "entry_gap_min_ret": 0.04,
        "entry_gap_min_gap": 0.025,
        "entry_gap_min_body": 0.006,
        "entry_min_turnover": 2.5,
        "entry_max_ret20": 0.45,
        "min_contract_days": 5,
        "max_entry_to_run_days": 250,
        "max_volume5_vs_entry": 0.62,
        "max_volume10_vs_entry": 0.55,
        "max_amount10_vs_entry": 0.68,
        "max_latest_volume_vs_entry": 0.70,
        "max_pre_run_ret": 0.70,
        "max_current_ret_since_entry": 0.55,
        "max_sample_ret20": 0.35,
        "latest_started_ret": 0.055,
        "latest_started_body": 0.035,
        "seq_len": 240,
        "candidate_min_amount": 50_000_000,
        "disable_entry_close_breakout_signal": False,
        "entry_close_breakout_recent_days": 12,
        "entry_close_breakout_min_body": 0.006,
        "entry_close_breakout_min_ret": 0.0,
        "entry_close_breakout_close_buffer": 0.0,
        "entry_close_breakout_prior_tolerance": 0.01,
        "entry_close_breakout_max_trigger_volume_vs_entry": 0.95,
        "entry_close_breakout_max_trigger_amount_vs_entry": 1.20,
        "entry_close_breakout_max_volume10_vs_entry": 0.62,
        "entry_close_breakout_max_latest_volume_vs_entry": 0.70,
        "entry_close_breakout_max_amount10_vs_entry": 0.72,
        "entry_close_breakout_max_pre_breakout_ret": 0.55,
        "entry_close_breakout_max_ret_since_entry": 0.75,
        "entry_close_breakout_require_macd_to_latest": True,
        "disable_macd_water_filter": False,
        "macd_waterline_min": 0.0,
        "macd_require_dea_above_water": True,
        "enable_breakout_hold_signal": True,
        "breakout_hold_lookback_days": 45,
        "breakout_hold_min_days": 1,
        "device": "cpu",
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def load_checkpoint(model_path: Path, device_name: str) -> Tuple[TemporalAttentionNet, Dict[str, np.ndarray], argparse.Namespace, torch.device]:
    ensure_torch()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_args = add_missing_args(argparse.Namespace(**checkpoint.get("args", {})))
    model_args.device = device_name
    ctx_cols = checkpoint.get("context_cols", CONTEXT_COLS)
    seq_cols = checkpoint.get("seq_cols", SEQ_COLS)
    model = TemporalAttentionNet(
        len(seq_cols),
        len(ctx_cols),
        d_model=int(checkpoint.get("args", {}).get("d_model", 96)),
        dropout=float(checkpoint.get("args", {}).get("dropout", 0.18)),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    norm = {
        "seq_mean": np.asarray(checkpoint["seq_mean"]),
        "seq_std": np.asarray(checkpoint["seq_std"]),
        "ctx_mean": np.asarray(checkpoint["ctx_mean"]).reshape(1, -1),
        "ctx_std": np.asarray(checkpoint["ctx_std"]).reshape(1, -1),
    }
    return model, norm, model_args, device


def load_feature_cache(items: Iterable[StockItem], history_dir: Path, min_rows: int) -> Dict[str, Tuple[StockItem, pd.DataFrame]]:
    cache: Dict[str, Tuple[StockItem, pd.DataFrame]] = {}
    for item in items:
        path = history_dir / f"{item.code}.csv"
        if not path.exists():
            continue
        try:
            hist = normalize_hist(pd.read_csv(path))
            feat = add_features(hist).reset_index(drop=True)
        except Exception:
            continue
        if len(feat) >= min_rows:
            cache[item.code] = (item, feat)
    return cache


def numeric_array(feat: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in feat.columns:
        return np.full(len(feat), default, dtype=float)
    return pd.to_numeric(feat[column], errors="coerce").fillna(default).to_numpy(dtype=float)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(np.where(values > 0, values, np.nan), dtype=float)
    return series.rolling(window, min_periods=1).mean().fillna(0.0).to_numpy(dtype=float)


def build_fast_feature_cache(
    feature_cache: Dict[str, Tuple[StockItem, pd.DataFrame]],
    args: argparse.Namespace,
) -> Dict[str, FastFeature]:
    fast: Dict[str, FastFeature] = {}
    for code, (item, feat) in feature_cache.items():
        dates = pd.to_datetime(feat["日期"]).dt.normalize().to_numpy(dtype="datetime64[D]")
        date_texts = pd.to_datetime(feat["日期"]).dt.strftime("%Y-%m-%d").to_numpy()
        volume = numeric_array(feat, "volume")
        amount = numeric_array(feat, "amount")
        try:
            triple_mask = triple_volume_bull_mask(feat, args)
        except Exception:
            triple_mask = np.zeros(len(feat), dtype=bool)
        fast[code] = FastFeature(
            item=item,
            feat=feat,
            dates=dates,
            date_texts=date_texts,
            close=numeric_array(feat, "close"),
            high=numeric_array(feat, "最高"),
            low=numeric_array(feat, "最低"),
            volume=volume,
            amount=amount,
            volume5=rolling_mean(volume, 5),
            volume10=rolling_mean(volume, 10),
            amount10=rolling_mean(amount, 10),
            day_ret=numeric_array(feat, "day_ret"),
            body_pct=numeric_array(feat, "body_pct"),
            pct_chg=numeric_array(feat, "pct_chg"),
            turnover=numeric_array(feat, "turnover"),
            ret20=numeric_array(feat, "ret_20"),
            volume_ratio20=numeric_array(feat, "volume_ratio_20"),
            amount_ratio20=numeric_array(feat, "amount_ratio_20"),
            macd_dif=numeric_array(feat, "macd_dif"),
            macd_dea=numeric_array(feat, "macd_dea"),
            above_ma20=numeric_array(feat, "above_ma20"),
            above_ma60=numeric_array(feat, "above_ma60"),
            close_to_high60=numeric_array(feat, "close_to_high_60"),
            triple_mask=np.asarray(triple_mask, dtype=bool),
        )
    return fast


def latest_index_asof(bundle: FastFeature, day: pd.Timestamp) -> Optional[int]:
    target = np.datetime64(day.strftime("%Y-%m-%d"), "D")
    idx = int(np.searchsorted(bundle.dates, target, side="right") - 1)
    if idx < 0 or bundle.dates[idx] != target:
        return None
    return idx


def trading_days(
    feature_cache: Dict[str, Tuple[StockItem, pd.DataFrame]],
    start: str,
    end: str,
    min_coverage: float = 0.3,
    min_count: int = 20,
) -> List[pd.Timestamp]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    counts: Dict[pd.Timestamp, int] = {}
    for _, feat in feature_cache.values():
        dates = pd.to_datetime(feat["日期"])
        for value in dates[(dates >= start_ts) & (dates <= end_ts)]:
            key = pd.Timestamp(value).normalize()
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return []
    threshold = max(int(min_count), int(len(feature_cache) * float(min_coverage)))
    return sorted([day for day, count in counts.items() if count >= threshold])


def prefix_asof(feat: pd.DataFrame, day: pd.Timestamp) -> Optional[pd.DataFrame]:
    dates = pd.to_datetime(feat["日期"])
    idxs = np.flatnonzero((dates <= day).to_numpy())
    if len(idxs) == 0:
        return None
    idx = int(idxs[-1])
    if pd.Timestamp(dates.iloc[idx]).normalize() != day.normalize():
        return None
    return feat.iloc[: idx + 1]


def infer_prob(
    model: TemporalAttentionNet,
    norm: Dict[str, np.ndarray],
    device: torch.device,
    seq: np.ndarray,
    ctx: np.ndarray,
) -> float:
    ensure_torch()
    ctx_mean = norm["ctx_mean"]
    ctx_std = norm["ctx_std"]
    if ctx.shape[0] > ctx_mean.shape[1]:
        ctx = ctx[: ctx_mean.shape[1]]
    elif ctx.shape[0] < ctx_mean.shape[1]:
        ctx = np.pad(ctx, (0, ctx_mean.shape[1] - ctx.shape[0]), constant_values=0)
    seq_n = sanitize((seq[None, :, :] - norm["seq_mean"]) / norm["seq_std"])
    ctx_n = sanitize((ctx[None, :] - ctx_mean) / ctx_std)
    with torch.no_grad():
        value = torch.sigmoid(
            model(torch.from_numpy(seq_n).float().to(device), torch.from_numpy(ctx_n).float().to(device))
        ).detach().cpu().numpy()[0]
    return float(value) if np.isfinite(value) else 0.0


def fast_pre_contracted(
    bundle: FastFeature,
    entry_idx: int,
    check_idx: int,
    args: argparse.Namespace,
    strict_breakout: bool,
) -> Tuple[bool, Dict[str, float]]:
    if check_idx - entry_idx < int(getattr(args, "min_contract_days", 5)):
        return False, {}
    entry_close = float(bundle.close[entry_idx])
    entry_volume = float(bundle.volume[entry_idx])
    entry_amount = float(bundle.amount[entry_idx])
    close_now = float(bundle.close[check_idx])
    if entry_close <= 0 or entry_volume <= 0 or entry_amount <= 0:
        return False, {}
    max_close = float(np.nanmax(bundle.close[entry_idx : check_idx + 1]))
    ctx = {
        "volume_5_vs_entry": safe_ratio(float(bundle.volume5[check_idx]), entry_volume),
        "volume_10_vs_entry": safe_ratio(float(bundle.volume10[check_idx]), entry_volume),
        "amount_10_vs_entry": safe_ratio(float(bundle.amount10[check_idx]), entry_amount),
        "latest_volume_vs_entry": safe_ratio(float(bundle.volume[check_idx]), entry_volume),
        "latest_amount_vs_entry": safe_ratio(float(bundle.amount[check_idx]), entry_amount),
        "ret_since_entry": close_now / entry_close - 1.0,
        "max_ret_since_entry": max_close / entry_close - 1.0,
    }
    if strict_breakout:
        max_volume10 = float(getattr(args, "entry_close_breakout_max_volume10_vs_entry", args.max_volume10_vs_entry))
        max_latest_volume = float(
            getattr(args, "entry_close_breakout_max_latest_volume_vs_entry", args.max_latest_volume_vs_entry)
        )
        max_amount10 = float(getattr(args, "entry_close_breakout_max_amount10_vs_entry", args.max_amount10_vs_entry))
        max_pre_ret = float(getattr(args, "entry_close_breakout_max_pre_breakout_ret", args.max_pre_run_ret))
    else:
        max_volume10 = float(args.max_volume10_vs_entry)
        max_latest_volume = float(args.max_latest_volume_vs_entry)
        max_amount10 = float(args.max_amount10_vs_entry)
        max_pre_ret = float(args.max_pre_run_ret)
    ok = bool(
        ctx["volume_5_vs_entry"] <= float(args.max_volume5_vs_entry)
        and ctx["volume_10_vs_entry"] <= max_volume10
        and ctx["amount_10_vs_entry"] <= max_amount10
        and ctx["latest_volume_vs_entry"] <= max_latest_volume
        and ctx["max_ret_since_entry"] <= max_pre_ret
        and ctx["ret_since_entry"] <= float(args.max_current_ret_since_entry)
    )
    return ok, ctx


def fast_macd_water(bundle: FastFeature, entry_idx: int, latest_idx: int, args: argparse.Namespace) -> Dict[str, object]:
    dif = bundle.macd_dif[entry_idx : latest_idx + 1]
    dea = bundle.macd_dea[entry_idx : latest_idx + 1]
    if len(dif) == 0 or len(dea) == 0 or np.isnan(dif).any() or np.isnan(dea).any():
        return {"ok": False, "min_dif": 0.0, "min_dea": 0.0}
    min_dif = float(np.nanmin(dif))
    min_dea = float(np.nanmin(dea))
    line = float(getattr(args, "macd_waterline_min", 0.0))
    require_dea = bool(getattr(args, "macd_require_dea_above_water", True))
    return {"ok": bool(min_dif >= line and (min_dea >= line if require_dea else True)), "min_dif": min_dif, "min_dea": min_dea}


def fast_entry_breakout_score(
    bundle: FastFeature,
    entry_idx: int,
    trigger_idx: int,
    latest_idx: int,
    info: Dict[str, float],
) -> float:
    score = 42.0
    score += 10.0 * clip01((float(bundle.volume_ratio20[entry_idx]) - 3.0) / 3.5)
    score += 8.0 * clip01((float(bundle.amount_ratio20[entry_idx]) - 2.5) / 4.0)
    score += 14.0 * clip01((0.62 - float(info["pre_volume_10_vs_entry"])) / 0.62)
    score += 10.0 * clip01((0.70 - float(info["pre_amount_10_vs_entry"])) / 0.70)
    score += 10.0 * clip01((float(info["entry_close_breakout_gap"]) - 0.002) / 0.08)
    score += 8.0 * clip01((0.95 - float(info["trigger_volume_vs_entry"])) / 0.95)
    score += 8.0 * math.exp(-((latest_idx - trigger_idx) / 8.0) ** 2)
    score += 5.0 * clip01(float(info["macd_min_dif"]) / 0.18)
    score += 5.0 * clip01(float(info["macd_min_dea"]) / 0.12)
    score += 4.0 * clip01((float(bundle.above_ma20[latest_idx]) + float(bundle.above_ma60[latest_idx])) / 2.0)
    score -= 12.0 * clip01((float(info["latest_ret_since_entry"]) - 0.65) / 0.35)
    return round(max(0.0, min(100.0, score)), 2)


def fast_entry_close_breakout(
    bundle: FastFeature,
    latest_idx: int,
    args: argparse.Namespace,
    min_day: np.datetime64,
) -> Optional[Dict[str, object]]:
    if bool(getattr(args, "disable_entry_close_breakout_signal", False)):
        return None
    entry_min = max(0, latest_idx - int(getattr(args, "max_entry_to_run_days", 250)))
    entry_max = latest_idx - max(1, int(getattr(args, "min_contract_days", 5)))
    if entry_max < entry_min:
        return None
    recent_days = int(getattr(args, "entry_close_breakout_recent_days", 12))
    min_body = float(getattr(args, "entry_close_breakout_min_body", 0.006))
    min_ret = float(getattr(args, "entry_close_breakout_min_ret", 0.0))
    close_buffer = float(getattr(args, "entry_close_breakout_close_buffer", 0.0))
    prior_tolerance = float(getattr(args, "entry_close_breakout_prior_tolerance", 0.01))
    max_trigger_volume = float(getattr(args, "entry_close_breakout_max_trigger_volume_vs_entry", 0.95))
    max_trigger_amount = float(getattr(args, "entry_close_breakout_max_trigger_amount_vs_entry", 1.20))
    max_ret_since_entry = float(getattr(args, "entry_close_breakout_max_ret_since_entry", 0.75))
    macd_required = not bool(getattr(args, "disable_macd_water_filter", False))

    best: Optional[Dict[str, object]] = None
    entries = np.flatnonzero(bundle.triple_mask[entry_min : entry_max + 1]) + entry_min
    for entry_idx in entries:
        entry_idx = int(entry_idx)
        if bundle.dates[entry_idx] < min_day:
            continue
        entry_close = float(bundle.close[entry_idx])
        entry_volume = float(bundle.volume[entry_idx])
        entry_amount = float(bundle.amount[entry_idx])
        if entry_close <= 0 or entry_volume <= 0 or entry_amount <= 0:
            continue
        latest_ret_since_entry = float(bundle.close[latest_idx]) / entry_close - 1.0
        if latest_ret_since_entry > max_ret_since_entry:
            continue
        macd_end = latest_idx if bool(getattr(args, "entry_close_breakout_require_macd_to_latest", True)) else latest_idx
        macd = fast_macd_water(bundle, entry_idx, macd_end, args)
        if macd_required and not bool(macd["ok"]):
            continue
        trigger_min = max(entry_idx + max(1, int(getattr(args, "min_contract_days", 5))), latest_idx - recent_days)
        trigger_max = min(latest_idx, entry_idx + int(getattr(args, "max_entry_to_run_days", 250)))
        if trigger_max < trigger_min:
            continue
        for trigger_idx in range(trigger_min, trigger_max + 1):
            if bundle.close[trigger_idx] <= entry_close * (1.0 + close_buffer):
                continue
            if bundle.body_pct[trigger_idx] < min_body or bundle.day_ret[trigger_idx] < min_ret:
                continue
            prior = bundle.close[entry_idx + 1 : trigger_idx]
            prior_max_close = float(np.nanmax(prior)) if len(prior) else 0.0
            if prior_max_close > entry_close * (1.0 + prior_tolerance):
                continue
            trigger_volume_vs_entry = safe_ratio(float(bundle.volume[trigger_idx]), entry_volume)
            trigger_amount_vs_entry = safe_ratio(float(bundle.amount[trigger_idx]), entry_amount)
            if trigger_volume_vs_entry > max_trigger_volume or trigger_amount_vs_entry > max_trigger_amount:
                continue
            ok, ctx = fast_pre_contracted(bundle, entry_idx, trigger_idx - 1, args, strict_breakout=True)
            if not ok:
                continue
            info = {
                "trigger_volume_vs_entry": trigger_volume_vs_entry,
                "trigger_amount_vs_entry": trigger_amount_vs_entry,
                "pre_volume_10_vs_entry": float(ctx["volume_10_vs_entry"]),
                "pre_amount_10_vs_entry": float(ctx["amount_10_vs_entry"]),
                "entry_close_breakout_gap": float(bundle.close[trigger_idx]) / entry_close - 1.0,
                "latest_ret_since_entry": latest_ret_since_entry,
                "macd_min_dif": float(macd["min_dif"]),
                "macd_min_dea": float(macd["min_dea"]),
            }
            score = fast_entry_breakout_score(bundle, entry_idx, trigger_idx, latest_idx, info)
            row = {
                "signal_priority": 2,
                "signal_type": "entry_close_breakout",
                "code": bundle.item.code,
                "name": bundle.item.name,
                "latest_date": str(bundle.date_texts[latest_idx]),
                "entry_date": str(bundle.date_texts[entry_idx]),
                "trigger_date": str(bundle.date_texts[trigger_idx]),
                "entry_age": latest_idx - entry_idx,
                "trigger_age": latest_idx - trigger_idx,
                "close": round(float(bundle.close[latest_idx]), 3),
                "pct_chg": round(float(bundle.pct_chg[latest_idx]), 2),
                "amount": float(bundle.amount[latest_idx]),
                "amount_text": _format_amount(float(bundle.amount[latest_idx])),
                "entry_volume_ratio_20": round(float(bundle.volume_ratio20[entry_idx]), 2),
                "entry_amount_ratio_20": round(float(bundle.amount_ratio20[entry_idx]), 2),
                "trigger_volume_vs_entry": round(trigger_volume_vs_entry, 3),
                "trigger_amount_vs_entry": round(trigger_amount_vs_entry, 3),
                "volume_10_vs_entry": round(float(ctx["volume_10_vs_entry"]), 3),
                "amount_10_vs_entry": round(float(ctx["amount_10_vs_entry"]), 3),
                "entry_close_breakout_gap": round(float(info["entry_close_breakout_gap"]) * 100, 2),
                "macd_water_ok": bool(macd["ok"]),
                "macd_min_dif": round(float(macd["min_dif"]), 4),
                "macd_min_dea": round(float(macd["min_dea"]), 4),
                "model_prob": np.nan,
                "rule_score": score,
                "final_score": score,
                "reason": "、".join(
                    [
                        f"{latest_idx - entry_idx}天前三倍爆量阳",
                        f"进场量{float(bundle.volume_ratio20[entry_idx]):.1f}倍",
                        f"缩量至{float(ctx['volume_10_vs_entry']):.2f}",
                        f"{latest_idx - trigger_idx}天前阳线突破爆量收盘",
                        "MACD水上",
                    ]
                ),
            }
            if best is None or score > float(best["final_score"]):
                best = row
    return best


def fast_contraction_score(bundle: FastFeature, entry_idx: int, latest_idx: int, ctx: Dict[str, float]) -> float:
    entry_close = float(bundle.close[entry_idx])
    latest_close = float(bundle.close[latest_idx])
    post_close = bundle.close[entry_idx : latest_idx + 1]
    post_min = float(np.nanmin(post_close)) if len(post_close) else entry_close
    post_min_ret = post_min / entry_close - 1.0 if entry_close > 0 else 0.0
    recovered = latest_close / post_min - 1.0 if post_min > 0 else 0.0
    age = latest_idx - entry_idx

    breakout_score = 18 * math.exp(-((age - 45) / 28) ** 2)
    breakout_score += 16 * clip01((float(bundle.volume_ratio20[entry_idx]) - 3.0) / 4.5)
    breakout_score += 14 * clip01((float(bundle.amount_ratio20[entry_idx]) - 2.5) / 5.0)
    breakout_score += 18 * clip01((0.62 - ctx["volume_10_vs_entry"]) / 0.62)
    breakout_score += 12 * clip01((0.70 - ctx["amount_10_vs_entry"]) / 0.70)
    breakout_score += 10 * clip01((float(bundle.close_to_high60[latest_idx]) + 0.15) / 0.15)
    breakout_score += 6 * clip01((float(bundle.above_ma20[latest_idx]) + float(bundle.above_ma60[latest_idx])) / 2.0)
    breakout_score += 6 * clip01((0.65 - ctx["max_ret_since_entry"]) / 0.65)

    low_position_score = 18 * math.exp(-((age - 110) / 75) ** 2)
    low_position_score += 22 * clip01((0.62 - ctx["volume_10_vs_entry"]) / 0.62)
    low_position_score += 18 * clip01((0.68 - ctx["amount_10_vs_entry"]) / 0.68)
    low_position_score += 14 * clip01((0.45 - ctx["max_ret_since_entry"]) / 0.45)
    low_position_score += 10 * clip01((0.55 - abs(ctx["ret_since_entry"])) / 0.55)
    low_position_score += 8 * clip01((0.08 - float(bundle.ret20[latest_idx])) / 0.30)
    low_position_score += 8 * clip01((1.15 - float(bundle.volume_ratio20[latest_idx])) / 1.15)
    low_position_score += 8 * clip01((float(bundle.volume_ratio20[entry_idx]) - 2.5) / 3.5)
    low_position_score += 6 * clip01((float(bundle.amount_ratio20[entry_idx]) - 2.5) / 3.5)

    retest_score = 8 * clip01((-0.05 - post_min_ret) / 0.25)
    retest_score += 8 * clip01((recovered - 0.04) / 0.28)
    return round(min(100.0, max(breakout_score, low_position_score) + 0.45 * retest_score), 2)


def fast_contraction_retest(
    bundle: FastFeature,
    latest_idx: int,
    args: argparse.Namespace,
    min_day: np.datetime64,
) -> Optional[Dict[str, object]]:
    entry_min = max(int(args.seq_len), latest_idx - int(args.max_entry_to_run_days))
    entry_max = latest_idx - int(args.min_contract_days)
    if entry_max < entry_min:
        return None
    best: Optional[Dict[str, object]] = None
    entries = np.flatnonzero(bundle.triple_mask[entry_min : entry_max + 1]) + entry_min
    for entry_idx in entries:
        entry_idx = int(entry_idx)
        if bundle.dates[entry_idx] < min_day:
            continue
        if float(bundle.ret20[latest_idx]) > float(args.max_sample_ret20):
            continue
        if float(bundle.day_ret[latest_idx]) >= float(args.latest_started_ret) and float(bundle.body_pct[latest_idx]) >= float(args.latest_started_body):
            continue
        ok, ctx = fast_pre_contracted(bundle, entry_idx, latest_idx, args, strict_breakout=False)
        if not ok:
            continue
        score = fast_contraction_score(bundle, entry_idx, latest_idx, ctx)
        post_close = bundle.close[entry_idx : latest_idx + 1]
        entry_close = float(bundle.close[entry_idx])
        post_min_ret = float(np.nanmin(post_close)) / entry_close - 1.0 if len(post_close) and entry_close > 0 else 0.0
        row = {
            "signal_priority": 1,
            "signal_type": "contraction_retest",
            "code": bundle.item.code,
            "name": bundle.item.name,
            "latest_date": str(bundle.date_texts[latest_idx]),
            "entry_date": str(bundle.date_texts[entry_idx]),
            "trigger_date": "",
            "entry_age": latest_idx - entry_idx,
            "trigger_age": np.nan,
            "close": round(float(bundle.close[latest_idx]), 3),
            "pct_chg": round(float(bundle.pct_chg[latest_idx]), 2),
            "amount": float(bundle.amount[latest_idx]),
            "amount_text": _format_amount(float(bundle.amount[latest_idx])),
            "entry_volume_ratio_20": round(float(bundle.volume_ratio20[entry_idx]), 2),
            "entry_amount_ratio_20": round(float(bundle.amount_ratio20[entry_idx]), 2),
            "trigger_volume_vs_entry": np.nan,
            "volume_10_vs_entry": round(float(ctx["volume_10_vs_entry"]), 3),
            "amount_10_vs_entry": round(float(ctx["amount_10_vs_entry"]), 3),
            "entry_close_breakout_gap": np.nan,
            "macd_water_ok": np.nan,
            "macd_min_dif": np.nan,
            "macd_min_dea": np.nan,
            "model_prob": np.nan,
            "rule_score": score,
            "final_score": score,
            "post_entry_min_ret": round(post_min_ret * 100, 2),
            "reason": "、".join(
                [
                    f"{latest_idx - entry_idx}天前三倍爆量阳",
                    f"进场量{float(bundle.volume_ratio20[entry_idx]):.1f}倍",
                    f"10日量缩至{float(ctx['volume_10_vs_entry']):.2f}",
                    f"当前较爆量收盘{float(ctx['ret_since_entry']) * 100:.1f}%",
                ]
            ),
        }
        if best is None or score > float(best["final_score"]):
            best = row
    return best


def fast_score_one_asof(
    bundle: FastFeature,
    latest_idx: int,
    args: argparse.Namespace,
    min_day: np.datetime64,
) -> Optional[Dict[str, object]]:
    if float(bundle.amount[latest_idx]) < float(args.candidate_min_amount):
        return None
    rows = [
        fast_entry_close_breakout(bundle, latest_idx, args, min_day),
        fast_contraction_retest(bundle, latest_idx, args, min_day),
    ]
    rows = [row for row in rows if row is not None]
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            int(row.get("signal_priority", 0) or 0),
            float(row.get("final_score", 0) or 0),
            float(row.get("amount", 0) or 0),
        ),
    )


def score_one_asof(
    item: StockItem,
    work: pd.DataFrame,
    model: TemporalAttentionNet,
    norm: Dict[str, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
    min_date: pd.Timestamp,
) -> Optional[Dict[str, object]]:
    latest_idx = len(work) - 1
    latest = work.iloc[latest_idx]
    if float(latest.get("amount", 0)) < float(args.candidate_min_amount):
        return None
    rows: List[Dict[str, object]] = []

    entry_breakout = find_entry_close_breakout_signal(work, latest_idx, args, min_date)
    if entry_breakout is not None:
        entry_idx, trigger_idx, info, rule = entry_breakout
        entry = work.iloc[entry_idx]
        rows.append(
            {
                "signal_priority": 2,
                "signal_type": "entry_close_breakout",
                "code": item.code,
                "name": item.name,
                "latest_date": date_text(latest["日期"]),
                "entry_date": date_text(entry["日期"]),
                "trigger_date": info["trigger_date"],
                "entry_age": latest_idx - entry_idx,
                "trigger_age": latest_idx - trigger_idx,
                "close": round(float(latest["close"]), 3),
                "pct_chg": round(float(latest["pct_chg"]), 2),
                "amount": float(latest["amount"]),
                "amount_text": _format_amount(float(latest["amount"])),
                "entry_volume_ratio_20": round(float(entry.get("volume_ratio_20", 0)), 2),
                "trigger_volume_vs_entry": round(float(info["trigger_volume_vs_entry"]), 3),
                "volume_10_vs_entry": round(float(info["pre_volume_10_vs_entry"]), 3),
                "amount_10_vs_entry": round(float(info["pre_amount_10_vs_entry"]), 3),
                "entry_close_breakout_gap": round(float(info["entry_close_breakout_gap"]) * 100, 2),
                "macd_water_ok": bool(info["macd_water_ok"]),
                "macd_min_dif": round(float(info["macd_min_dif"]), 4),
                "macd_min_dea": round(float(info["macd_min_dea"]), 4),
                "model_prob": np.nan,
                "rule_score": rule,
                "final_score": rule,
                "reason": entry_close_breakout_reason_text(work, entry_idx, trigger_idx, latest_idx, info),
            }
        )

    entry_mask = triple_volume_bull_mask(work, args)
    dates = pd.to_datetime(work["日期"])
    entry_min = max(int(args.seq_len), latest_idx - int(args.max_entry_to_run_days))
    entry_max = latest_idx - int(args.min_contract_days)
    if entry_max >= entry_min:
        entries = np.flatnonzero(entry_mask[entry_min : entry_max + 1]) + entry_min
        entries = [
            int(x)
            for x in entries
            if dates.iloc[int(x)] >= min_date and is_contracted_not_running(work, int(x), latest_idx, args)
        ]
        if entries:
            best_entry = max(
                entries,
                key=lambda x: float(work.iloc[x].get("volume_ratio_20", 0))
                + float(work.iloc[x].get("amount_ratio_20", 0)),
            )
            seq = make_sequence(work, latest_idx, int(args.seq_len))
            if seq is not None:
                ctx = contraction_context(work, best_entry, latest_idx)
                retest = post_entry_retest_features(work, best_entry, latest_idx)
                prob = infer_prob(model, norm, device, seq, ctx)
                rule = contraction_rule_score(work, best_entry, latest_idx)
                final = 0.68 * prob * 100 + 0.32 * rule
                entry = work.iloc[best_entry]
                ctxd = dict(zip(CONTEXT_COLS, ctx))
                rows.append(
                    {
                        "signal_priority": 1,
                        "signal_type": "contraction_retest",
                        "code": item.code,
                        "name": item.name,
                        "latest_date": date_text(latest["日期"]),
                        "entry_date": date_text(entry["日期"]),
                        "trigger_date": "",
                        "entry_age": latest_idx - best_entry,
                        "trigger_age": np.nan,
                        "close": round(float(latest["close"]), 3),
                        "pct_chg": round(float(latest["pct_chg"]), 2),
                        "amount": float(latest["amount"]),
                        "amount_text": _format_amount(float(latest["amount"])),
                        "entry_volume_ratio_20": round(float(entry.get("volume_ratio_20", 0)), 2),
                        "trigger_volume_vs_entry": np.nan,
                        "volume_10_vs_entry": round(float(ctxd["volume_10_vs_entry"]), 3),
                        "amount_10_vs_entry": round(float(ctxd["amount_10_vs_entry"]), 3),
                        "entry_close_breakout_gap": np.nan,
                        "macd_water_ok": np.nan,
                        "macd_min_dif": np.nan,
                        "macd_min_dea": np.nan,
                        "model_prob": round(prob * 100, 2),
                        "rule_score": rule,
                        "final_score": round(final, 2),
                        "post_entry_min_ret": round(float(retest["post_entry_min_ret"]) * 100, 2),
                        "reason": reason_text(work, best_entry, latest_idx),
                    }
                )

    if bool(getattr(args, "enable_breakout_hold_signal", True)):
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
            rows.append(
                {
                    "signal_priority": 1,
                    "signal_type": "breakout_hold",
                    "code": item.code,
                    "name": item.name,
                    "latest_date": date_text(latest["日期"]),
                    "entry_date": date_text(entry["日期"]),
                    "trigger_date": "",
                    "entry_age": latest_idx - breakout_entry,
                    "trigger_age": np.nan,
                    "close": round(float(latest["close"]), 3),
                    "pct_chg": round(float(latest["pct_chg"]), 2),
                    "amount": float(latest["amount"]),
                    "amount_text": _format_amount(float(latest["amount"])),
                    "entry_volume_ratio_20": round(float(entry.get("volume_ratio_20", 0)), 2),
                    "trigger_volume_vs_entry": np.nan,
                    "volume_10_vs_entry": np.nan,
                    "amount_10_vs_entry": np.nan,
                    "entry_close_breakout_gap": np.nan,
                    "macd_water_ok": np.nan,
                    "macd_min_dif": np.nan,
                    "macd_min_dea": np.nan,
                    "model_prob": np.nan,
                    "rule_score": rule,
                    "final_score": rule,
                    "pullback_low_date": info["pullback_low_date"],
                    "pullback_low_vs_entry_low": round(float(info["pullback_low_vs_entry_low"]) * 100, 2),
                    "breakout_ret_since_entry": round(float(info["breakout_ret_since_entry"]) * 100, 2),
                    "reason": "、".join(reasons[:7]),
                }
            )

    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            int(row.get("signal_priority", 0) or 0),
            float(row.get("final_score", 0) or 0),
            float(row.get("amount", 0) or 0),
        ),
    )


def attach_forward_returns(row: Dict[str, object], feat: pd.DataFrame, forward_days: int) -> Dict[str, object]:
    latest_date = pd.Timestamp(row["latest_date"])
    idxs = np.flatnonzero((pd.to_datetime(feat["日期"]).dt.normalize() == latest_date.normalize()).to_numpy())
    if len(idxs) == 0:
        return row
    idx = int(idxs[-1])
    close = feat["close"].astype(float).to_numpy()
    high = feat["最高"].astype(float).to_numpy()
    low = feat["最低"].astype(float).to_numpy()
    dates = pd.to_datetime(feat["日期"]).dt.strftime("%Y-%m-%d").to_numpy()
    base = close[idx]
    for step in range(1, forward_days + 1):
        j = idx + step
        key = f"ret_{step}d_pct"
        row[f"date_{step}d"] = dates[j] if j < len(feat) else ""
        row[key] = round((close[j] / base - 1) * 100, 3) if j < len(feat) and base > 0 else np.nan
    end = min(len(feat) - 1, idx + forward_days)
    if end > idx and base > 0:
        row[f"high_{forward_days}d_pct"] = round((float(np.max(high[idx + 1 : end + 1])) / base - 1) * 100, 3)
        row[f"low_{forward_days}d_pct"] = round((float(np.min(low[idx + 1 : end + 1])) / base - 1) * 100, 3)
        row[f"ret_{forward_days}d_pct"] = row.get(f"ret_{forward_days}d_pct", np.nan)
        row["win_3d"] = bool(float(row[f"ret_{forward_days}d_pct"]) > 0) if np.isfinite(row[f"ret_{forward_days}d_pct"]) else False
    else:
        row[f"high_{forward_days}d_pct"] = np.nan
        row[f"low_{forward_days}d_pct"] = np.nan
        row["win_3d"] = False
    return row


def attach_forward_returns_fast(
    row: Dict[str, object],
    bundle: FastFeature,
    latest_idx: int,
    forward_days: int,
) -> Dict[str, object]:
    base = float(bundle.close[latest_idx])
    for step in range(1, forward_days + 1):
        j = latest_idx + step
        row[f"date_{step}d"] = str(bundle.date_texts[j]) if j < len(bundle.close) else ""
        row[f"ret_{step}d_pct"] = round((float(bundle.close[j]) / base - 1.0) * 100, 3) if j < len(bundle.close) and base > 0 else np.nan
    end = min(len(bundle.close) - 1, latest_idx + forward_days)
    if end > latest_idx and base > 0:
        row[f"high_{forward_days}d_pct"] = round((float(np.nanmax(bundle.high[latest_idx + 1 : end + 1])) / base - 1.0) * 100, 3)
        row[f"low_{forward_days}d_pct"] = round((float(np.nanmin(bundle.low[latest_idx + 1 : end + 1])) / base - 1.0) * 100, 3)
        value = row.get(f"ret_{forward_days}d_pct", np.nan)
        row["win_3d"] = bool(np.isfinite(value) and float(value) > 0)
    else:
        row[f"high_{forward_days}d_pct"] = np.nan
        row[f"low_{forward_days}d_pct"] = np.nan
        row["win_3d"] = False
    return row


def summarize(daily_top: pd.DataFrame, benchmark: pd.DataFrame, forward_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ret_col = f"ret_{forward_days}d_pct"
    daily_rows = []
    for day, group in daily_top.groupby("latest_date", sort=True):
        bench = benchmark[benchmark["latest_date"] == day]
        valid = group.dropna(subset=[ret_col])
        daily_rows.append(
            {
                "date": day,
                "count": int(len(group)),
                "mean_ret_3d_pct": round(float(valid[ret_col].mean()), 3) if not valid.empty else np.nan,
                "median_ret_3d_pct": round(float(valid[ret_col].median()), 3) if not valid.empty else np.nan,
                "win_rate_pct": round(float((valid[ret_col] > 0).mean() * 100), 2) if not valid.empty else np.nan,
                "benchmark_mean_3d_pct": round(float(bench[ret_col].mean()), 3) if not bench.empty else np.nan,
                "benchmark_win_rate_pct": round(float((bench[ret_col] > 0).mean() * 100), 2) if not bench.empty else np.nan,
            }
        )
    daily_summary = pd.DataFrame(daily_rows)

    signal_rows = []
    for signal_type, group in daily_top.groupby("signal_type", sort=True):
        valid = group.dropna(subset=[ret_col])
        signal_rows.append(
            {
                "signal_type": signal_type,
                "count": int(len(group)),
                "mean_ret_3d_pct": round(float(valid[ret_col].mean()), 3) if not valid.empty else np.nan,
                "median_ret_3d_pct": round(float(valid[ret_col].median()), 3) if not valid.empty else np.nan,
                "win_rate_pct": round(float((valid[ret_col] > 0).mean() * 100), 2) if not valid.empty else np.nan,
            }
        )
    signal_summary = pd.DataFrame(signal_rows)
    return daily_summary, signal_summary


def json_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    clean = df.replace({np.nan: None})
    return clean.to_dict(orient="records")


def run(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    static_dir = Path(args.static_output_dir)
    static_dir.mkdir(parents=True, exist_ok=True)

    model, norm, model_args, device = load_checkpoint(Path(args.model_path), args.device)
    model_args = add_missing_args(model_args)
    model_args.candidate_min_amount = args.candidate_min_amount
    model_args.enable_breakout_hold_signal = not args.disable_breakout_hold_signal
    model_args.disable_entry_close_breakout_signal = args.disable_entry_close_breakout_signal

    items = load_universe(Path(args.universe_file), main_board_only=args.main_board_only)
    if args.max_stocks > 0:
        items = items[: args.max_stocks]
    print(f"[{datetime.now():%F %T}] loading feature cache for {len(items)} stocks", flush=True)
    feature_cache = load_feature_cache(items, Path(args.history_dir), min_rows=int(model_args.seq_len) + 30)
    fast_cache = build_fast_feature_cache(feature_cache, model_args)
    days = trading_days(feature_cache, args.start, args.end, args.min_day_coverage, args.min_day_count)
    if not days:
        raise RuntimeError("No trading days found in requested range")
    print(f"[{datetime.now():%F %T}] backtesting days: {[d.strftime('%Y-%m-%d') for d in days]}", flush=True)
    print(
        f"[{datetime.now():%F %T}] score mode: {'fast-rule-only' if args.fast_rule_only else 'full-model'}",
        flush=True,
    )

    min_date = pd.Timestamp(args.min_signal_date) if args.min_signal_date else pd.Timestamp(args.start) - pd.Timedelta(days=365 * 5)
    min_day = np.datetime64(min_date.strftime("%Y-%m-%d"), "D")
    top_rows: List[Dict[str, object]] = []
    all_candidate_rows: List[Dict[str, object]] = []
    benchmark_rows: List[Dict[str, object]] = []

    for day_no, day in enumerate(days, start=1):
        candidates: List[Dict[str, object]] = []
        bench_for_day: List[Dict[str, object]] = []
        for code, bundle in fast_cache.items():
            latest_idx = latest_index_asof(bundle, day)
            if latest_idx is None or latest_idx < int(model_args.seq_len) + int(model_args.min_contract_days) + 1:
                continue
            if float(bundle.amount[latest_idx]) >= float(args.benchmark_min_amount):
                bench = {
                    "latest_date": day.strftime("%Y-%m-%d"),
                    "code": bundle.item.code,
                    "name": bundle.item.name,
                    "close": round(float(bundle.close[latest_idx]), 3),
                }
                bench_for_day.append(attach_forward_returns_fast(bench, bundle, latest_idx, args.forward_days))
            if args.fast_rule_only:
                row = fast_score_one_asof(bundle, latest_idx, model_args, min_day)
            else:
                work = bundle.feat.iloc[: latest_idx + 1]
                row = score_one_asof(bundle.item, work, model, norm, device, model_args, min_date)
            if row is not None:
                candidates.append(attach_forward_returns_fast(row, bundle, latest_idx, args.forward_days))
        ranked = sorted(
            candidates,
            key=lambda row: (float(row.get("final_score", 0) or 0), float(row.get("amount", 0) or 0)),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            all_candidate_rows.append(row)
            if rank <= args.top_n:
                top_rows.append(dict(row))
        benchmark_rows.extend(bench_for_day)
        print(
            f"[{datetime.now():%F %T}] {day_no}/{len(days)} {day:%Y-%m-%d}: "
            f"candidates={len(ranked)}, top={min(args.top_n, len(ranked))}, benchmark={len(bench_for_day)}",
            flush=True,
        )

    all_candidates = pd.DataFrame(all_candidate_rows)
    daily_top = pd.DataFrame(top_rows)
    benchmark = pd.DataFrame(benchmark_rows)
    if not daily_top.empty:
        daily_top = daily_top.sort_values(["latest_date", "rank"]).reset_index(drop=True)
    if not all_candidates.empty:
        all_candidates = all_candidates.sort_values(["latest_date", "rank"]).reset_index(drop=True)
    daily_summary, signal_summary = summarize(daily_top, benchmark, args.forward_days)

    all_candidates.to_csv(out_dir / "june_all_candidates.csv", index=False, encoding="utf-8-sig")
    daily_top.to_csv(out_dir / "june_daily_top20.csv", index=False, encoding="utf-8-sig")
    benchmark.to_csv(out_dir / "june_benchmark_returns.csv", index=False, encoding="utf-8-sig")
    daily_summary.to_csv(out_dir / "june_daily_summary.csv", index=False, encoding="utf-8-sig")
    signal_summary.to_csv(out_dir / "june_signal_summary.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start": args.start,
        "end": args.end,
        "top_n": args.top_n,
        "forward_days": args.forward_days,
        "score_mode": "fast-rule-only" if args.fast_rule_only else "full-model",
        "universe_size": len(items),
        "history_ok": len(fast_cache),
        "trading_days": [d.strftime("%Y-%m-%d") for d in days],
        "daily_summary": json_records(daily_summary),
        "signal_summary": json_records(signal_summary),
        "top_rows": json_records(daily_top),
    }
    (out_dir / "june_top20_backtest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (static_dir / "june_top20_backtest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    template_path = Path("templates") / "june_backtest.html"
    if template_path.exists():
        html = template_path.read_text(encoding="utf-8")
        static_root = static_dir.parent
        (static_root / "index.html").write_text(html, encoding="utf-8")
        (static_root / "june-backtest.html").write_text(html, encoding="utf-8")
    print(f"[{datetime.now():%F %T}] outputs written under {out_dir} and {static_dir}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--forward-days", type=int, default=3)
    parser.add_argument("--model-path", default="model_cache/volume_contraction_breakout_5y_tcn.pt")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--universe-file", default="data_cache/volume_contraction_screen_20260701_mainboard_entry_close/refresh_status.csv")
    parser.add_argument("--output-dir", default="data_cache/june_top20_backtest")
    parser.add_argument("--static-output-dir", default="static/reports")
    parser.add_argument("--candidate-min-amount", type=float, default=50_000_000)
    parser.add_argument("--benchmark-min-amount", type=float, default=0)
    parser.add_argument("--min-day-coverage", type=float, default=0.3)
    parser.add_argument("--min-day-count", type=int, default=20)
    parser.add_argument("--main-board-only", action="store_true", default=True)
    parser.add_argument("--include-non-main-board", dest="main_board_only", action="store_false")
    parser.add_argument("--disable-breakout-hold-signal", action="store_true")
    parser.add_argument("--disable-entry-close-breakout-signal", action="store_true")
    parser.add_argument(
        "--fast-rule-only",
        action="store_true",
        help="Use cached array rule scoring for historical replay instead of neural inference.",
    )
    parser.add_argument("--min-signal-date", default="")
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
