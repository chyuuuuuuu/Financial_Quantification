#!/usr/bin/env python
"""Analyze buy/sell rules for the 5-year contraction model.

This is a research backtest over historical labeled contraction samples. It
excludes ChiNext codes (300/301), scores the later chronological validation
segment with the trained model, and compares rule-based entry/exit variants.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from latent_uptrend_model import TemporalAttentionNet, choose_device, ensure_torch
from main_uptrend_model import add_features
from volume_contraction_breakout_model import (
    CONTEXT_COLS,
    SEQ_COLS,
    contraction_rule_score,
    make_sequence,
    contraction_context,
    sanitize,
)

try:
    import torch
except ImportError as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class SellRule:
    max_hold: int
    stop_loss: float
    trail_trigger: float
    trail_pct: float

    @property
    def key(self) -> str:
        return f"h{self.max_hold}_sl{int(self.stop_loss*100)}_tr{int(self.trail_trigger*100)}_{int(self.trail_pct*100)}"


def is_chinext(code: str) -> bool:
    code = str(code).zfill(6)
    return code.startswith("300") or code.startswith("301")


def chronological_validation_mask(df: pd.DataFrame, val_size: float) -> np.ndarray:
    dates = pd.to_datetime(df["sample_date"], errors="coerce")
    order = np.argsort(dates.to_numpy())
    split = int(len(order) * (1 - val_size))
    mask = np.zeros(len(df), dtype=bool)
    mask[order[split:]] = True
    return mask


def load_model(model_path: Path, device_name: str) -> Tuple[TemporalAttentionNet, Dict[str, np.ndarray], Dict[str, object], torch.device]:
    ensure_torch()
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(device=device_name)
    device = choose_device(args)
    model_args = ckpt.get("args", {})
    d_model = int(model_args.get("d_model", 96))
    dropout = float(model_args.get("dropout", 0.18))
    model = TemporalAttentionNet(len(SEQ_COLS), len(CONTEXT_COLS), d_model=d_model, dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    norm = {
        "seq_mean": ckpt["seq_mean"],
        "seq_std": ckpt["seq_std"],
        "ctx_mean": ckpt["ctx_mean"],
        "ctx_std": ckpt["ctx_std"],
    }
    return model, norm, ckpt, device


def infer_scores(
    rows: pd.DataFrame,
    history_dir: Path,
    model: TemporalAttentionNet,
    norm: Dict[str, np.ndarray],
    device: torch.device,
    seq_len: int,
    batch_size: int,
) -> pd.DataFrame:
    ensure_torch()
    scored_rows: List[Dict[str, object]] = []
    seq_batches: List[np.ndarray] = []
    ctx_batches: List[np.ndarray] = []
    row_refs: List[int] = []

    grouped = rows.groupby("code", sort=False)
    for code, group in grouped:
        path = history_dir / f"{code}.csv"
        if not path.exists():
            continue
        hist = pd.read_csv(path)
        feat = add_features(hist)
        if feat.empty:
            continue
        work = feat.reset_index(drop=True)
        date_to_idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(work["日期"])}
        for row_idx, row in group.iterrows():
            sample_date = str(row["sample_date"])
            entry_date = str(row["entry_date"])
            sample_idx = date_to_idx.get(sample_date)
            entry_idx = date_to_idx.get(entry_date)
            if sample_idx is None or entry_idx is None or sample_idx + 1 >= len(work):
                continue
            seq = make_sequence(work, sample_idx, seq_len)
            if seq is None:
                continue
            ctx = contraction_context(work, entry_idx, sample_idx)
            seq_batches.append(seq)
            ctx_batches.append(ctx)
            row_refs.append(row_idx)
            if len(seq_batches) >= batch_size:
                flush_scores(rows, scored_rows, seq_batches, ctx_batches, row_refs, work_lookup=None, model=model, norm=norm, device=device)
    flush_scores(rows, scored_rows, seq_batches, ctx_batches, row_refs, work_lookup=None, model=model, norm=norm, device=device)
    return pd.DataFrame(scored_rows)


def flush_scores(
    source_rows: pd.DataFrame,
    scored_rows: List[Dict[str, object]],
    seq_batches: List[np.ndarray],
    ctx_batches: List[np.ndarray],
    row_refs: List[int],
    work_lookup: Optional[object],
    model: TemporalAttentionNet,
    norm: Dict[str, np.ndarray],
    device: torch.device,
) -> None:
    if not seq_batches:
        return
    seq = np.stack(seq_batches).astype(np.float32)
    ctx = np.stack(ctx_batches).astype(np.float32)
    seq = sanitize((seq - norm["seq_mean"]) / norm["seq_std"])
    ctx = sanitize((ctx - norm["ctx_mean"]) / norm["ctx_std"])
    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.from_numpy(seq).float().to(device), torch.from_numpy(ctx).float().to(device))
        ).detach().cpu().numpy()
    for ref, prob, ctx_raw in zip(row_refs, probs, ctx_batches):
        row = source_rows.loc[ref].to_dict()
        # The stored context already has the same fields, so this rule score can
        # be recomputed cheaply from those values without reloading history.
        rule = rule_score_from_row(row)
        row["model_prob"] = float(prob) * 100
        row["rule_score"] = rule
        row["final_score"] = 0.68 * float(prob) * 100 + 0.32 * rule
        scored_rows.append(row)
    seq_batches.clear()
    ctx_batches.clear()
    row_refs.clear()


def rule_score_from_row(row: Dict[str, object]) -> float:
    def clip01(x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        return max(0.0, min(1.0, float(x)))

    age = float(row.get("entry_age", 0) or 0)
    score = 18 * math.exp(-((age - 45) / 28) ** 2)
    score += 16 * clip01((float(row.get("entry_volume_ratio_20", 0) or 0) - 3.0) / 4.5)
    score += 14 * clip01((float(row.get("entry_amount_ratio_20", 0) or 0) - 2.5) / 5.0)
    score += 18 * clip01((0.62 - float(row.get("volume_10_vs_entry", 0) or 0)) / 0.62)
    score += 12 * clip01((0.70 - float(row.get("amount_10_vs_entry", 0) or 0)) / 0.70)
    score += 10 * clip01((float(row.get("latest_close_to_high_60", -1) or -1) + 0.15) / 0.15)
    score += 6 * clip01((float(row.get("above_ma20", 0) or 0) + float(row.get("above_ma60", 0) or 0)) / 2.0)
    score += 6 * clip01((0.65 - float(row.get("max_ret_since_entry", 0) or 0)) / 0.65)
    return round(score, 2)


def attach_trade_paths(scored: pd.DataFrame, history_dir: Path, max_hold: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for code, group in scored.groupby("code", sort=False):
        path = history_dir / f"{code}.csv"
        if not path.exists():
            continue
        hist = pd.read_csv(path)
        feat = add_features(hist)
        if feat.empty:
            continue
        work = feat.reset_index(drop=True)
        date_to_idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(work["日期"])}
        open_ = work["开盘"].astype(float).to_numpy()
        high = work["最高"].astype(float).to_numpy()
        low = work["最低"].astype(float).to_numpy()
        close = work["close"].astype(float).to_numpy()
        dates = pd.to_datetime(work["日期"]).dt.strftime("%Y-%m-%d").to_numpy()
        for _, row in group.iterrows():
            sample_idx = date_to_idx.get(str(row["sample_date"]))
            if sample_idx is None or sample_idx + 1 >= len(work):
                continue
            entry_idx = sample_idx + 1
            entry_price = open_[entry_idx]
            sample_close = float(row["close"])
            if not np.isfinite(entry_price) or entry_price <= 0 or sample_close <= 0:
                continue
            end_idx = min(len(work) - 1, entry_idx + max_hold - 1)
            data = row.to_dict()
            data.update(
                {
                    "trade_entry_date": dates[entry_idx],
                    "trade_entry_price": float(entry_price),
                    "buy_gap": float(entry_price / sample_close - 1),
                    "path_high": high[entry_idx : end_idx + 1].astype(float),
                    "path_low": low[entry_idx : end_idx + 1].astype(float),
                    "path_close": close[entry_idx : end_idx + 1].astype(float),
                    "path_dates": dates[entry_idx : end_idx + 1],
                }
            )
            rows.append(data)
    return pd.DataFrame(rows)


def simulate_one(row: pd.Series, rule: SellRule) -> Tuple[float, str, str, int, float, float]:
    entry = float(row["trade_entry_price"])
    highs = row["path_high"]
    lows = row["path_low"]
    closes = row["path_close"]
    dates = row["path_dates"]
    if len(closes) == 0 or entry <= 0:
        return np.nan, "", "", 0, np.nan, np.nan
    stop_price = entry * (1 - rule.stop_loss)
    peak_close = entry
    activated = False
    mfe = -np.inf
    mae = np.inf
    exit_price = float(closes[-1])
    exit_reason = "time"
    exit_date = str(dates[-1])
    hold_days = len(closes)
    for i, (h, l, c, d) in enumerate(zip(highs, lows, closes, dates), start=1):
        h = float(h)
        l = float(l)
        c = float(c)
        mfe = max(mfe, h / entry - 1)
        mae = min(mae, l / entry - 1)
        if l <= stop_price:
            exit_price = stop_price
            exit_reason = "stop"
            exit_date = str(d)
            hold_days = i
            break
        if h >= entry * (1 + rule.trail_trigger) or c >= entry * (1 + rule.trail_trigger):
            activated = True
        peak_close = max(peak_close, c)
        if activated and c <= peak_close * (1 - rule.trail_pct):
            exit_price = c
            exit_reason = "trail"
            exit_date = str(d)
            hold_days = i
            break
    return exit_price / entry - 1, exit_reason, exit_date, hold_days, mfe, mae


def metric_summary(returns: List[float], holds: List[int]) -> Dict[str, float]:
    arr = np.asarray(returns, dtype=float)
    if len(arr) == 0:
        return {}
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    return {
        "trades": int(len(arr)),
        "avg_ret": round(float(arr.mean()) * 100, 2),
        "median_ret": round(float(np.median(arr)) * 100, 2),
        "win_rate": round(float((arr > 0).mean()) * 100, 2),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else np.inf,
        "p10_ret": round(float(np.quantile(arr, 0.10)) * 100, 2),
        "p90_ret": round(float(np.quantile(arr, 0.90)) * 100, 2),
        "hit_20": round(float((arr >= 0.20).mean()) * 100, 2),
        "hit_50": round(float((arr >= 0.50).mean()) * 100, 2),
        "avg_hold": round(float(np.mean(holds)), 2) if holds else np.nan,
    }


def evaluate_strategy(
    signals: pd.DataFrame,
    sell_results: Dict[str, Dict[str, np.ndarray]],
    sell_rule: SellRule,
    score_min: float,
    amount_min: float,
    age_min: int,
    age_max: int,
    vol10_max: float,
    max_ret_since_entry: float,
    max_buy_gap: float,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> Dict[str, object]:
    dates = pd.to_datetime(signals["sample_date"])
    mask = (
        (signals["final_score"] >= score_min)
        & (signals["amount"] >= amount_min)
        & (signals["entry_age"] >= age_min)
        & (signals["entry_age"] <= age_max)
        & (signals["volume_10_vs_entry"] <= vol10_max)
        & (signals["max_ret_since_entry"] <= max_ret_since_entry)
        & (signals["buy_gap"] <= max_buy_gap)
        & (signals["buy_gap"] >= -0.08)
    )
    if start_date is not None:
        mask &= dates >= start_date
    if end_date is not None:
        mask &= dates < end_date
    idxs = np.flatnonzero(mask.to_numpy())
    if len(idxs) == 0:
        return {}
    idxs = idxs[np.argsort(dates.iloc[idxs].to_numpy())]
    res = sell_results[sell_rule.key]
    busy_until: Dict[str, pd.Timestamp] = {}
    returns: List[float] = []
    holds: List[int] = []
    reasons: List[str] = []
    picked: List[int] = []
    for idx in idxs:
        code = str(signals.iloc[idx]["code"])
        sample_date = dates.iloc[idx]
        if code in busy_until and sample_date <= busy_until[code]:
            continue
        ret = float(res["ret"][idx])
        if not np.isfinite(ret):
            continue
        exit_date = pd.Timestamp(res["exit_date"][idx])
        busy_until[code] = exit_date
        returns.append(ret)
        holds.append(int(res["hold"][idx]))
        reasons.append(str(res["reason"][idx]))
        picked.append(idx)
    metrics = metric_summary(returns, holds)
    if not metrics:
        return {}
    metrics.update(
        {
            "score_min": score_min,
            "amount_min": amount_min,
            "age_min": age_min,
            "age_max": age_max,
            "vol10_max": vol10_max,
            "max_ret_since_entry": max_ret_since_entry,
            "max_buy_gap": max_buy_gap,
            "max_hold": sell_rule.max_hold,
            "stop_loss": sell_rule.stop_loss,
            "trail_trigger": sell_rule.trail_trigger,
            "trail_pct": sell_rule.trail_pct,
            "stop_exit_pct": round(100 * (np.asarray(reasons) == "stop").mean(), 2) if reasons else 0.0,
            "trail_exit_pct": round(100 * (np.asarray(reasons) == "trail").mean(), 2) if reasons else 0.0,
            "time_exit_pct": round(100 * (np.asarray(reasons) == "time").mean(), 2) if reasons else 0.0,
        }
    )
    return metrics


def build_sell_results(signals: pd.DataFrame, sell_rules: List[SellRule]) -> Dict[str, Dict[str, np.ndarray]]:
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for rule in sell_rules:
        rets, reasons, exits, holds, mfes, maes = [], [], [], [], [], []
        for _, row in signals.iterrows():
            ret, reason, exit_date, hold, mfe, mae = simulate_one(row, rule)
            rets.append(ret)
            reasons.append(reason)
            exits.append(exit_date)
            holds.append(hold)
            mfes.append(mfe)
            maes.append(mae)
        results[rule.key] = {
            "ret": np.asarray(rets, dtype=float),
            "reason": np.asarray(reasons, dtype=object),
            "exit_date": np.asarray(exits, dtype=object),
            "hold": np.asarray(holds, dtype=int),
            "mfe": np.asarray(mfes, dtype=float),
            "mae": np.asarray(maes, dtype=float),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="data_cache/volume_contraction_5y/contraction_training_samples.csv")
    parser.add_argument("--candidates", default="data_cache/volume_contraction_5y/contraction_current_candidates.csv")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--model-path", default="model_cache/volume_contraction_breakout_5y_tcn.pt")
    parser.add_argument("--output-dir", default="data_cache/volume_contraction_5y")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--max-hold", type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(args.samples, dtype={"code": str})
    samples["code"] = samples["code"].str.zfill(6)
    val_mask = chronological_validation_mask(samples, args.val_size)
    val = samples[val_mask].copy()
    val = val[~val["code"].map(is_chinext)].copy()
    val = val.sort_values(["sample_date", "code"]).reset_index(drop=True)

    model, norm, ckpt, device = load_model(Path(args.model_path), args.device)
    scored = infer_scores(val, Path(args.history_dir), model, norm, device, args.seq_len, args.batch_size)
    scored = scored.sort_values(["sample_date", "code"]).reset_index(drop=True)
    signals = attach_trade_paths(scored, Path(args.history_dir), args.max_hold)
    signals = signals.sort_values(["sample_date", "code"]).reset_index(drop=True)

    sell_rules = [
        SellRule(max_hold=h, stop_loss=sl, trail_trigger=trig, trail_pct=trail)
        for h in [20, 30, 45, 60]
        for sl in [0.07, 0.10, 0.12]
        for trig, trail in [(0.20, 0.10), (0.30, 0.12)]
    ]
    sell_results = build_sell_results(signals, sell_rules)
    signal_dates = pd.to_datetime(signals["sample_date"])
    rule_split = signal_dates.sort_values().iloc[int(len(signal_dates) * 0.5)] if len(signal_dates) else None

    rows: List[Dict[str, object]] = []
    for sell_rule in sell_rules:
        for score_min, amount_min, age_pair, vol10_max, max_ret in itertools.product(
            [70, 75, 80, 85],
            [50_000_000, 100_000_000],
            [(20, 120), (30, 180), (20, 250)],
            [0.25, 0.35],
            [0.35, 0.55],
        ):
            age_min, age_max = age_pair
            common = dict(
                signals=signals,
                sell_results=sell_results,
                sell_rule=sell_rule,
                score_min=score_min,
                amount_min=amount_min,
                age_min=age_min,
                age_max=age_max,
                vol10_max=vol10_max,
                max_ret_since_entry=max_ret,
                max_buy_gap=0.05,
            )
            dev = evaluate_strategy(start_date=None, end_date=rule_split, **common)
            test = evaluate_strategy(start_date=rule_split, end_date=None, **common)
            if not dev or not test:
                continue
            if dev["trades"] < 30 or test["trades"] < 20:
                continue
            row = {f"dev_{k}": v for k, v in dev.items()}
            row.update({f"test_{k}": v for k, v in test.items() if k not in common})
            row.update(
                {
                    "score_min": score_min,
                    "amount_min": amount_min,
                    "age_min": age_min,
                    "age_max": age_max,
                    "vol10_max": vol10_max,
                    "max_ret_since_entry": max_ret,
                    "max_buy_gap": 0.05,
                    "max_hold": sell_rule.max_hold,
                    "stop_loss": sell_rule.stop_loss,
                    "trail_trigger": sell_rule.trail_trigger,
                    "trail_pct": sell_rule.trail_pct,
                }
            )
            row["robust_score"] = (
                float(row["dev_avg_ret"]) * 0.35
                + float(row["test_avg_ret"]) * 0.45
                + float(row["test_median_ret"]) * 0.10
                + min(float(row["test_trades"]), 120.0) * 0.02
                + float(row["test_profit_factor"]) * 0.50
            )
            rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["robust_score", "test_profit_factor", "test_avg_ret", "test_trades"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    result.to_csv(out_dir / "trade_rule_analysis_no_chinext.csv", index=False, encoding="utf-8-sig")

    scored_out = signals.drop(columns=["path_high", "path_low", "path_close", "path_dates"], errors="ignore")
    scored_out.to_csv(out_dir / "validation_signals_no_chinext_scored.csv", index=False, encoding="utf-8-sig")

    candidates = pd.read_csv(args.candidates, dtype={"code": str})
    candidates["code"] = candidates["code"].str.zfill(6)
    candidates = candidates[~candidates["code"].map(is_chinext)].copy()
    candidates.to_csv(out_dir / "current_candidates_no_chinext.csv", index=False, encoding="utf-8-sig")

    summary = {
        "samples_total": int(len(samples)),
        "validation_after_chinext_exclusion": int(len(val)),
        "scored_validation_signals": int(len(signals)),
        "rule_split_date": str(rule_split.date()) if rule_split is not None else "",
        "strategies_tested": int(len(result)),
        "top_strategy": result.head(1).to_dict(orient="records")[0] if not result.empty else {},
        "current_candidates_no_chinext": int(len(candidates)),
        "model_metrics": ckpt.get("metrics", {}),
    }
    (out_dir / "trade_rule_analysis_no_chinext_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not result.empty:
        cols = [
            "score_min",
            "amount_min",
            "age_min",
            "age_max",
            "vol10_max",
            "max_ret_since_entry",
            "max_hold",
            "stop_loss",
            "trail_trigger",
            "trail_pct",
            "dev_trades",
            "dev_avg_ret",
            "dev_win_rate",
            "dev_profit_factor",
            "test_trades",
            "test_avg_ret",
            "test_median_ret",
            "test_win_rate",
            "test_profit_factor",
            "test_hit_20",
            "test_hit_50",
        ]
        print(result[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
