#!/usr/bin/env python
"""Latent main-uptrend model.

Target pattern:
1. A stock had a heavy-volume "main-force entry" signal.
2. It did not immediately enter the main run.
3. Between 30 and 100 trading days later, a rapid rally day appeared.
4. Within 200 trading days from the entry signal, the stock reached 5x.

The model is trained to score the quiet days shortly before the rapid rally,
not the already-obvious rally day.
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import pickle
import random
import time
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
    _code6,
    _format_amount,
    _latest_trading_start,
    _now_text,
    add_features,
    fetch_hist,
    future_multiples,
    future_peak_info,
    load_spot_universe,
    make_universe,
)

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
CONTEXT_COLS = [
    "entry_age",
    "entry_day_ret",
    "entry_amount_ratio_20",
    "entry_volume_ratio_20",
    "entry_turnover",
    "ret_since_entry",
    "max_ret_since_entry",
    "drawdown_from_post_entry_high",
    "days_since_post_entry_high",
    "post_entry_range",
    "latest_ret_20",
    "latest_amount_ratio_20",
    "latest_volume_ratio_20",
    "latest_close_to_high_60",
    "latest_turnover",
]


@dataclass
class SeqSample:
    code: str
    name: str
    sample_idx: int
    entry_idx: int
    rapid_idx: int
    label: int
    event_id: str
    future_max_multiple: float
    future_peak_idx: int
    seq: np.ndarray
    context: np.ndarray
    meta: Dict[str, object]


def ensure_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for latent_uptrend_model.py") from TORCH_IMPORT_ERROR


def sanitize_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def make_sequence(feat: pd.DataFrame, idx: int, seq_len: int) -> Optional[np.ndarray]:
    start = idx - seq_len + 1
    if start < 0:
        return None
    seq = feat.iloc[start : idx + 1][SEQ_COLS].to_numpy(dtype=np.float32)
    if seq.shape != (seq_len, len(SEQ_COLS)):
        return None
    return sanitize_array(seq)


def signal_masks(feat: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    day_ret = feat["day_ret"].to_numpy(dtype=float)
    body = feat["body_pct"].to_numpy(dtype=float)
    amp = feat["amplitude"].to_numpy(dtype=float)
    amount_ratio = feat["amount_ratio_20"].to_numpy(dtype=float)
    volume_ratio = feat["volume_ratio_20"].to_numpy(dtype=float)
    amount_pctile = feat["amount_pctile_120"].to_numpy(dtype=float)
    turnover = feat["turnover"].to_numpy(dtype=float)
    ret20 = feat["ret_20"].to_numpy(dtype=float)
    close_to_high_60 = feat["close_to_high_60"].to_numpy(dtype=float)
    breakout_60 = feat["breakout_60"].to_numpy(dtype=float)

    entry = (
        (amount_ratio >= args.entry_min_amount_ratio)
        & (volume_ratio >= args.entry_min_volume_ratio)
        & (amount_pctile >= args.entry_min_amount_pctile)
        & (turnover >= args.entry_min_turnover)
        & (ret20 <= args.entry_max_ret20)
        & (
            (day_ret >= args.entry_min_ret)
            | ((body >= args.entry_min_body) & (amp >= args.entry_min_amplitude))
        )
    )
    rapid = (
        (day_ret >= args.rapid_min_ret)
        & (body >= args.rapid_min_body)
        & (amount_ratio >= args.rapid_min_amount_ratio)
        & (volume_ratio >= args.rapid_min_volume_ratio)
        & ((breakout_60 == 1) | (close_to_high_60 >= args.rapid_min_close_to_high_60))
    )
    entry = np.nan_to_num(entry.astype(bool), nan=False)
    rapid = np.nan_to_num(rapid.astype(bool), nan=False)
    return entry, rapid


def context_features(feat: pd.DataFrame, entry_idx: int, sample_idx: int) -> np.ndarray:
    entry = feat.iloc[entry_idx]
    sample = feat.iloc[sample_idx]
    close = feat["close"].reset_index(drop=True)
    entry_close = float(close.iloc[entry_idx])
    sample_close = float(close.iloc[sample_idx])
    post = close.iloc[entry_idx : sample_idx + 1]
    post_high = float(post.max()) if len(post) else sample_close
    post_low = float(post.min()) if len(post) else sample_close
    high_idx = int(post.idxmax()) if len(post) else sample_idx

    ret_since_entry = sample_close / entry_close - 1 if entry_close > 0 else 0.0
    max_ret_since_entry = post_high / entry_close - 1 if entry_close > 0 else 0.0
    drawdown = sample_close / post_high - 1 if post_high > 0 else 0.0
    post_range = post_high / post_low - 1 if post_low > 0 else 0.0

    values = np.array(
        [
            sample_idx - entry_idx,
            float(entry.get("day_ret", 0)),
            float(entry.get("amount_ratio_20", 0)),
            float(entry.get("volume_ratio_20", 0)),
            float(entry.get("turnover", 0)),
            ret_since_entry,
            max_ret_since_entry,
            drawdown,
            sample_idx - high_idx,
            post_range,
            float(sample.get("ret_20", 0)),
            float(sample.get("amount_ratio_20", 0)),
            float(sample.get("volume_ratio_20", 0)),
            float(sample.get("close_to_high_60", 0)),
            float(sample.get("turnover", 0)),
        ],
        dtype=np.float32,
    )
    return sanitize_array(values)


def is_not_already_running(feat: pd.DataFrame, entry_idx: int, sample_idx: int, args: argparse.Namespace) -> bool:
    if sample_idx <= entry_idx:
        return False
    close = feat["close"].reset_index(drop=True)
    entry_close = float(close.iloc[entry_idx])
    sample_close = float(close.iloc[sample_idx])
    if entry_close <= 0:
        return False
    post = close.iloc[entry_idx : sample_idx + 1]
    max_multiple = float(post.max() / entry_close)
    latest = feat.iloc[sample_idx]
    already_big_day = (
        float(latest.get("day_ret", 0)) >= args.latest_started_ret
        and float(latest.get("body_pct", 0)) >= args.latest_started_body
    )
    if already_big_day:
        return False
    if max_multiple > args.max_pre_run_multiple:
        return False
    if sample_close / entry_close > args.max_sample_to_entry_multiple:
        return False
    if float(latest.get("ret_20", 0)) > args.max_sample_ret20:
        return False
    return True


def make_sample(
    item: StockItem,
    feat: pd.DataFrame,
    sample_idx: int,
    entry_idx: int,
    rapid_idx: int,
    label: int,
    event_id: str,
    future_mult: float,
    peak_idx: int,
    seq_len: int,
) -> Optional[SeqSample]:
    seq = make_sequence(feat, sample_idx, seq_len)
    if seq is None:
        return None
    ctx = context_features(feat, entry_idx, sample_idx)
    sample = feat.iloc[sample_idx]
    entry = feat.iloc[entry_idx]
    rapid_date = feat.iloc[rapid_idx]["日期"].strftime("%Y-%m-%d") if 0 <= rapid_idx < len(feat) else ""
    peak_date = feat.iloc[peak_idx]["日期"].strftime("%Y-%m-%d") if 0 <= peak_idx < len(feat) else ""
    meta = {
        "code": item.code,
        "name": item.name,
        "sample_date": sample["日期"].strftime("%Y-%m-%d"),
        "entry_date": entry["日期"].strftime("%Y-%m-%d"),
        "rapid_date": rapid_date,
        "entry_age": sample_idx - entry_idx,
        "lead_days_to_rapid": rapid_idx - sample_idx if rapid_idx >= 0 else np.nan,
        "label": label,
        "close": float(sample["close"]),
        "pct_chg": float(sample["pct_chg"]),
        "amount": float(sample["amount"]),
        "entry_amount": float(entry["amount"]),
        "entry_pct_chg": float(entry["pct_chg"]),
        "future_max_multiple": round(float(future_mult), 4) if np.isfinite(future_mult) else np.nan,
        "future_peak_date": peak_date,
        "event_id": event_id,
    }
    for name, value in zip(CONTEXT_COLS, ctx):
        meta[name] = float(value)
    return SeqSample(
        code=item.code,
        name=item.name,
        sample_idx=sample_idx,
        entry_idx=entry_idx,
        rapid_idx=rapid_idx,
        label=label,
        event_id=event_id,
        future_max_multiple=future_mult,
        future_peak_idx=peak_idx,
        seq=seq,
        context=ctx,
        meta=meta,
    )


def detect_latent_events_for_stock(
    item: StockItem,
    hist: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[List[SeqSample], List[SeqSample], List[Dict[str, object]]]:
    feat = add_features(hist)
    if feat.empty or len(feat) < args.seq_len + args.fivefold_days + 5:
        return [], [], []

    work = feat.reset_index(drop=True)
    close = work["close"].reset_index(drop=True)
    future_mult = future_multiples(close, args.fivefold_days)
    entry_mask, rapid_mask = signal_masks(work, args)

    max_entry = len(work) - args.fivefold_days - 1
    min_entry = max(args.seq_len, 140)
    positives: List[SeqSample] = []
    negatives: List[SeqSample] = []
    events: List[Dict[str, object]] = []
    positive_sample_indices = set()
    rng = random.Random(args.random_state + int(item.code))

    if max_entry <= min_entry:
        return [], [], []

    entry_indices = np.flatnonzero(entry_mask)
    entry_indices = entry_indices[(entry_indices >= min_entry) & (entry_indices <= max_entry)]

    for entry_idx in entry_indices:
        r_start = entry_idx + args.delay_min
        r_end = min(entry_idx + args.delay_max, len(work) - 1)
        if r_start >= len(work):
            continue

        rapid_candidates = np.flatnonzero(rapid_mask[r_start : r_end + 1]) + r_start
        qualified_rapid = []
        for rapid_idx in rapid_candidates:
            if close.iloc[rapid_idx] / close.iloc[entry_idx] < args.rapid_min_from_entry:
                continue
            qualified_rapid.append(int(rapid_idx))
        rapid_idx = qualified_rapid[0] if qualified_rapid else -1

        has_5x = np.isfinite(future_mult[entry_idx]) and future_mult[entry_idx] >= args.multiple
        if has_5x and rapid_idx >= 0:
            exact_mult, peak_idx = future_peak_info(close, entry_idx, args.fivefold_days)
            event_id = f"{item.code}_{work.iloc[entry_idx]['日期'].strftime('%Y%m%d')}_{rapid_idx}"
            events.append(
                {
                    "code": item.code,
                    "name": item.name,
                    "entry_date": work.iloc[entry_idx]["日期"].strftime("%Y-%m-%d"),
                    "rapid_date": work.iloc[rapid_idx]["日期"].strftime("%Y-%m-%d"),
                    "entry_to_rapid_days": rapid_idx - entry_idx,
                    "entry_pct_chg": float(work.iloc[entry_idx]["pct_chg"]),
                    "entry_amount": float(work.iloc[entry_idx]["amount"]),
                    "entry_amount_ratio_20": float(work.iloc[entry_idx]["amount_ratio_20"]),
                    "entry_volume_ratio_20": float(work.iloc[entry_idx]["volume_ratio_20"]),
                    "future_max_multiple": round(float(exact_mult), 4),
                    "future_peak_date": work.iloc[peak_idx]["日期"].strftime("%Y-%m-%d"),
                }
            )
            ps_start = max(entry_idx + args.delay_min, rapid_idx - args.lead_max)
            ps_end = rapid_idx - args.lead_min
            for sample_idx in range(ps_start, ps_end + 1):
                if sample_idx < args.seq_len or sample_idx >= len(work):
                    continue
                if not is_not_already_running(work, entry_idx, sample_idx, args):
                    continue
                sample = make_sample(
                    item,
                    work,
                    sample_idx,
                    entry_idx,
                    rapid_idx,
                    1,
                    event_id,
                    exact_mult,
                    peak_idx,
                    args.seq_len,
                )
                if sample is not None:
                    positives.append(sample)
                    positive_sample_indices.add((entry_idx, sample_idx))
        else:
            if rng.random() > args.negative_entry_sample_rate:
                continue
            sample_pool = list(range(entry_idx + args.delay_min, min(entry_idx + args.delay_max, len(work) - 1) + 1, args.negative_stride))
            rng.shuffle(sample_pool)
            picked = 0
            for sample_idx in sample_pool:
                if picked >= args.max_negatives_per_entry:
                    break
                if (entry_idx, sample_idx) in positive_sample_indices:
                    continue
                if sample_idx + args.lead_max < len(work) and rapid_mask[sample_idx + args.lead_min : sample_idx + args.lead_max + 1].any():
                    continue
                if not is_not_already_running(work, entry_idx, sample_idx, args):
                    continue
                exact_mult, peak_idx = future_peak_info(close, entry_idx, args.fivefold_days)
                sample = make_sample(
                    item,
                    work,
                    sample_idx,
                    entry_idx,
                    -1,
                    0,
                    f"{item.code}_{work.iloc[entry_idx]['日期'].strftime('%Y%m%d')}_neg",
                    exact_mult,
                    peak_idx,
                    args.seq_len,
                )
                if sample is not None:
                    negatives.append(sample)
                    picked += 1

    if len(negatives) > args.max_negatives_per_stock:
        negatives = rng.sample(negatives, args.max_negatives_per_stock)
    return positives, negatives, events


class TemporalAttentionNet(nn.Module):
    def __init__(self, seq_features: int, context_features: int, d_model: int = 96, dropout: float = 0.18):
        super().__init__()
        self.input_proj = nn.Linear(seq_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 256, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Linear(context_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2 + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(seq)
        x = x + self.pos[:, : x.size(1), :]
        x = self.encoder(x)
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        c = self.context(ctx)
        return self.head(torch.cat([avg_pool, max_pool, c], dim=1)).squeeze(-1)


def choose_device(args: argparse.Namespace) -> torch.device:
    ensure_torch()
    if args.device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_arrays(samples: List[SeqSample]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    X_seq = np.stack([s.seq for s in samples]).astype(np.float32)
    X_ctx = np.stack([s.context for s in samples]).astype(np.float32)
    y = np.array([s.label for s in samples], dtype=np.float32)
    meta = pd.DataFrame([s.meta for s in samples])
    return X_seq, X_ctx, y, meta


def chronological_split(meta: pd.DataFrame, y: np.ndarray, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(meta["sample_date"], errors="coerce")
    order = np.argsort(dates.to_numpy())
    split = int(len(order) * (1 - test_size))
    train_idx = order[:split]
    val_idx = order[split:]
    if y[train_idx].sum() >= 3 and y[val_idx].sum() >= 2 and (y[val_idx] == 0).sum() >= 5:
        return train_idx, val_idx
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=y)
    return np.asarray(train_idx), np.asarray(val_idx)


def train_deep_model(
    samples: List[SeqSample],
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[TemporalAttentionNet, Dict[str, object], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    ensure_torch()
    X_seq, X_ctx, y, meta = build_arrays(samples)
    train_idx, val_idx = chronological_split(meta, y, args.val_size, args.random_state)

    seq_mean = X_seq[train_idx].mean(axis=(0, 1), keepdims=True)
    seq_std = X_seq[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
    ctx_mean = X_ctx[train_idx].mean(axis=0, keepdims=True)
    ctx_std = X_ctx[train_idx].std(axis=0, keepdims=True) + 1e-6
    X_seq_n = (X_seq - seq_mean) / seq_std
    X_ctx_n = (X_ctx - ctx_mean) / ctx_std
    X_seq_n = np.nan_to_num(X_seq_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_ctx_n = np.nan_to_num(X_ctx_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    device = choose_device(args)
    model = TemporalAttentionNet(len(SEQ_COLS), len(CONTEXT_COLS), d_model=args.d_model, dropout=args.dropout)
    if device.type == "cuda":
        model = model.to(device)
        if torch.cuda.device_count() > 1 and not args.single_gpu:
            print(f"[{_now_text()}] using DataParallel on {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
    else:
        model = model.to(device)
        print(f"[{_now_text()}] CUDA unavailable, training on CPU")

    y_train = y[train_idx]
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    sample_weights = np.where(y_train > 0, max(1.0, neg / max(pos, 1.0)), 1.0).astype(np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
    train_ds = TensorDataset(
        torch.from_numpy(X_seq_n[train_idx]).float(),
        torch.from_numpy(X_ctx_n[train_idx]).float(),
        torch.from_numpy(y[train_idx]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_seq_n[val_idx]).float(),
        torch.from_numpy(X_ctx_n[val_idx]).float(),
        torch.from_numpy(y[val_idx]).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    # The sampler already balances classes.  Adding pos_weight on top made the
    # tiny positive class too aggressive and can destabilize attention layers.
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(2, args.epochs))

    best_auc = -1.0
    best_state = None
    best_metrics: Dict[str, object] = {}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for seq_b, ctx_b, y_b in train_loader:
            seq_b = seq_b.to(device, non_blocking=True)
            ctx_b = ctx_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(seq_b, ctx_b)
            loss = loss_fn(logits, y_b)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        model.eval()
        probs = []
        ys = []
        with torch.no_grad():
            for seq_b, ctx_b, y_b in val_loader:
                logits = model(seq_b.to(device), ctx_b.to(device))
                probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
                ys.extend(y_b.numpy().tolist())
        y_val = np.asarray(ys, dtype=np.float32)
        p_val = np.asarray(probs, dtype=np.float32)
        finite_probs = np.isfinite(p_val).all()
        p_val = np.nan_to_num(p_val, nan=0.5, posinf=1.0, neginf=0.0)
        if len(np.unique(y_val)) > 1:
            auc = float(roc_auc_score(y_val, p_val))
            ap = float(average_precision_score(y_val, p_val))
        else:
            auc = 0.5
            ap = 0.0
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"[{_now_text()}] epoch {epoch:03d} loss={mean_loss:.4f} val_auc={auc:.4f} val_ap={ap:.4f}")
        params_finite = all(torch.isfinite(p.detach()).all().item() for p in model.parameters())
        if not params_finite:
            print(f"[{_now_text()}] non-finite model parameters at epoch {epoch}; stopping and restoring best state")
            break
        if finite_probs and auc > best_auc:
            best_auc = auc
            best_metrics = {"val_auc": round(auc, 4), "val_ap": round(ap, 4), "epoch": epoch}
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(f"[{_now_text()}] early stopping at epoch {epoch}")
            break

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    if best_state is not None:
        raw_model.load_state_dict(best_state)
    raw_model = raw_model.to(device)

    # Final validation report.
    raw_model.eval()
    with torch.no_grad():
        seq_val = torch.from_numpy(X_seq_n[val_idx]).float().to(device)
        ctx_val = torch.from_numpy(X_ctx_n[val_idx]).float().to(device)
        p_val = torch.sigmoid(raw_model(seq_val, ctx_val)).detach().cpu().numpy()
    y_val = y[val_idx]
    p_val = np.nan_to_num(p_val, nan=0.5, posinf=1.0, neginf=0.0)
    pred = (p_val >= args.threshold).astype(int)
    report = classification_report(y_val, pred, output_dict=True, zero_division=0)
    best_metrics.update(
        {
            "samples": int(len(samples)),
            "positives": int(y.sum()),
            "negatives": int((y == 0).sum()),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "val_positives": int(y_val.sum()),
            "val_negatives": int((y_val == 0).sum()),
            "classification_report": report,
            "device": str(device),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    )

    model_path = Path("model_cache") / "latent_uptrend_tcn.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": raw_model.state_dict(),
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
    return raw_model, best_metrics, (seq_mean, seq_std, ctx_mean, ctx_std)


def latent_rule_score(feat: pd.DataFrame, entry_idx: int, latest_idx: int) -> float:
    row = feat.iloc[latest_idx]
    entry = feat.iloc[entry_idx]
    ctx = context_features(feat, entry_idx, latest_idx)
    values = dict(zip(CONTEXT_COLS, ctx))

    def clip01(x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        return max(0.0, min(1.0, float(x)))

    age = values["entry_age"]
    age_score = math.exp(-((age - 68.0) / 28.0) ** 2)
    score = 20.0 * age_score
    score += 15.0 * clip01((values["entry_amount_ratio_20"] - 1.8) / 3.5)
    score += 12.0 * clip01((values["entry_volume_ratio_20"] - 1.8) / 3.0)
    score += 10.0 * clip01((values["entry_turnover"] - 3.0) / 12.0)
    score += 12.0 * clip01((values["latest_close_to_high_60"] + 0.12) / 0.14)
    score += 8.0 * clip01((float(row.get("above_ma20", 0)) + float(row.get("above_ma60", 0))) / 2.0)
    score += 8.0 * clip01((values["ret_since_entry"] + 0.05) / 0.45)
    score += 7.0 * clip01((0.55 - values["max_ret_since_entry"]) / 0.55)
    score += 5.0 * clip01((values["drawdown_from_post_entry_high"] + 0.18) / 0.18)
    score += 3.0 * clip01((1.35 - values["latest_amount_ratio_20"]) / 1.0)
    return round(score, 2)


def candidate_reason(feat: pd.DataFrame, entry_idx: int, latest_idx: int) -> str:
    row = feat.iloc[latest_idx]
    entry = feat.iloc[entry_idx]
    ctx = dict(zip(CONTEXT_COLS, context_features(feat, entry_idx, latest_idx)))
    reasons = [
        f"{int(ctx['entry_age'])}天前爆量",
        f"进场成交额{entry.get('amount_ratio_20', 0):.1f}倍",
        f"进场成交量{entry.get('volume_ratio_20', 0):.1f}倍",
    ]
    if ctx["latest_close_to_high_60"] >= -0.08:
        reasons.append("接近60日高点")
    if row.get("above_ma20", 0) == 1 and row.get("above_ma60", 0) == 1:
        reasons.append("站上20/60日线")
    if ctx["max_ret_since_entry"] <= 0.6:
        reasons.append("尚未明显主升")
    if row.get("amount_ratio_20", 0) <= 1.2:
        reasons.append("近期缩量蓄势")
    return "、".join(reasons[:6])


def score_current_candidates(
    histories: Dict[str, Tuple[StockItem, pd.DataFrame]],
    model: TemporalAttentionNet,
    normalizers: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> pd.DataFrame:
    ensure_torch()
    seq_mean, seq_std, ctx_mean, ctx_std = normalizers
    device = choose_device(args)
    model = model.to(device)
    model.eval()
    rows = []
    for pos, (code, (item, hist)) in enumerate(histories.items(), start=1):
        feat = add_features(hist)
        if feat.empty or len(feat) <= args.seq_len + args.entry_age_max + 5:
            continue
        work = feat.reset_index(drop=True)
        latest_idx = len(work) - 1
        latest = work.iloc[latest_idx]
        if float(latest.get("amount", 0)) < args.candidate_min_amount:
            continue
        if not is_not_already_running(work, max(0, latest_idx - args.entry_age_max), latest_idx, args):
            # This rough precheck removes obvious limit-up/current-run bars.
            if float(latest.get("day_ret", 0)) >= args.latest_started_ret:
                continue

        entry_mask, rapid_mask = signal_masks(work, args)
        min_entry = max(args.seq_len, latest_idx - args.entry_age_max)
        max_entry = latest_idx - args.entry_age_min
        if max_entry < min_entry:
            continue
        candidates = np.flatnonzero(entry_mask[min_entry : max_entry + 1]) + min_entry
        if len(candidates) == 0:
            continue
        # Prefer the strongest recent entry signal.
        best_entry = None
        best_entry_power = -np.inf
        for entry_idx in candidates:
            if rapid_mask[entry_idx + args.delay_min : latest_idx + 1].any():
                continue
            if not is_not_already_running(work, int(entry_idx), latest_idx, args):
                continue
            entry = work.iloc[int(entry_idx)]
            power = float(entry.get("amount_ratio_20", 0)) + float(entry.get("volume_ratio_20", 0)) + 0.12 * float(entry.get("turnover", 0))
            if power > best_entry_power:
                best_entry_power = power
                best_entry = int(entry_idx)
        if best_entry is None:
            continue
        seq = make_sequence(work, latest_idx, args.seq_len)
        if seq is None:
            continue
        ctx = context_features(work, best_entry, latest_idx)
        seq_n = (seq[None, :, :] - seq_mean) / seq_std
        ctx_n = (ctx[None, :] - ctx_mean) / ctx_std
        seq_n = np.nan_to_num(seq_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        ctx_n = np.nan_to_num(ctx_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        with torch.no_grad():
            prob = torch.sigmoid(
                model(
                    torch.from_numpy(seq_n).float().to(device),
                    torch.from_numpy(ctx_n).float().to(device),
                )
            ).detach().cpu().numpy()[0]
        if not np.isfinite(prob):
            prob = 0.0
        rule = latent_rule_score(work, best_entry, latest_idx)
        final = 0.68 * float(prob) * 100.0 + 0.32 * rule
        entry = work.iloc[best_entry]
        rows.append(
            {
                "code": item.code,
                "name": item.name,
                "latest_date": latest["日期"].strftime("%Y-%m-%d"),
                "entry_date": entry["日期"].strftime("%Y-%m-%d"),
                "entry_age": latest_idx - best_entry,
                "close": round(float(latest["close"]), 3),
                "pct_chg": round(float(latest["pct_chg"]), 2),
                "amount": float(latest["amount"]),
                "amount_text": _format_amount(float(latest["amount"])),
                "turnover": round(float(latest.get("turnover", 0)), 2),
                "entry_pct_chg": round(float(entry["pct_chg"]), 2),
                "entry_amount_text": _format_amount(float(entry["amount"])),
                "entry_amount_ratio_20": round(float(entry.get("amount_ratio_20", 0)), 2),
                "entry_volume_ratio_20": round(float(entry.get("volume_ratio_20", 0)), 2),
                "ret_since_entry": round(float(ctx[CONTEXT_COLS.index("ret_since_entry")]) * 100, 2),
                "max_ret_since_entry": round(float(ctx[CONTEXT_COLS.index("max_ret_since_entry")]) * 100, 2),
                "drawdown_from_post_entry_high": round(float(ctx[CONTEXT_COLS.index("drawdown_from_post_entry_high")]) * 100, 2),
                "latest_amount_ratio_20": round(float(latest.get("amount_ratio_20", 0)), 2),
                "latest_volume_ratio_20": round(float(latest.get("volume_ratio_20", 0)), 2),
                "close_to_high_60": round(float(latest.get("close_to_high_60", 0)) * 100, 2),
                "model_prob": round(float(prob) * 100, 2),
                "rule_score": rule,
                "final_score": round(final, 2),
                "reason": candidate_reason(work, best_entry, latest_idx),
            }
        )
        if pos % args.progress_every == 0:
            print(f"[{_now_text()}] current scoring progress {pos}/{len(histories)}, candidates={len(rows)}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["final_score", "amount"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def load_histories(items: List[StockItem], args: argparse.Namespace, out_dir: Path) -> Dict[str, Tuple[StockItem, pd.DataFrame]]:
    hist_dir = Path(args.history_dir) if args.history_dir else out_dir / "hist"
    hist_dir.mkdir(parents=True, exist_ok=True)
    start_date, end_date = _latest_trading_start(args.lookback_years)
    histories: Dict[str, Tuple[StockItem, pd.DataFrame]] = {}
    failures = []
    print(f"[{_now_text()}] loading history: {len(items)} stocks from {hist_dir}")
    for i, item in enumerate(items, start=1):
        path = hist_dir / f"{item.code}.csv"
        if path.exists() and not args.refresh_history:
            try:
                df = pd.read_csv(path)
                histories[item.code] = (item, df)
            except Exception as exc:
                failures.append({"code": item.code, "name": item.name, "error": str(exc)})
        else:
            item2, df, error = fetch_hist(item, hist_dir, start_date, end_date, args.cache_days, args.refresh_history, args.retry)
            if error or df.empty:
                failures.append({"code": item.code, "name": item.name, "error": error or "empty"})
            else:
                histories[item2.code] = (item2, df)
        if i % args.progress_every == 0 or i == len(items):
            print(f"[{_now_text()}] history progress {i}/{len(items)}, ok={len(histories)}, failed={len(failures)}")
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "history_failures.csv", index=False, encoding="utf-8-sig")
    return histories


def run(args: argparse.Namespace) -> Dict[str, object]:
    ensure_torch()
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    if torch is not None:
        torch.manual_seed(args.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_state)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{_now_text()}] loading spot universe")
    spot_cache_dir = Path(args.spot_cache_dir) if args.spot_cache_dir else out_dir
    spot = load_spot_universe(spot_cache_dir, cache_hours=args.spot_cache_hours, refresh=args.refresh_spot)
    items = make_universe(spot, min_amount=args.min_amount, max_stocks=args.max_stocks, include_bj=args.include_bj)
    pd.DataFrame([item.__dict__ for item in items]).to_csv(out_dir / "universe.csv", index=False, encoding="utf-8-sig")
    print(f"[{_now_text()}] universe after filters: {len(items)} stocks")

    histories = load_histories(items, args, out_dir)
    print(f"[{_now_text()}] building latent samples")
    positives: List[SeqSample] = []
    negatives: List[SeqSample] = []
    events: List[Dict[str, object]] = []
    for i, (code, (item, hist)) in enumerate(histories.items(), start=1):
        ps, ns, es = detect_latent_events_for_stock(item, hist, args)
        positives.extend(ps)
        negatives.extend(ns)
        events.extend(es)
        if i % args.progress_every == 0 or i == len(histories):
            print(f"[{_now_text()}] sample progress {i}/{len(histories)}, positives={len(positives)}, negatives={len(negatives)}, events={len(events)}")

    if not positives:
        raise RuntimeError("No positive latent samples found; relax entry/rapid thresholds")

    rng = random.Random(args.random_state)
    target_neg = min(len(negatives), max(args.min_negative_samples, len(positives) * args.negative_ratio))
    if len(negatives) > target_neg:
        negatives = rng.sample(negatives, target_neg)
    samples = positives + negatives
    rng.shuffle(samples)

    _, _, _, meta = build_arrays(samples)
    meta.to_csv(out_dir / "latent_training_samples.csv", index=False, encoding="utf-8-sig")
    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df.sort_values(["entry_date", "future_max_multiple"], ascending=[False, False]).to_csv(
            out_dir / "latent_5x_entry_events.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        events_df.to_csv(out_dir / "latent_5x_entry_events.csv", index=False, encoding="utf-8-sig")

    print(f"[{_now_text()}] training deep temporal model: samples={len(samples)}, positives={len(positives)}, negatives={len(negatives)}")
    model, metrics, normalizers = train_deep_model(samples, args, out_dir)
    print(f"[{_now_text()}] scoring current latent candidates")
    candidates = score_current_candidates(histories, model, normalizers, args)
    candidates.to_csv(out_dir / "latent_current_candidates.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_time": datetime.now().isoformat(),
        "definition": {
            "fivefold_days": args.fivefold_days,
            "multiple": args.multiple,
            "entry_to_rapid_window": [args.delay_min, args.delay_max],
            "positive_lead_window_before_rapid": [args.lead_min, args.lead_max],
        },
        "universe_size": len(items),
        "history_ok": len(histories),
        "latent_events": len(events),
        "positive_samples": len(positives),
        "negative_samples": len(negatives),
        "training_samples": len(samples),
        "candidate_rows": len(candidates),
        "metrics": metrics,
        "args": vars(args),
        "outputs": {
            "events": str(out_dir / "latent_5x_entry_events.csv"),
            "samples": str(out_dir / "latent_training_samples.csv"),
            "candidates": str(out_dir / "latent_current_candidates.csv"),
            "summary": str(out_dir / "latent_summary.json"),
        },
    }
    (out_dir / "latent_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 200日5倍：爆量进场后30-100日再启动事件 Top ===")
    if events_df.empty:
        print("(empty)")
    else:
        cols = [
            "code",
            "name",
            "entry_date",
            "rapid_date",
            "entry_to_rapid_days",
            "entry_pct_chg",
            "entry_amount_ratio_20",
            "entry_volume_ratio_20",
            "future_max_multiple",
            "future_peak_date",
        ]
        print(events_df.sort_values(["entry_date", "future_max_multiple"], ascending=[False, False])[cols].head(args.print_top).to_string(index=False))

    print("\n=== 当前即将主升候选 Top ===")
    if candidates.empty:
        print("(empty)")
    else:
        cols = [
            "rank",
            "code",
            "name",
            "latest_date",
            "entry_date",
            "entry_age",
            "pct_chg",
            "amount_text",
            "entry_amount_ratio_20",
            "entry_volume_ratio_20",
            "ret_since_entry",
            "max_ret_since_entry",
            "close_to_high_60",
            "model_prob",
            "rule_score",
            "final_score",
            "reason",
        ]
        print(candidates[cols].head(args.print_top).to_string(index=False))

    print(f"\n[{_now_text()}] outputs written under {out_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data_cache/latent_uptrend")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--spot-cache-dir", default="data_cache/main_uptrend")
    parser.add_argument("--lookback-years", type=int, default=7)
    parser.add_argument("--min-amount", type=float, default=0)
    parser.add_argument("--candidate-min-amount", type=float, default=50_000_000)
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--include-bj", action="store_true")
    parser.add_argument("--spot-cache-hours", type=float, default=2.0)
    parser.add_argument("--refresh-spot", action="store_true")
    parser.add_argument("--refresh-history", action="store_true")
    parser.add_argument("--cache-days", type=float, default=1.0)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=400)

    parser.add_argument("--multiple", type=float, default=5.0)
    parser.add_argument("--fivefold-days", type=int, default=200)
    parser.add_argument("--delay-min", type=int, default=30)
    parser.add_argument("--delay-max", type=int, default=100)
    parser.add_argument("--lead-min", type=int, default=1)
    parser.add_argument("--lead-max", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=80)

    parser.add_argument("--entry-min-amount-ratio", type=float, default=2.0)
    parser.add_argument("--entry-min-volume-ratio", type=float, default=2.0)
    parser.add_argument("--entry-min-amount-pctile", type=float, default=0.72)
    parser.add_argument("--entry-min-turnover", type=float, default=3.0)
    parser.add_argument("--entry-min-ret", type=float, default=0.025)
    parser.add_argument("--entry-min-body", type=float, default=0.015)
    parser.add_argument("--entry-min-amplitude", type=float, default=0.045)
    parser.add_argument("--entry-max-ret20", type=float, default=0.55)

    parser.add_argument("--rapid-min-ret", type=float, default=0.07)
    parser.add_argument("--rapid-min-body", type=float, default=0.04)
    parser.add_argument("--rapid-min-amount-ratio", type=float, default=1.5)
    parser.add_argument("--rapid-min-volume-ratio", type=float, default=1.4)
    parser.add_argument("--rapid-min-close-to-high-60", type=float, default=-0.03)
    parser.add_argument("--rapid-min-from-entry", type=float, default=1.05)

    parser.add_argument("--max-pre-run-multiple", type=float, default=1.9)
    parser.add_argument("--max-sample-to-entry-multiple", type=float, default=1.75)
    parser.add_argument("--max-sample-ret20", type=float, default=0.42)
    parser.add_argument("--latest-started-ret", type=float, default=0.055)
    parser.add_argument("--latest-started-body", type=float, default=0.035)
    parser.add_argument("--entry-age-min", type=int, default=30)
    parser.add_argument("--entry-age-max", type=int, default=100)

    parser.add_argument("--negative-entry-sample-rate", type=float, default=0.45)
    parser.add_argument("--negative-stride", type=int, default=7)
    parser.add_argument("--max-negatives-per-entry", type=int, default=2)
    parser.add_argument("--max-negatives-per-stock", type=int, default=12)
    parser.add_argument("--negative-ratio", type=int, default=6)
    parser.add_argument("--min-negative-samples", type=int, default=300)

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--print-top", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
