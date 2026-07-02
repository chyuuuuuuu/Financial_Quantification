"""Single-stock analysis service for the 5-year contraction breakout model."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from latent_uptrend_model import TemporalAttentionNet
from main_uptrend_model import StockItem, _format_amount, add_features, normalize_hist
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
    future_run_multiple,
    is_contracted_not_running,
    main_start_mask,
    make_sequence,
    post_entry_retest_features,
    reason_text,
    sanitize,
    triple_volume_bull_mask,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


ROOT = Path(__file__).parent
HIST_DIR = ROOT / "data_cache" / "main_uptrend" / "hist"
MODEL_PATH = ROOT / "model_cache" / "volume_contraction_breakout_5y_tcn.pt"
VOLUME_DIR = ROOT / "data_cache" / "volume_contraction_5y"


def normalize_code(code: str) -> str:
    code = str(code or "").strip().upper()
    code = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
    return code.zfill(6)[-6:]


def pct(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return round(float(value) * 100, 2)


def pct_from_ratio(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return round(float(value), 2)


class UptrendAnalysisService:
    def __init__(self) -> None:
        self._universe: Optional[pd.DataFrame] = None
        self._model = None
        self._checkpoint = None
        self._device = None
        self.args = argparse.Namespace(
            entry_volume_ratio=3.0,
            entry_amount_ratio=2.5,
            entry_min_ret=0.03,
            entry_min_body=0.018,
            entry_gap_min_ret=0.04,
            entry_gap_min_gap=0.025,
            entry_gap_min_body=0.006,
            entry_min_turnover=2.5,
            entry_max_ret20=0.45,
            run_start_min_ret=0.035,
            run_start_min_body=0.015,
            run_start_ret3=0.12,
            run_start_volume_ratio=1.2,
            run_start_amount_ratio=1.2,
            second_volume_ratio=3.0,
            second_amount_ratio=2.5,
            second_min_ret=0.03,
            second_min_body=0.015,
            allow_confirmed_run_start=True,
            month_days=20,
            month_multiple=2.0,
            min_contract_days=5,
            max_entry_to_run_days=250,
            lead_min=1,
            lead_max=10,
            seq_len=240,
            max_volume5_vs_entry=0.62,
            max_volume10_vs_entry=0.55,
            max_amount10_vs_entry=0.68,
            max_latest_volume_vs_entry=0.70,
            max_pre_run_ret=0.70,
            max_current_ret_since_entry=0.55,
            max_sample_ret20=0.35,
            latest_started_ret=0.055,
            latest_started_body=0.035,
            disable_entry_close_breakout_signal=False,
            entry_close_breakout_recent_days=12,
            entry_close_breakout_min_body=0.006,
            entry_close_breakout_min_ret=0.0,
            entry_close_breakout_close_buffer=0.0,
            entry_close_breakout_prior_tolerance=0.01,
            entry_close_breakout_max_trigger_volume_vs_entry=0.95,
            entry_close_breakout_max_trigger_amount_vs_entry=1.20,
            entry_close_breakout_max_volume10_vs_entry=0.62,
            entry_close_breakout_max_latest_volume_vs_entry=0.70,
            entry_close_breakout_max_amount10_vs_entry=0.72,
            entry_close_breakout_max_pre_breakout_ret=0.55,
            entry_close_breakout_max_ret_since_entry=0.75,
            entry_close_breakout_require_macd_to_latest=True,
            disable_macd_water_filter=False,
            macd_waterline_min=0.0,
            macd_require_dea_above_water=True,
            enable_breakout_hold_signal=True,
            breakout_hold_lookback_days=45,
            breakout_hold_min_days=1,
        )

    def load_universe(self) -> pd.DataFrame:
        if self._universe is not None:
            return self._universe
        frames: List[pd.DataFrame] = []
        for path in [
            VOLUME_DIR / "universe.csv",
            ROOT / "data_cache" / "main_uptrend" / "spot_universe.csv",
            VOLUME_DIR / "contraction_current_candidates.csv",
        ]:
            if not path.exists():
                continue
            df = pd.read_csv(path, dtype={"code": str, "代码": str})
            if "code" in df.columns:
                part = df[["code", "name"]].copy() if "name" in df.columns else df[["code"]].copy()
                if "name" not in part.columns:
                    part["name"] = ""
            elif "代码" in df.columns:
                part = df[["代码", "名称"]].rename(columns={"代码": "code", "名称": "name"}).copy()
            else:
                continue
            part["code"] = part["code"].map(normalize_code)
            part["name"] = part["name"].fillna("").astype(str)
            frames.append(part)
        if not frames:
            self._universe = pd.DataFrame(columns=["code", "name"])
            return self._universe
        out = pd.concat(frames, ignore_index=True).drop_duplicates("code", keep="last")
        out = out.sort_values(["code"]).reset_index(drop=True)
        self._universe = out
        return out

    def search(self, q: str, exclude_chinext: bool = False, limit: int = 30) -> List[Dict[str, object]]:
        q = str(q or "").strip()
        df = self.load_universe()
        if exclude_chinext:
            df = df[~df["code"].str.startswith(("300", "301"))]
        if q:
            nq = normalize_code(q) if any(ch.isdigit() for ch in q) else ""
            mask = df["name"].str.contains(q, na=False)
            if nq.strip("0"):
                mask |= df["code"].str.contains(nq.strip("0"), na=False) | (df["code"] == nq)
            df = df[mask]
        return df.head(limit)[["code", "name"]].to_dict(orient="records")

    def stock_name(self, code: str) -> str:
        code = normalize_code(code)
        df = self.load_universe()
        row = df[df["code"] == code]
        if not row.empty:
            return str(row.iloc[0]["name"])
        return code

    def fetch_latest_daily(self, code: str, days: int = 45) -> Optional[pd.DataFrame]:
        market = "1" if code.startswith("6") else "0"
        end = datetime.now().strftime("%Y%m%d")
        begin = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        url = (
            "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            f"?secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6"
            "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
            f"&klt=101&fqt=1&beg={begin}&end={end}"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/124 Safari/537.36",
            "Referer": "https://quote.eastmoney.com/",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data") or {}
            rows = []
            for item in data.get("klines") or []:
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
            return pd.DataFrame(rows) if rows else None
        except Exception:
            return None

    def load_history(self, code: str, refresh: bool = True) -> Tuple[pd.DataFrame, str]:
        code = normalize_code(code)
        path = HIST_DIR / f"{code}.csv"
        if not path.exists():
            raise FileNotFoundError(f"本地没有 {code} 的历史日线缓存")
        hist = normalize_hist(pd.read_csv(path))
        source = "local"
        if refresh:
            latest = self.fetch_latest_daily(code)
            if latest is not None and not latest.empty:
                hist = (
                    pd.concat([hist, normalize_hist(latest)], ignore_index=True)
                    .sort_values("日期")
                    .drop_duplicates("日期", keep="last")
                    .reset_index(drop=True)
                )
                hist.to_csv(path, index=False, encoding="utf-8-sig")
                source = "eastmoney"
        return hist, source

    def load_model(self):
        if self._model is not None:
            return self._model, self._checkpoint, self._device
        if torch is None:
            raise RuntimeError("PyTorch 未安装，无法加载主升模型")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model_args = checkpoint.get("args", {})
        ctx_mean = np.asarray(checkpoint.get("ctx_mean", np.zeros((1, len(CONTEXT_COLS)))))
        ctx_dim = int(ctx_mean.shape[-1])
        model = TemporalAttentionNet(
            len(SEQ_COLS),
            ctx_dim,
            d_model=int(model_args.get("d_model", 96)),
            dropout=float(model_args.get("dropout", 0.18)),
        )
        model.load_state_dict(checkpoint["model_state"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        self._model = model
        self._checkpoint = checkpoint
        self._device = device
        return model, checkpoint, device

    def score_entry_close_breakout(self, feat: pd.DataFrame) -> Optional[Dict[str, object]]:
        latest_idx = len(feat) - 1
        min_date = pd.Timestamp(datetime.now() - timedelta(days=1825))
        found = find_entry_close_breakout_signal(feat, latest_idx, self.args, min_date)
        if found is None:
            return None
        entry_idx, trigger_idx, info, rule = found
        latest = feat.iloc[latest_idx]
        entry = feat.iloc[entry_idx]
        return {
            "signal_priority": 2,
            "signal_type": "entry_close_breakout",
            "entry_date": entry["日期"].strftime("%Y-%m-%d"),
            "trigger_date": info["trigger_date"],
            "entry_age": int(latest_idx - entry_idx),
            "trigger_age": int(latest_idx - trigger_idx),
            "entry_close": round(float(info["entry_close"]), 2),
            "trigger_close": round(float(info["trigger_close"]), 2),
            "entry_pct_chg": round(float(entry["pct_chg"]), 2),
            "entry_amount": float(entry["amount"]),
            "entry_amount_text": _format_amount(float(entry["amount"])),
            "entry_volume_ratio_20": round(float(entry.get("volume_ratio_20", 0)), 2),
            "entry_amount_ratio_20": round(float(entry.get("amount_ratio_20", 0)), 2),
            "trigger_volume_vs_entry": round(float(info["trigger_volume_vs_entry"]), 3),
            "trigger_amount_vs_entry": round(float(info["trigger_amount_vs_entry"]), 3),
            "volume_10_vs_entry": round(float(info["pre_volume_10_vs_entry"]), 3),
            "amount_10_vs_entry": round(float(info["pre_amount_10_vs_entry"]), 3),
            "ret_since_entry": pct(info["latest_ret_since_entry"]),
            "entry_close_breakout_gap": pct(info["entry_close_breakout_gap"]),
            "macd_water_ok": bool(info["macd_water_ok"]),
            "macd_min_dif": round(float(info["macd_min_dif"]), 4),
            "macd_min_dea": round(float(info["macd_min_dea"]), 4),
            "model_prob": None,
            "rule_score": rule,
            "final_score": rule,
            "reason": entry_close_breakout_reason_text(feat, entry_idx, trigger_idx, latest_idx, info),
            "tradable_signal": bool(rule >= 80),
            "latest_close": round(float(latest["close"]), 2),
        }

    def score_breakout_hold(self, feat: pd.DataFrame) -> Optional[Dict[str, object]]:
        if not bool(getattr(self.args, "enable_breakout_hold_signal", True)):
            return None
        latest_idx = len(feat) - 1
        min_date = pd.Timestamp(datetime.now() - timedelta(days=1825))
        best_entry = find_breakout_pullback_hold_entry(feat, latest_idx, self.args, min_date)
        if best_entry is None:
            return None
        latest = feat.iloc[latest_idx]
        entry = feat.iloc[best_entry]
        info = breakout_pullback_hold_features(feat, best_entry, latest_idx)
        rule = breakout_pullback_hold_rule_score(feat, best_entry, latest_idx)
        reasons = [
            f"{latest_idx - best_entry}天前爆量突破",
            f"突破量{float(entry.get('volume_ratio_20', 0)):.1f}倍",
            "回踩不破爆量低点",
            f"自回踩低点修复{float(info['breakout_recovery_from_pullback']) * 100:.1f}%",
        ]
        if float(latest.get("above_ma20", 0)) == 1 and float(latest.get("above_ma60", 0)) == 1:
            reasons.append("站上20/60日线")
        if float(latest.get("close_to_high_60", 0)) >= -0.08:
            reasons.append("接近60日高点")
        return {
            "signal_priority": 1,
            "signal_type": "breakout_hold",
            "entry_date": entry["日期"].strftime("%Y-%m-%d"),
            "entry_age": int(latest_idx - best_entry),
            "entry_close": round(float(info["entry_close"]), 2),
            "entry_low": round(float(info["entry_low"]), 2),
            "entry_pct_chg": round(float(entry["pct_chg"]), 2),
            "entry_amount": float(entry["amount"]),
            "entry_amount_text": _format_amount(float(entry["amount"])),
            "entry_volume_ratio_20": round(float(entry.get("volume_ratio_20", 0)), 2),
            "entry_amount_ratio_20": round(float(entry.get("amount_ratio_20", 0)), 2),
            "pullback_low_date": info["pullback_low_date"],
            "pullback_low_vs_entry_low": pct(info["pullback_low_vs_entry_low"]),
            "pullback_close_vs_entry_close": pct(info["pullback_close_vs_entry_close"]),
            "breakout_ret_since_entry": pct(info["breakout_ret_since_entry"]),
            "breakout_recovery_from_pullback": pct(info["breakout_recovery_from_pullback"]),
            "breakout_entry_score": round(float(info["breakout_entry_score"]), 2),
            "breakout_hold_signal": bool(info["breakout_hold_signal"]),
            "model_prob": None,
            "rule_score": rule,
            "final_score": rule,
            "reason": "、".join(reasons[:7]),
            "tradable_signal": bool(rule >= 70),
            "latest_close": round(float(latest["close"]), 2),
        }

    def score_latest(self, feat: pd.DataFrame) -> Optional[Dict[str, object]]:
        latest_idx = len(feat) - 1
        entry_breakout_result = self.score_entry_close_breakout(feat)
        contraction_result: Optional[Dict[str, object]] = None
        entry_mask = triple_volume_bull_mask(feat, self.args)
        entry_min = max(self.args.seq_len, latest_idx - self.args.max_entry_to_run_days)
        entry_max = latest_idx - self.args.min_contract_days
        entries: List[int] = []
        if entry_max >= entry_min:
            entries = np.flatnonzero(entry_mask[entry_min : entry_max + 1]) + entry_min
            entries = [int(x) for x in entries if is_contracted_not_running(feat, int(x), latest_idx, self.args)]
        if entries:
            best_entry = max(
                entries,
                key=lambda x: float(feat.iloc[x].get("volume_ratio_20", 0))
                + float(feat.iloc[x].get("amount_ratio_20", 0)),
            )
            seq = make_sequence(feat, latest_idx, self.args.seq_len)
            if seq is not None:
                ctx = contraction_context(feat, best_entry, latest_idx)
                retest = post_entry_retest_features(feat, best_entry, latest_idx)
                model, checkpoint, device = self.load_model()
                seq_n = sanitize((seq[None, :, :] - checkpoint["seq_mean"]) / checkpoint["seq_std"])
                ctx_mean = np.asarray(checkpoint["ctx_mean"]).reshape(-1)
                ctx_std = np.asarray(checkpoint["ctx_std"]).reshape(-1)
                if ctx.shape[0] > ctx_mean.shape[0]:
                    ctx = ctx[: ctx_mean.shape[0]]
                elif ctx.shape[0] < ctx_mean.shape[0]:
                    ctx = np.pad(ctx, (0, ctx_mean.shape[0] - ctx.shape[0]), constant_values=0)
                ctx_n = sanitize((ctx[None, :] - ctx_mean) / ctx_std)
                with torch.no_grad():
                    prob = torch.sigmoid(
                        model(torch.from_numpy(seq_n).float().to(device), torch.from_numpy(ctx_n).float().to(device))
                    ).detach().cpu().numpy()[0]
                rule = contraction_rule_score(feat, best_entry, latest_idx)
                final = round(0.68 * float(prob) * 100 + 0.32 * rule, 2)
                latest = feat.iloc[latest_idx]
                entry = feat.iloc[best_entry]
                ctxd = dict(zip(CONTEXT_COLS, ctx))
                contraction_result = {
                    "signal_priority": 1,
                    "signal_type": "contraction_retest",
                    "entry_date": entry["日期"].strftime("%Y-%m-%d"),
                    "entry_age": int(latest_idx - best_entry),
                    "entry_close": round(float(retest["entry_close"]), 2),
                    "entry_pct_chg": round(float(entry["pct_chg"]), 2),
                    "entry_amount": float(entry["amount"]),
                    "entry_amount_text": _format_amount(float(entry["amount"])),
                    "entry_volume_ratio_20": round(float(entry.get("volume_ratio_20", 0)), 2),
                    "entry_amount_ratio_20": round(float(entry.get("amount_ratio_20", 0)), 2),
                    "volume_10_vs_entry": round(float(ctxd["volume_10_vs_entry"]), 3),
                    "amount_10_vs_entry": round(float(ctxd["amount_10_vs_entry"]), 3),
                    "ret_since_entry": pct(ctxd["ret_since_entry"]),
                    "max_ret_since_entry": pct(ctxd["max_ret_since_entry"]),
                    "post_entry_min_ret": pct(retest["post_entry_min_ret"]),
                    "post_entry_min_date": retest["post_entry_min_date"],
                    "entry_close_retest_gap": pct(retest["entry_close_retest_gap"]),
                    "recovered_from_post_entry_low": pct(retest["recovered_from_post_entry_low"]),
                    "entry_close_retest_signal": bool(retest["entry_close_retest_signal"]),
                    "model_prob": round(float(prob) * 100, 2),
                    "rule_score": rule,
                    "final_score": final,
                    "reason": reason_text(feat, best_entry, latest_idx),
                    "tradable_signal": bool(final >= 80 and float(ctxd["volume_10_vs_entry"]) <= 0.35),
                    "latest_close": round(float(latest["close"]), 2),
                }

        breakout_result = self.score_breakout_hold(feat)
        results = [row for row in [entry_breakout_result, contraction_result, breakout_result] if row]
        if not results:
            return None
        return max(
            results,
            key=lambda row: (int(row.get("signal_priority") or 0), float(row.get("final_score") or 0)),
        )

    def signal_days(self, feat: pd.DataFrame, start: Optional[pd.Timestamp] = None) -> Dict[str, List[Dict[str, object]]]:
        entry_mask = triple_volume_bull_mask(feat, self.args)
        future_mult = future_run_multiple(feat["close"].reset_index(drop=True), self.args.month_days)
        run_mask = main_start_mask(feat, future_mult, self.args)
        rows_entry = []
        rows_run = []
        for i in np.flatnonzero(entry_mask):
            row = feat.iloc[int(i)]
            if start is not None and row["日期"] < start:
                continue
            rows_entry.append(
                {
                    "date": row["日期"].strftime("%Y-%m-%d"),
                    "close": round(float(row["close"]), 2),
                    "pct_chg": round(float(row["pct_chg"]), 2),
                    "amount_text": _format_amount(float(row["amount"])),
                    "volume_ratio": round(float(row["volume_ratio_20"]), 2),
                    "amount_ratio": round(float(row["amount_ratio_20"]), 2),
                    "turnover": round(float(row["turnover"]), 2),
                }
            )
        for i in np.flatnonzero(run_mask):
            row = feat.iloc[int(i)]
            if start is not None and row["日期"] < start:
                continue
            rows_run.append(
                {
                    "date": row["日期"].strftime("%Y-%m-%d"),
                    "close": round(float(row["close"]), 2),
                    "pct_chg": round(float(row["pct_chg"]), 2),
                    "amount_text": _format_amount(float(row["amount"])),
                    "volume_ratio": round(float(row["volume_ratio_20"]), 2),
                    "amount_ratio": round(float(row["amount_ratio_20"]), 2),
                    "future_20d_multiple": round(float(future_mult[int(i)]), 3),
                }
            )
        return {"entries": rows_entry[-12:], "main_starts": rows_run[-12:]}

    def key_levels(self, feat: pd.DataFrame) -> Dict[str, object]:
        latest = feat.iloc[-1]
        close = float(latest["close"])
        recent = feat.tail(90).copy()
        peak_idx = recent["close"].idxmax()
        trough_idx = recent["close"].idxmin()
        peak = feat.loc[peak_idx]
        trough = feat.loc[trough_idx]
        mas = {}
        for win in [5, 10, 20, 30, 60, 120, 250]:
            value = float(feat["close"].rolling(win).mean().iloc[-1])
            mas[f"MA{win}"] = {"value": round(value, 2), "distance_pct": round(close / value * 100 - 100, 2)}

        profile = recent.copy()
        step = 10 if close >= 80 else 5
        profile["bucket"] = (profile["close"] / step).round() * step
        prof = (
            profile.groupby("bucket")
            .agg(days=("close", "size"), amount=("amount", "sum"), min_close=("close", "min"), max_close=("close", "max"))
            .reset_index()
            .sort_values("amount", ascending=False)
        )
        profile_rows = []
        for _, row in prof.head(10).sort_values("bucket").iterrows():
            bucket = float(row["bucket"])
            profile_rows.append(
                {
                    "price": round(bucket, 2),
                    "days": int(row["days"]),
                    "amount_text": _format_amount(float(row["amount"])),
                    "position": "above" if bucket > close else "below",
                }
            )

        raw_levels = []
        for name, obj in mas.items():
            raw_levels.append((obj["value"], name))
        for _, row in prof.head(8).iterrows():
            raw_levels.append((float(row["bucket"]), "成交密集区"))
        raw_levels.extend(
            [
                (float(peak["close"]), f"90日收盘高点 {peak['日期'].strftime('%Y-%m-%d')}"),
                (float(trough["close"]), f"90日收盘低点 {trough['日期'].strftime('%Y-%m-%d')}"),
            ]
        )
        above = sorted([(p, r) for p, r in raw_levels if p > close * 1.01], key=lambda x: x[0])[:6]
        below = sorted([(p, r) for p, r in raw_levels if p < close * 0.99], key=lambda x: x[0], reverse=True)[:6]
        return {
            "moving_averages": mas,
            "pressure": [
                {"price": round(p, 2), "distance_pct": round(p / close * 100 - 100, 2), "reason": r}
                for p, r in above
            ],
            "support": [
                {"price": round(p, 2), "distance_pct": round(p / close * 100 - 100, 2), "reason": r}
                for p, r in below
            ],
            "profile": profile_rows,
            "recent_peak": {
                "date": peak["日期"].strftime("%Y-%m-%d"),
                "close": round(float(peak["close"]), 2),
                "drawdown_pct": round(close / float(peak["close"]) * 100 - 100, 2),
            },
            "recent_trough": {
                "date": trough["日期"].strftime("%Y-%m-%d"),
                "close": round(float(trough["close"]), 2),
                "rebound_pct": round(close / float(trough["close"]) * 100 - 100, 2),
            },
        }

    def scenarios(self, feat: pd.DataFrame, levels: Dict[str, object]) -> List[Dict[str, object]]:
        latest = feat.iloc[-1]
        close = float(latest["close"])
        pressure = levels.get("pressure", [])
        support = levels.get("support", [])
        p1 = pressure[0]["price"] if pressure else round(close * 1.08, 2)
        p2 = pressure[1]["price"] if len(pressure) > 1 else round(close * 1.18, 2)
        p3 = pressure[2]["price"] if len(pressure) > 2 else round(close * 1.30, 2)
        s1 = support[0]["price"] if support else round(close * 0.93, 2)
        return [
            {
                "name": "弱反弹",
                "condition": f"冲到 {p1:.2f} 附近放量滞涨或收长上影",
                "action": "降低仓位，避免反抽结束后二次下杀",
                "target": round(p1, 2),
            },
            {
                "name": "中级修复",
                "condition": f"收盘站上 {p1:.2f} 后回踩不破，并继续上攻 {p2:.2f}",
                "action": "按压力位分批处理，不把前高当基础预期",
                "target": round(p2, 2),
            },
            {
                "name": "强二波",
                "condition": f"放量站稳 {p2:.2f}-{p3:.2f}，且回落缩量",
                "action": "保留剩余仓位观察二波，但跌回压力区内要退出",
                "target": round(p3, 2),
            },
            {
                "name": "失败",
                "condition": f"放量跌破 {s1:.2f}",
                "action": "反弹失败，按风险控制处理",
                "target": round(s1, 2),
            },
        ]

    def trade_view(self, feat: pd.DataFrame, score: Optional[Dict[str, object]], levels: Dict[str, object]) -> Dict[str, object]:
        latest = feat.iloc[-1]
        close = float(latest["close"])
        ma5 = levels["moving_averages"].get("MA5", {}).get("value")
        ma10 = levels["moving_averages"].get("MA10", {}).get("value")
        ma20 = levels["moving_averages"].get("MA20", {}).get("value")
        peak = levels["recent_peak"]
        status = "观察"
        if score and score.get("signal_type") == "breakout_hold":
            status = "突破回踩转强"
        elif score and score.get("tradable_signal"):
            status = "潜伏候选"
        if peak["drawdown_pct"] < -25 and close < ma20:
            status = "退潮反抽"
        if close > ma10 and close > ma20:
            status = "修复转强" if status == "观察" else status
        notes = []
        if score:
            notes.append(f"模型综合分 {score['final_score']}，{score['reason']}")
        else:
            notes.append("当前不满足爆量后缩量潜伏或突破回踩不破模型的候选条件")
        notes.append(f"最新价相对 90 日高点回撤 {peak['drawdown_pct']}%")
        if ma5 and close > ma5:
            notes.append("已站上 MA5，短线止跌反弹成立")
        if ma10 and close < ma10:
            notes.append("仍在 MA10 下方，趋势修复尚未确认")
        if ma20 and close < ma20:
            notes.append("仍在 MA20 下方，不能按反转处理")
        return {
            "status": status,
            "notes": notes,
            "risk_controls": [
                "不要用单日大阳线确认反转，至少看能否站上第一压力并缩量回踩不破",
                "如果冲压力位放量滞涨，优先降低风险而不是等回本",
                "跌破最近关键支撑且放量时，反弹逻辑失效",
            ],
        }

    def analyze(self, code: str, refresh: bool = True, start_date: Optional[str] = None) -> Dict[str, object]:
        code = normalize_code(code)
        hist, source = self.load_history(code, refresh=refresh)
        feat = add_features(hist).reset_index(drop=True)
        if feat.empty:
            raise ValueError(f"{code} 历史数据不足，无法分析")
        score = self.score_latest(feat)
        levels = self.key_levels(feat)
        start = pd.Timestamp(start_date) if start_date else None
        latest = feat.iloc[-1]
        is_chinext = code.startswith(("300", "301"))
        response = {
            "code": code,
            "name": self.stock_name(code),
            "is_chinext": is_chinext,
            "data_source": source,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "latest": {
                "date": latest["日期"].strftime("%Y-%m-%d"),
                "open": round(float(latest["开盘"]), 2),
                "close": round(float(latest["close"]), 2),
                "high": round(float(latest["最高"]), 2),
                "low": round(float(latest["最低"]), 2),
                "pct_chg": round(float(latest["pct_chg"]), 2),
                "volume": float(latest["volume"]),
                "amount": float(latest["amount"]),
                "amount_text": _format_amount(float(latest["amount"])),
                "turnover": round(float(latest["turnover"]), 2),
                "volume_ratio_20": round(float(latest["volume_ratio_20"]), 3),
                "amount_ratio_20": round(float(latest["amount_ratio_20"]), 3),
                "ret_3": pct(latest.get("ret_3", 0)),
                "ret_5": pct(latest.get("ret_5", 0)),
                "ret_10": pct(latest.get("ret_10", 0)),
                "ret_20": pct(latest.get("ret_20", 0)),
            },
            "model": score,
            "levels": levels,
            "signals": self.signal_days(feat, start=start),
            "scenarios": self.scenarios(feat, levels),
            "trade_view": self.trade_view(feat, score, levels),
            "recent_bars": self.recent_bars(feat, 24),
        }
        return response

    def recent_bars(self, feat: pd.DataFrame, n: int) -> List[Dict[str, object]]:
        rows = []
        for _, row in feat.tail(n).iterrows():
            rows.append(
                {
                    "date": row["日期"].strftime("%Y-%m-%d"),
                    "open": round(float(row["开盘"]), 2),
                    "close": round(float(row["close"]), 2),
                    "high": round(float(row["最高"]), 2),
                    "low": round(float(row["最低"]), 2),
                    "pct_chg": round(float(row["pct_chg"]), 2),
                    "volume": float(row["volume"]),
                    "amount_text": _format_amount(float(row["amount"])),
                    "turnover": round(float(row["turnover"]), 2),
                    "volume_ratio_20": round(float(row["volume_ratio_20"]), 3),
                    "amount_ratio_20": round(float(row["amount_ratio_20"]), 3),
                }
            )
        return rows


uptrend_service = UptrendAnalysisService()
