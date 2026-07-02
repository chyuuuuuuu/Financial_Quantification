#!/usr/bin/env python
"""Find A-share 5x half-year winners and score current main-uptrend candidates.

This is a research tool, not an investment recommendation engine.  It builds
labels from historical daily bars, with turnover, volume and amount features as
first-class signals.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
import builtins
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

print = partial(builtins.print, flush=True)

try:
    import akshare as ak
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    ak = None
    AKSHARE_IMPORT_ERROR = exc
else:
    AKSHARE_IMPORT_ERROR = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    RandomForestClassifier = None
    classification_report = None
    roc_auc_score = None
    train_test_split = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


FEATURE_COLS = [
    "day_ret",
    "gap_ret",
    "body_pct",
    "amplitude",
    "upper_shadow",
    "lower_shadow",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "volatility_20",
    "volume_ratio_5",
    "volume_ratio_20",
    "volume_ratio_60",
    "amount_ratio_5",
    "amount_ratio_20",
    "amount_ratio_60",
    "amount_pctile_120",
    "turnover",
    "turnover_ratio_20",
    "amount_log",
    "ma5_dist",
    "ma10_dist",
    "ma20_dist",
    "ma60_dist",
    "ma20_slope_10",
    "ma60_slope_20",
    "close_to_high_20",
    "close_to_high_60",
    "close_to_high_120",
    "close_to_low_60",
    "breakout_20",
    "breakout_60",
    "breakout_120",
    "above_ma20",
    "above_ma60",
    "up_days_5",
]

META_COLS = [
    "code",
    "name",
    "date",
    "close",
    "pct_chg",
    "amount",
    "volume",
    "future_max_multiple",
    "future_peak_date",
]


@dataclass
class StockItem:
    code: str
    name: str
    amount: float = 0.0
    turnover: float = 0.0
    pct_chg: float = 0.0
    close: float = 0.0


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_akshare() -> None:
    if ak is None:
        raise RuntimeError(
            "akshare is required. Install it in a Python 3.8+ environment, "
            "for example: /home/luochangyu/anaconda3/envs/py310/bin/python -m pip install akshare"
        ) from AKSHARE_IMPORT_ERROR


def _ensure_sklearn() -> None:
    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is required for model training") from SKLEARN_IMPORT_ERROR


def _to_num(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("--", "", regex=False),
        errors="coerce",
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(str(value).replace(",", "").replace("%", ""))
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _code6(value: object) -> str:
    s = str(value).strip().upper()
    s = s.replace("SH", "").replace("SZ", "").replace("BJ", "")
    return s.zfill(6)[-6:]


def is_hs_main_board_code(value: object) -> bool:
    code = _code6(value)
    return code.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def _format_amount(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f}亿"
    if abs(value) >= 1e4:
        return f"{value / 1e4:.2f}万"
    return f"{value:.0f}"


def _fetch_spot_em_direct(retry: int = 4, page_size: int = 100) -> pd.DataFrame:
    """Fetch Eastmoney A-share spot pages directly with per-page retries.

    akshare wraps the same endpoint but aborts the whole request if one page is
    interrupted.  For this workflow a partial page retry is materially more
    reliable.
    """
    hosts = [
        "82.push2.eastmoney.com",
        "push2.eastmoney.com",
        "16.push2.eastmoney.com",
    ]
    fields = [
        "f2",   # latest
        "f3",   # pct change
        "f4",
        "f5",   # volume
        "f6",   # amount
        "f7",
        "f8",   # turnover
        "f9",
        "f10",
        "f12",  # code
        "f13",
        "f14",  # name
        "f15",
        "f16",
        "f17",
        "f18",
        "f20",
        "f21",
        "f23",
    ]
    params_base = {
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "fields": ",".join(fields),
        "pz": str(page_size),
    }
    field_map = {
        "f2": "最新价",
        "f3": "涨跌幅",
        "f4": "涨跌额",
        "f5": "成交量",
        "f6": "成交额",
        "f7": "振幅",
        "f8": "换手率",
        "f9": "市盈率-动态",
        "f10": "量比",
        "f12": "代码",
        "f13": "市场",
        "f14": "名称",
        "f15": "最高",
        "f16": "最低",
        "f17": "今开",
        "f18": "昨收",
        "f20": "总市值",
        "f21": "流通市值",
        "f23": "市净率",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
    }

    last_error: Optional[Exception] = None
    for host in hosts:
        url = f"https://{host}/api/qt/clist/get"
        rows: List[Dict[str, object]] = []
        total = None
        page = 1
        while total is None or len(rows) < total:
            params = dict(params_base)
            params["pn"] = str(page)
            page_data = None
            for attempt in range(retry):
                try:
                    resp = requests.get(url, params=params, headers=headers, timeout=12)
                    resp.raise_for_status()
                    payload = resp.json()
                    data = payload.get("data") or {}
                    page_data = data.get("diff") or []
                    total = int(data.get("total") or len(page_data))
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(0.6 * (attempt + 1))
            if page_data is None:
                break
            rows.extend(page_data)
            if not page_data or len(page_data) < page_size:
                break
            page += 1
            time.sleep(0.05)
        if rows:
            df = pd.DataFrame(rows).rename(columns=field_map)
            if "代码" in df.columns:
                return df
    raise RuntimeError(f"direct Eastmoney spot fetch failed: {last_error}")


def _latest_trading_start(lookback_years: int) -> Tuple[str, str]:
    end = datetime.now()
    start = end - timedelta(days=int(lookback_years * 365.25) + 30)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def load_spot_universe(cache_dir: Path, cache_hours: float = 2.0, refresh: bool = False) -> pd.DataFrame:
    """Load current A-share spot universe from Eastmoney via akshare."""
    _ensure_akshare()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "spot_universe.csv"
    meta_path = cache_dir / "spot_universe.meta.json"

    if not refresh and cache_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(meta["created_at"])
            if datetime.now() - ts <= timedelta(hours=cache_hours):
                return pd.read_csv(cache_path, dtype={"代码": str})
        except Exception:
            pass

    last_error: Optional[Exception] = None
    df = pd.DataFrame()
    try:
        df = _fetch_spot_em_direct()
    except Exception as exc:
        last_error = exc
        for attempt in range(3):
            try:
                df = ak.stock_zh_a_spot_em()
                break
            except Exception as retry_exc:
                last_error = retry_exc
                time.sleep(1.5 * (attempt + 1))

    if (df is None or df.empty) and cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"代码": str})
        if not cached.empty:
            print(f"[{_now_text()}] using stale spot cache after live fetch failure: {last_error}")
            return cached

    if df is None or df.empty:
        raise RuntimeError(f"spot universe fetch returned no rows: {last_error}")

    df = df.copy()
    if "代码" not in df.columns:
        raise RuntimeError(f"spot data has no 代码 column: {list(df.columns)}")
    df["代码"] = df["代码"].map(_code6)
    if "名称" not in df.columns:
        df["名称"] = df["代码"]

    for col in ["最新价", "涨跌幅", "成交量", "成交额", "换手率"]:
        if col in df.columns:
            df[col] = _to_num(df[col])

    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(
        json.dumps({"created_at": datetime.now().isoformat(), "rows": len(df)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return df


def make_universe(
    spot_df: pd.DataFrame,
    min_amount: float,
    max_stocks: int,
    include_bj: bool,
    main_board_only: bool = False,
) -> List[StockItem]:
    df = spot_df.copy()
    df["代码"] = df["代码"].map(_code6)
    df["名称"] = df["名称"].astype(str)

    if main_board_only:
        mask = df["代码"].map(is_hs_main_board_code)
    else:
        valid_prefix = ("0", "3", "6", "8") if include_bj else ("0", "3", "6")
        mask = df["代码"].str.startswith(valid_prefix)
    mask &= ~df["名称"].str.contains("ST|退|N|C", case=False, na=False)
    if "最新价" in df.columns:
        mask &= df["最新价"].fillna(0) > 1
    if "成交额" in df.columns and min_amount > 0:
        mask &= df["成交额"].fillna(0) >= min_amount
    df = df[mask].copy()

    if "成交额" in df.columns:
        df = df.sort_values("成交额", ascending=False)
    if max_stocks and max_stocks > 0:
        df = df.head(max_stocks)

    items = []
    for _, row in df.iterrows():
        items.append(
            StockItem(
                code=_code6(row.get("代码")),
                name=str(row.get("名称", "")),
                amount=_safe_float(row.get("成交额", 0)),
                turnover=_safe_float(row.get("换手率", 0)),
                pct_chg=_safe_float(row.get("涨跌幅", 0)),
                close=_safe_float(row.get("最新价", 0)),
            )
        )
    return items


def normalize_hist(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    col_map = {
        "date": "日期",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "volume": "成交量",
        "amount": "成交额",
        "turnover": "换手率",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns and v not in df.columns})
    required = ["日期", "开盘", "收盘", "最高", "最低"]
    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    out = df.copy()
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅", "换手率"]:
        if col in out.columns:
            out[col] = _to_num(out[col])
    if "成交量" not in out.columns:
        out["成交量"] = np.nan
    if "成交额" not in out.columns:
        out["成交额"] = out["成交量"].fillna(0) * out["收盘"].fillna(0)
    if "涨跌幅" not in out.columns:
        out["涨跌幅"] = out["收盘"].pct_change() * 100
    if "换手率" not in out.columns:
        out["换手率"] = 0.0

    out = out.dropna(subset=["日期", "开盘", "收盘", "最高", "最低"])
    out = out.sort_values("日期").drop_duplicates("日期").reset_index(drop=True)
    return out


def fetch_hist(
    item: StockItem,
    hist_dir: Path,
    start_date: str,
    end_date: str,
    cache_days: float,
    refresh: bool = False,
    retry: int = 3,
) -> Tuple[StockItem, pd.DataFrame, Optional[str]]:
    """Fetch one symbol's adjusted daily bars, with local CSV cache."""
    _ensure_akshare()
    hist_dir.mkdir(parents=True, exist_ok=True)
    path = hist_dir / f"{item.code}.csv"

    if not refresh and path.exists():
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime <= timedelta(days=cache_days):
                cached = normalize_hist(pd.read_csv(path))
                if not cached.empty:
                    return item, cached, None
        except Exception:
            pass

    last_error = None
    for attempt in range(retry):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=item.code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            df = normalize_hist(raw)
            if df.empty:
                return item, df, "empty"
            df.to_csv(path, index=False, encoding="utf-8-sig")
            return item, df, None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5 * (attempt + 1))

    return item, pd.DataFrame(), last_error


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price, volume, amount and breakout features."""
    df = normalize_hist(df)
    if df.empty or len(df) < 160:
        return pd.DataFrame()

    out = df.copy()
    open_ = out["开盘"].astype(float)
    close = out["收盘"].astype(float)
    high = out["最高"].astype(float)
    low = out["最低"].astype(float)
    volume = out["成交量"].replace(0, np.nan).astype(float)
    amount = out["成交额"].replace(0, np.nan).astype(float)
    turnover = out["换手率"].fillna(0).astype(float)
    prev_close = close.shift(1)

    out["day_ret"] = close / prev_close - 1
    out["gap_ret"] = open_ / prev_close - 1
    out["body_pct"] = close / open_.replace(0, np.nan) - 1
    out["amplitude"] = (high - low) / prev_close.replace(0, np.nan)
    out["upper_shadow"] = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
    out["lower_shadow"] = (np.minimum(open_, close) - low) / close.replace(0, np.nan)

    for win in [3, 5, 10, 20, 60]:
        out[f"ret_{win}"] = close.pct_change(win)

    out["volatility_20"] = close.pct_change().rolling(20).std()
    for win in [5, 20, 60]:
        out[f"volume_ratio_{win}"] = volume / volume.rolling(win).mean().replace(0, np.nan)
        out[f"amount_ratio_{win}"] = amount / amount.rolling(win).mean().replace(0, np.nan)
        prev_volume_base = volume.shift(1).rolling(win).mean().replace(0, np.nan)
        prev_amount_base = amount.shift(1).rolling(win).mean().replace(0, np.nan)
        out[f"volume_ratio_{win}_prev"] = volume / prev_volume_base
        out[f"amount_ratio_{win}_prev"] = amount / prev_amount_base

    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    out["macd_dif"] = ema12 - ema26
    out["macd_dea"] = out["macd_dif"].ewm(span=9, adjust=False, min_periods=1).mean()
    out["macd_hist"] = 2.0 * (out["macd_dif"] - out["macd_dea"])

    amount_min_120 = amount.rolling(120, min_periods=60).min()
    amount_max_120 = amount.rolling(120, min_periods=60).max()
    out["amount_pctile_120"] = (amount - amount_min_120) / (amount_max_120 - amount_min_120).replace(0, np.nan)
    out["turnover"] = turnover
    out["turnover_ratio_20"] = turnover / turnover.replace(0, np.nan).rolling(20).mean().replace(0, np.nan)
    out["amount_log"] = np.log1p(amount.fillna(0))

    for win in [5, 10, 20, 60]:
        ma = close.rolling(win).mean()
        out[f"ma{win}_dist"] = close / ma.replace(0, np.nan) - 1
    out["ma20_slope_10"] = close.rolling(20).mean() / close.rolling(20).mean().shift(10) - 1
    out["ma60_slope_20"] = close.rolling(60).mean() / close.rolling(60).mean().shift(20) - 1

    for win in [20, 60, 120]:
        prior_high = high.shift(1).rolling(win).max()
        out[f"close_to_high_{win}"] = close / prior_high.replace(0, np.nan) - 1
        out[f"breakout_{win}"] = (close >= prior_high * 0.995).astype(int)

    prior_low_60 = low.shift(1).rolling(60).min()
    out["close_to_low_60"] = close / prior_low_60.replace(0, np.nan) - 1
    out["above_ma20"] = (close > close.rolling(20).mean()).astype(int)
    out["above_ma60"] = (close > close.rolling(60).mean()).astype(int)
    out["up_days_5"] = (out["day_ret"] > 0).rolling(5).sum()
    out["pct_chg"] = out["涨跌幅"].where(out["涨跌幅"].notna(), out["day_ret"] * 100)
    out["amount"] = out["成交额"].fillna(0)
    out["volume"] = out["成交量"].fillna(0)
    out["close"] = close
    return out


def future_peak_info(close: pd.Series, start_idx: int, horizon: int) -> Tuple[float, int]:
    end_idx = min(len(close) - 1, start_idx + horizon)
    window = close.iloc[start_idx : end_idx + 1]
    if window.empty or close.iloc[start_idx] <= 0:
        return np.nan, start_idx
    peak_idx = int(window.idxmax())
    multiple = float(window.max() / close.iloc[start_idx])
    return multiple, peak_idx


def future_multiples(close: pd.Series, horizon: int) -> np.ndarray:
    """Future max close / current close, including the current day."""
    future_max = close.iloc[::-1].rolling(horizon + 1, min_periods=1).max().iloc[::-1]
    return (future_max / close.replace(0, np.nan)).to_numpy(dtype=float)


def is_explosive_bull(row: pd.Series, min_ret: float, min_body: float, min_amount_ratio: float, min_volume_ratio: float) -> bool:
    return bool(
        row.get("day_ret", 0) >= min_ret
        and row.get("body_pct", 0) >= min_body
        and row.get("amount_ratio_20", 0) >= min_amount_ratio
        and row.get("volume_ratio_20", 0) >= min_volume_ratio
        and (row.get("breakout_60", 0) == 1 or row.get("close_to_high_60", -1) >= -0.02)
    )


def detect_5x_events(
    feat: pd.DataFrame,
    item: StockItem,
    half_year_days: int,
    multiple: float,
    min_ret: float,
    min_body: float,
    min_amount_ratio: float,
    min_volume_ratio: float,
    min_history: int = 140,
) -> List[Dict[str, object]]:
    """Find one start-day sample per continuous 5x-in-half-year episode."""
    if feat.empty or len(feat) < min_history + half_year_days:
        return []

    close = feat["close"].reset_index(drop=True)
    work = feat.reset_index(drop=True)
    last_start = len(work) - half_year_days - 1
    future_mult = future_multiples(close, half_year_days)
    is_event = future_mult >= multiple
    is_event[:min_history] = False
    if last_start + 1 < len(is_event):
        is_event[last_start + 1 :] = False
    events: List[Dict[str, object]] = []
    idx = min_history
    while idx <= last_start:
        if not is_event[idx]:
            idx += 1
            continue

        block_start = idx
        while idx <= last_start and is_event[idx]:
            idx += 1
        block_end = idx - 1

        _, block_peak_idx = future_peak_info(close, block_start, half_year_days)
        search_start = max(min_history, block_start - 10)
        search_end = min(block_peak_idx, block_start + 35, len(work) - half_year_days - 1)
        start_idx = block_start
        found_explosion = False

        for cand_idx in range(search_start, search_end + 1):
            if future_mult[cand_idx] < multiple:
                continue
            if is_explosive_bull(
                work.iloc[cand_idx],
                min_ret=min_ret,
                min_body=min_body,
                min_amount_ratio=min_amount_ratio,
                min_volume_ratio=min_volume_ratio,
            ):
                start_idx = cand_idx
                found_explosion = True
                break

        if not found_explosion:
            continue

        start_mult, peak_idx = future_peak_info(close, start_idx, half_year_days)
        row = work.iloc[start_idx]
        events.append(
            {
                "code": item.code,
                "name": item.name,
                "date": row["日期"].strftime("%Y-%m-%d"),
                "close": float(row["close"]),
                "pct_chg": float(row["pct_chg"]),
                "amount": float(row["amount"]),
                "volume": float(row["volume"]),
                "future_max_multiple": round(float(start_mult), 4),
                "future_peak_date": work.iloc[peak_idx]["日期"].strftime("%Y-%m-%d"),
                "episode_block_start": work.iloc[block_start]["日期"].strftime("%Y-%m-%d"),
                "explosive_bull_found": int(found_explosion),
                "label": 1,
            }
        )

    return events


def row_to_sample(feat: pd.DataFrame, idx: int, item: StockItem, label: int, future_mult: float, peak_idx: int) -> Dict[str, object]:
    row = feat.iloc[idx]
    sample = {
        "code": item.code,
        "name": item.name,
        "date": row["日期"].strftime("%Y-%m-%d"),
        "close": float(row["close"]),
        "pct_chg": float(row["pct_chg"]),
        "amount": float(row["amount"]),
        "volume": float(row["volume"]),
        "future_max_multiple": round(float(future_mult), 4) if np.isfinite(future_mult) else np.nan,
        "future_peak_date": feat.iloc[peak_idx]["日期"].strftime("%Y-%m-%d") if 0 <= peak_idx < len(feat) else "",
        "label": label,
    }
    for col in FEATURE_COLS:
        sample[col] = _safe_float(row.get(col, 0))
    return sample


def build_samples_for_stock(
    item: StockItem,
    hist: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    feat = add_features(hist)
    if feat.empty:
        return [], [], {"code": item.code, "name": item.name, "status": "no_features"}

    events = detect_5x_events(
        feat,
        item,
        half_year_days=args.half_year_days,
        multiple=args.multiple,
        min_ret=args.big_bull_ret,
        min_body=args.big_bull_body,
        min_amount_ratio=args.min_amount_ratio,
        min_volume_ratio=args.min_volume_ratio,
    )

    positive_indices = set()
    date_to_idx = {row["日期"].strftime("%Y-%m-%d"): idx for idx, row in feat.iterrows()}
    for event in events:
        idx = date_to_idx.get(str(event["date"]))
        if idx is not None:
            positive_indices.add(idx)

    positives = []
    for idx in sorted(positive_indices):
        mult, peak_idx = future_peak_info(feat["close"].reset_index(drop=True), idx, args.half_year_days)
        positives.append(row_to_sample(feat.reset_index(drop=True), idx, item, 1, mult, peak_idx))

    # Hard negatives: similar-looking volume/breakout days that did not produce a 5x run.
    rng = random.Random(args.random_state + int(item.code))
    work = feat.reset_index(drop=True)
    last_start = len(work) - args.half_year_days - 1
    negatives = []
    if last_start >= 140:
        future_mult = future_multiples(work["close"], args.half_year_days)
        eligible = np.arange(140, last_start + 1)
        valid = np.isfinite(future_mult[eligible]) & (future_mult[eligible] < args.multiple)
        if positive_indices:
            positive_arr = np.array(sorted(positive_indices))
            valid &= ~np.isin(eligible, positive_arr)
        eligible = eligible[valid]

        hard_mask = (
            (work["day_ret"].to_numpy(dtype=float) >= max(0.03, args.big_bull_ret * 0.65))
            & (work["body_pct"].to_numpy(dtype=float) >= max(0.02, args.big_bull_body * 0.65))
            & (work["amount_ratio_20"].to_numpy(dtype=float) >= max(1.4, args.min_amount_ratio * 0.65))
            & (work["volume_ratio_20"].to_numpy(dtype=float) >= max(1.4, args.min_volume_ratio * 0.65))
            & ((work["breakout_60"].to_numpy(dtype=float) == 1) | (work["close_to_high_60"].to_numpy(dtype=float) >= -0.02))
        )
        hard_indices = [int(x) for x in eligible[hard_mask[eligible]]]
        other_indices = [int(x) for x in eligible[~hard_mask[eligible]]]
        rng.shuffle(hard_indices)
        rng.shuffle(other_indices)
        selected_indices = hard_indices[: args.max_hard_negatives_per_stock]
        selected_indices += other_indices[: max(0, args.max_random_negatives_per_stock - len(selected_indices))]

        for idx in selected_indices:
            mult, peak_idx = future_peak_info(work["close"], idx, args.half_year_days)
            negatives.append(row_to_sample(work, idx, item, 0, mult, peak_idx))

    info = {
        "code": item.code,
        "name": item.name,
        "status": "ok",
        "rows": len(work),
        "positives": len(positives),
        "negatives": len(negatives),
        "latest_date": work.iloc[-1]["日期"].strftime("%Y-%m-%d"),
    }
    return positives, negatives, info


def rule_score(row: pd.Series) -> float:
    def clip01(x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        return max(0.0, min(1.0, float(x)))

    score = 0.0
    score += 16 * clip01((row.get("day_ret", 0) - 0.015) / 0.075)
    score += 10 * clip01((row.get("body_pct", 0) - 0.01) / 0.06)
    score += 16 * clip01((row.get("amount_ratio_20", 0) - 1.1) / 3.0)
    score += 12 * clip01((row.get("volume_ratio_20", 0) - 1.1) / 3.0)
    score += 12 * clip01((row.get("amount_pctile_120", 0) - 0.55) / 0.45)
    score += 12 * clip01((row.get("close_to_high_60", -0.3) + 0.06) / 0.08)
    score += 8 * clip01((row.get("ret_20", 0) + 0.03) / 0.22)
    score += 6 * clip01((row.get("ma20_slope_10", 0) + 0.01) / 0.08)
    score += 4 * clip01((row.get("turnover", 0) - 2.0) / 10.0)
    score += 4 * clip01((row.get("above_ma20", 0) + row.get("above_ma60", 0)) / 2.0)
    return round(score, 2)


def reason_text(row: pd.Series) -> str:
    reasons = []
    if row.get("amount_ratio_20", 0) >= 2:
        reasons.append(f"成交额{row.get('amount_ratio_20', 0):.1f}倍")
    if row.get("volume_ratio_20", 0) >= 2:
        reasons.append(f"成交量{row.get('volume_ratio_20', 0):.1f}倍")
    if row.get("day_ret", 0) >= 0.055 and row.get("body_pct", 0) >= 0.035:
        reasons.append("大阳线")
    if row.get("breakout_60", 0) == 1:
        reasons.append("突破60日高点")
    elif row.get("close_to_high_60", -1) >= -0.02:
        reasons.append("贴近60日高点")
    if row.get("ret_20", 0) > 0.08:
        reasons.append(f"20日涨幅{row.get('ret_20', 0) * 100:.1f}%")
    if row.get("turnover", 0) >= 8:
        reasons.append(f"换手{row.get('turnover', 0):.1f}%")
    return "、".join(reasons[:5]) or "量价结构接近历史样本"


def ready_signal(row: pd.Series) -> Tuple[int, str]:
    """Current-bar filter for a tradable start-day signal."""
    day_ret = float(row.get("day_ret", 0) or 0)
    body_pct = float(row.get("body_pct", 0) or 0)
    amount_ratio = float(row.get("amount_ratio_20", 0) or 0)
    volume_ratio = float(row.get("volume_ratio_20", 0) or 0)
    close_to_high_60 = float(row.get("close_to_high_60", -1) or -1)
    turnover = float(row.get("turnover", 0) or 0)
    breakout = int(row.get("breakout_60", 0) or 0)
    above_ma20 = int(row.get("above_ma20", 0) or 0)

    explosive = (
        day_ret >= 0.055
        and body_pct >= 0.035
        and amount_ratio >= 1.6
        and volume_ratio >= 1.5
        and turnover >= 3.0
    )
    breakout_push = (
        day_ret >= 0.025
        and body_pct >= 0.015
        and amount_ratio >= 1.8
        and volume_ratio >= 1.6
        and close_to_high_60 >= -0.04
        and above_ma20 == 1
    )
    volume_turn = (
        day_ret >= 0.0
        and amount_ratio >= 2.2
        and volume_ratio >= 2.0
        and turnover >= 6.0
        and close_to_high_60 >= -0.08
    )
    if explosive:
        return 1, "爆量大阳"
    if breakout and breakout_push:
        return 1, "放量突破"
    if breakout_push:
        return 1, "放量近突破"
    if volume_turn:
        return 1, "放量换手"
    return 0, "观察"


def score_latest(
    item: StockItem,
    hist: pd.DataFrame,
    model: Optional[RandomForestClassifier],
    feature_medians: Optional[pd.Series],
) -> Optional[Dict[str, object]]:
    feat = add_features(hist)
    if feat.empty:
        return None
    row = feat.iloc[-1].copy()
    if row[FEATURE_COLS].isna().all():
        return None

    x = pd.DataFrame([row[FEATURE_COLS]], columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan)
    if feature_medians is not None:
        x = x.fillna(feature_medians)
    x = x.fillna(0)

    model_prob = np.nan
    if model is not None:
        try:
            model_prob = float(model.predict_proba(x)[0][1]) * 100
        except Exception:
            model_prob = np.nan

    rs = rule_score(row)
    ready, ready_type = ready_signal(row)
    final = rs if not np.isfinite(model_prob) else 0.62 * model_prob + 0.38 * rs
    return {
        "code": item.code,
        "name": item.name,
        "latest_date": row["日期"].strftime("%Y-%m-%d"),
        "close": round(float(row["close"]), 3),
        "pct_chg": round(float(row["pct_chg"]), 2),
        "amount": float(row["amount"]),
        "amount_text": _format_amount(float(row["amount"])),
        "turnover": round(float(row.get("turnover", 0)), 2),
        "day_ret": round(float(row.get("day_ret", 0)) * 100, 2),
        "body_pct": round(float(row.get("body_pct", 0)) * 100, 2),
        "ret_20": round(float(row.get("ret_20", 0)) * 100, 2),
        "volume_ratio_20": round(float(row.get("volume_ratio_20", 0)), 2),
        "amount_ratio_20": round(float(row.get("amount_ratio_20", 0)), 2),
        "close_to_high_60": round(float(row.get("close_to_high_60", 0)) * 100, 2),
        "breakout_60": int(row.get("breakout_60", 0)),
        "rule_score": rs,
        "model_prob": round(model_prob, 2) if np.isfinite(model_prob) else None,
        "final_score": round(float(final), 2),
        "ready_signal": ready,
        "ready_type": ready_type,
        "reason": reason_text(row),
    }


def fetch_histories(items: List[StockItem], args: argparse.Namespace, out_dir: Path) -> Dict[str, Tuple[StockItem, pd.DataFrame]]:
    hist_dir = out_dir / "hist"
    start_date, end_date = _latest_trading_start(args.lookback_years)
    histories: Dict[str, Tuple[StockItem, pd.DataFrame]] = {}
    failures = []

    print(f"[{_now_text()}] fetching history: {len(items)} stocks, {start_date}-{end_date}, workers={args.workers}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                fetch_hist,
                item,
                hist_dir,
                start_date,
                end_date,
                args.cache_days,
                args.refresh_history,
                args.retry,
            ): item
            for item in items
        }
        done = 0
        for fut in as_completed(future_map):
            item, hist, error = fut.result()
            done += 1
            if error or hist.empty:
                failures.append({"code": item.code, "name": item.name, "error": error or "empty"})
            else:
                histories[item.code] = (item, hist)
            if done % args.progress_every == 0 or done == len(items):
                print(f"[{_now_text()}] history progress {done}/{len(items)}, ok={len(histories)}, failed={len(failures)}")
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "history_failures.csv", index=False, encoding="utf-8-sig")
    return histories


def train_model(samples: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> Tuple[Optional[RandomForestClassifier], pd.Series, Dict[str, object]]:
    _ensure_sklearn()
    clean = samples.copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["label"])
    y = clean["label"].astype(int)
    feature_medians = clean[FEATURE_COLS].median(numeric_only=True).fillna(0)
    X = clean[FEATURE_COLS].fillna(feature_medians).fillna(0)
    positives = int(y.sum())
    negatives = int((y == 0).sum())

    summary: Dict[str, object] = {
        "samples": int(len(clean)),
        "positives": positives,
        "negatives": negatives,
        "feature_cols": FEATURE_COLS,
        "warning": None,
    }
    if positives < args.min_positive_samples or negatives < max(positives, 20):
        summary["warning"] = (
            f"positive/negative samples are limited: positives={positives}, negatives={negatives}; "
            "scores should be treated as pattern matching, not a robust classifier"
        )

    if positives < 2 or negatives < 2:
        return None, feature_medians, summary

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=args.random_state,
        n_jobs=-1,
    )

    can_test = positives >= 5 and negatives >= 20 and len(clean) >= 60
    if can_test:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=args.random_state,
            stratify=y,
        )
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        try:
            summary["roc_auc"] = round(float(roc_auc_score(y_test, prob)), 4)
        except Exception:
            summary["roc_auc"] = None
        summary["classification_report"] = classification_report(y_test, pred, output_dict=True, zero_division=0)
    else:
        model.fit(X, y)
        summary["roc_auc"] = None
        summary["classification_report"] = None

    model.fit(X, y)
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        summary["feature_importance"] = [
            {"feature": col, "importance": round(float(val), 6)}
            for col, val in sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)[:20]
        ]

    model_path = Path("model_cache") / "main_uptrend_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(pickle.dumps({"model": model, "feature_medians": feature_medians, "summary": summary}))
    summary["model_path"] = str(model_path)
    return model, feature_medians, summary


def print_table(df: pd.DataFrame, cols: List[str], limit: int) -> None:
    if df.empty:
        print("(empty)")
        return
    printable = df[cols].head(limit).copy()
    print(printable.to_string(index=False))


def run(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = out_dir.parent / "model_cache"
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.random_state)
    np.random.seed(args.random_state)

    print(f"[{_now_text()}] loading spot universe")
    spot = load_spot_universe(out_dir, cache_hours=args.spot_cache_hours, refresh=args.refresh_spot)
    items = make_universe(spot, min_amount=args.min_amount, max_stocks=args.max_stocks, include_bj=args.include_bj)
    if not items:
        raise RuntimeError("No stocks left after universe filters")
    pd.DataFrame([item.__dict__ for item in items]).to_csv(out_dir / "universe.csv", index=False, encoding="utf-8-sig")
    print(f"[{_now_text()}] universe after filters: {len(items)} stocks")

    histories = fetch_histories(items, args, out_dir)
    if not histories:
        raise RuntimeError("No historical bars fetched")

    positives: List[Dict[str, object]] = []
    negatives: List[Dict[str, object]] = []
    infos: List[Dict[str, object]] = []
    latest_scores: List[Dict[str, object]] = []

    print(f"[{_now_text()}] detecting 5x half-year events and building samples")
    for pos, (code, (item, hist)) in enumerate(histories.items(), start=1):
        ps, ns, info = build_samples_for_stock(item, hist, args)
        positives.extend(ps)
        negatives.extend(ns)
        infos.append(info)
        if pos % args.progress_every == 0 or pos == len(histories):
            print(
                f"[{_now_text()}] sample progress {pos}/{len(histories)}, "
                f"positives={len(positives)}, negatives={len(negatives)}"
            )

    pos_df = pd.DataFrame(positives)
    neg_df = pd.DataFrame(negatives)
    info_df = pd.DataFrame(infos)
    info_df.to_csv(out_dir / "stock_processing_summary.csv", index=False, encoding="utf-8-sig")

    if not pos_df.empty:
        pos_df.sort_values(["date", "future_max_multiple"], ascending=[False, False]).to_csv(
            out_dir / "detected_5x_events.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame(columns=META_COLS + FEATURE_COLS + ["label"]).to_csv(
            out_dir / "detected_5x_events.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if pos_df.empty:
        samples = neg_df.copy()
    else:
        target_neg = min(len(neg_df), max(args.min_negative_samples, len(pos_df) * args.negative_ratio))
        if target_neg > 0 and len(neg_df) > target_neg:
            neg_df = neg_df.sample(target_neg, random_state=args.random_state)
        samples = pd.concat([pos_df, neg_df], ignore_index=True)

    if samples.empty:
        raise RuntimeError("No training samples were built")
    samples.to_csv(out_dir / "training_samples.csv", index=False, encoding="utf-8-sig")

    print(f"[{_now_text()}] training samples: positives={len(pos_df)}, negatives={len(samples) - len(pos_df)}")
    model, feature_medians, train_summary = train_model(samples, args, out_dir)

    print(f"[{_now_text()}] scoring latest bars")
    for pos, (code, (item, hist)) in enumerate(histories.items(), start=1):
        scored = score_latest(item, hist, model, feature_medians)
        if scored is not None:
            latest_scores.append(scored)
        if pos % args.progress_every == 0 or pos == len(histories):
            print(f"[{_now_text()}] scoring progress {pos}/{len(histories)}, scored={len(latest_scores)}")

    pred_df = pd.DataFrame(latest_scores)
    if not pred_df.empty and args.candidate_min_amount > 0:
        pred_df = pred_df[pred_df["amount"] >= args.candidate_min_amount].copy()
    if not pred_df.empty and args.candidate_ready_only:
        pred_df = pred_df[pred_df["ready_signal"] == 1].copy()
    if not pred_df.empty:
        pred_df = pred_df.sort_values(["final_score", "amount"], ascending=[False, False]).reset_index(drop=True)
        pred_df.insert(0, "rank", np.arange(1, len(pred_df) + 1))
    pred_df.to_csv(out_dir / "current_main_uptrend_candidates.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_time": datetime.now().isoformat(),
        "universe_size": len(items),
        "history_ok": len(histories),
        "positive_5x_events": len(pos_df),
        "training_samples": int(len(samples)),
        "prediction_rows": int(len(pred_df)),
        "args": vars(args),
        "train_summary": train_summary,
        "outputs": {
            "universe": str(out_dir / "universe.csv"),
            "detected_5x_events": str(out_dir / "detected_5x_events.csv"),
            "training_samples": str(out_dir / "training_samples.csv"),
            "candidates": str(out_dir / "current_main_uptrend_candidates.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 半年内5倍历史样本 Top ===")
    if pos_df.empty:
        print("没有在当前股票池/参数下识别到 5x 半年样本。")
    else:
        show_events = pos_df.sort_values(["date", "future_max_multiple"], ascending=[False, False])
        print_table(
            show_events,
            ["code", "name", "date", "pct_chg", "amount", "future_max_multiple", "future_peak_date"],
            args.print_top,
        )

    print("\n=== 当前主升候选 Top ===")
    if pred_df.empty:
        print("没有生成候选。")
    else:
        print_table(
            pred_df,
            [
                "rank",
                "code",
                "name",
                "latest_date",
                "pct_chg",
                "amount_text",
                "turnover",
                "amount_ratio_20",
                "volume_ratio_20",
                "breakout_60",
                "ready_type",
                "model_prob",
                "rule_score",
                "final_score",
                "reason",
            ],
            args.print_top,
        )

    print(f"\n[{_now_text()}] outputs written under {out_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data_cache/main_uptrend", help="Output/cache directory")
    parser.add_argument("--lookback-years", type=int, default=6, help="Historical years to fetch")
    parser.add_argument("--half-year-days", type=int, default=120, help="Trading-day horizon for half-year label")
    parser.add_argument("--multiple", type=float, default=5.0, help="Future max close multiple for positive labels")
    parser.add_argument("--min-amount", type=float, default=50_000_000, help="Current spot amount filter for fetched universe in RMB")
    parser.add_argument("--candidate-min-amount", type=float, default=50_000_000, help="Latest amount filter for final candidates in RMB")
    parser.add_argument("--candidate-ready-only", action="store_true", help="Only keep current bars with volume/bullish start-day signals")
    parser.add_argument("--max-stocks", type=int, default=1200, help="Max stocks by current amount; 0 means all")
    parser.add_argument("--include-bj", action="store_true", help="Include Beijing exchange symbols")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent history fetch workers")
    parser.add_argument("--retry", type=int, default=3, help="Fetch retry count")
    parser.add_argument("--cache-days", type=float, default=1.0, help="History cache max age in days")
    parser.add_argument("--spot-cache-hours", type=float, default=2.0, help="Spot cache max age in hours")
    parser.add_argument("--refresh-spot", action="store_true", help="Ignore spot cache")
    parser.add_argument("--refresh-history", action="store_true", help="Ignore history cache")
    parser.add_argument("--big-bull-ret", type=float, default=0.055, help="Explosive bull daily return threshold")
    parser.add_argument("--big-bull-body", type=float, default=0.035, help="Explosive bull real-body threshold")
    parser.add_argument("--min-amount-ratio", type=float, default=2.0, help="20-day amount burst threshold")
    parser.add_argument("--min-volume-ratio", type=float, default=2.0, help="20-day volume burst threshold")
    parser.add_argument("--max-hard-negatives-per-stock", type=int, default=8)
    parser.add_argument("--max-random-negatives-per-stock", type=int, default=3)
    parser.add_argument("--negative-ratio", type=int, default=8)
    parser.add_argument("--min-negative-samples", type=int, default=200)
    parser.add_argument("--min-positive-samples", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=50, help="Kept for API compatibility; CSV contains all scored rows")
    parser.add_argument("--print-top", type=int, default=30, help="Rows to print to console")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
