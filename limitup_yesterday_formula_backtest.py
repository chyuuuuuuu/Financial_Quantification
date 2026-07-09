#!/usr/bin/env python
"""Backtest limit-up formula signals bought on the signal close."""

from __future__ import annotations

import argparse
import builtins
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from formula_breakout_backtest import max_drawdown, round_or_none
from formula_breakout_cash_backtest import fee_breakdown, prepare_history
from formula_breakout_pipeline import clean_records, load_universe, macd, normalize_code, now_text

print = partial(builtins.print, flush=True)


@dataclass
class Holding:
    lot_id: int
    slot: int
    code: str
    name: str
    shares: int
    entry_date: str
    signal_date: str
    entry_trade_index: int
    entry_price: float
    entry_open: float
    entry_fee: float
    entry_gross: float
    entry_rank: int
    entry_score: float


def row_float(row: pd.Series, field: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def bearish_doji(row: pd.Series) -> bool:
    open_ = row_float(row, "开盘")
    close = row_float(row, "收盘")
    high = row_float(row, "最高")
    low = row_float(row, "最低")
    span = high - low
    return close < open_ and span > 0 and abs(close - open_) < span * 0.20


def volume_bearish(row: pd.Series) -> bool:
    open_ = row_float(row, "开盘")
    close = row_float(row, "收盘")
    volume = row_float(row, "成交量")
    prev_volume = row_float(row, "prev_volume")
    return close < open_ and prev_volume > 0 and volume > prev_volume


def two_bearish(row: pd.Series) -> bool:
    open_ = row_float(row, "开盘")
    close = row_float(row, "收盘")
    prev_open = row_float(row, "prev_open")
    prev_close = row_float(row, "prev_close")
    return close < open_ and prev_close < prev_open


def sell_decision(row: pd.Series, holding: Holding, trade_index: int) -> Dict[str, object]:
    close = row_float(row, "收盘")
    days_after_entry = trade_index - holding.entry_trade_index
    if days_after_entry == 1:
        stop_price = holding.entry_open
        stop_reason = "第二天收盘价跌破买入日开盘价止损"
    else:
        stop_price = holding.entry_price
        stop_reason = "第三天后收盘价跌破买入价止损"
    if days_after_entry >= 1 and stop_price > 0 and close < stop_price:
        return {
            "reasons": [stop_reason],
            "price": close,
            "execution_time": "收盘止损",
            "trigger_price": stop_price,
        }

    reasons: List[str] = []
    if volume_bearish(row):
        reasons.append("放量阴线")
    if bearish_doji(row):
        reasons.append("阴线十字星")
    if two_bearish(row):
        reasons.append("连续两根阴线")
    return {
        "reasons": reasons,
        "price": close,
        "execution_time": "收盘信号卖出" if reasons else "收盘",
        "trigger_price": None,
    }


def max_affordable_shares(price: float, budget: float, args: argparse.Namespace) -> Tuple[int, float, Dict[str, float]]:
    if price <= 0 or budget <= 0:
        return 0, 0.0, fee_breakdown(0.0, "buy", args)
    lots = int(budget // (price * args.lot_size))
    while lots > 0:
        shares = lots * args.lot_size
        gross = price * shares
        fees = fee_breakdown(gross, "buy", args)
        cost = gross + fees["total_fee"]
        if cost <= budget + 1e-9:
            return shares, cost, fees
        lots -= 1
    return 0, 0.0, fee_breakdown(0.0, "buy", args)


def asof_close(history: pd.DataFrame, date_text: str) -> Optional[float]:
    date = pd.Timestamp(date_text)
    dates = history["date"].to_numpy(dtype="datetime64[ns]")
    idx = int(np.searchsorted(dates, np.datetime64(date), side="right") - 1)
    if idx < 0:
        return None
    close = float(history.iloc[idx]["收盘"])
    return close if math.isfinite(close) else None


def bar_for_date(history: pd.DataFrame, date_text: str) -> Optional[pd.Series]:
    matched = history[history["date_text"] == date_text]
    if matched.empty:
        return None
    return matched.iloc[0]


def strict_mainboard_limit_up(close_y: float, close_prev: float) -> bool:
    if close_prev <= 0:
        return False
    rounded_ratio = round((close_y / close_prev) * 100.0) / 100.0
    return rounded_ratio >= 1.098 and rounded_ratio < 1.105


def score_signal(
    n0: int,
    volume_vs_prev: float,
    vol_ma10_vs_triple: float,
    dif_y: float,
    dea_y: float,
    triple_close: float,
    breakout_gap_pct: float,
    max_close_vs_triple: float,
) -> float:
    base_score = 45.0
    contraction_score = max(0.0, min(25.0, (1.0 - min(vol_ma10_vs_triple, 1.0)) * 35.0))
    triple_score = max(0.0, min(15.0, (volume_vs_prev - 3.0) * 4.0))
    timing_score = max(0.0, min(12.0, 12.0 * (1.0 - (float(n0) - 2.0) / 118.0)))
    macd_score = max(0.0, min(12.0, (dif_y - dea_y) / triple_close * 1200.0)) if triple_close > 0 else 0.0
    breakout_score = max(0.0, min(10.0, breakout_gap_pct * 2.0))
    heat_penalty = max(0.0, min(8.0, max_close_vs_triple - 8.0))
    return round(base_score + contraction_score + triple_score + timing_score + macd_score + breakout_score - heat_penalty, 3)


def evaluate_yesterday_limitup_formula(
    code: str,
    name: str,
    hist: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    buy_next_trading_day: bool = False,
) -> Tuple[List[Dict[str, object]], List[str]]:
    if hist.empty or len(hist) < 35:
        return [], []
    hist = hist.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    open_ = pd.to_numeric(hist["开盘"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(hist["收盘"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(hist["最高"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(hist["最低"], errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(hist["成交量"], errors="coerce").fillna(0).to_numpy(dtype=float)
    amount = pd.to_numeric(hist["成交额"], errors="coerce").fillna(0).to_numpy(dtype=float)
    pct_chg = pd.to_numeric(hist["涨跌幅"], errors="coerce").fillna(0).to_numpy(dtype=float)
    date_text = hist["date_text"].astype(str).tolist()
    dates = hist["date"]

    market_dates = hist[(dates >= start) & (dates <= end)]["date_text"].astype(str).tolist()
    if not market_dates:
        return [], []

    valid_price = np.isfinite(open_) & np.isfinite(close) & np.isfinite(high) & np.isfinite(low)
    vol_ma5 = pd.Series(volume).rolling(5, min_periods=5).mean().to_numpy(dtype=float)
    vol_ma10 = pd.Series(volume).rolling(10, min_periods=10).mean().to_numpy(dtype=float)
    ma20 = pd.Series(close).rolling(20, min_periods=20).mean().to_numpy(dtype=float)
    dif, dea = macd(pd.Series(close))
    dif_arr = dif.to_numpy(dtype=float)
    dea_arr = dea.to_numpy(dtype=float)

    prev_volume = np.r_[np.nan, volume[:-1]]
    triple_mask = (volume > 3.0 * prev_volume) & (close > open_) & valid_price
    rows: List[Dict[str, object]] = []

    for signal_idx in range(1, len(hist)):
        signal_date = pd.Timestamp(dates.iloc[signal_idx])
        if signal_date < start or signal_date > end:
            continue
        y = signal_idx
        if not (valid_price[y] and valid_price[y - 1]):
            continue
        triple_candidates = np.flatnonzero(triple_mask[: y + 1])
        if len(triple_candidates) == 0:
            continue
        triple_idx = int(triple_candidates[-1])
        n0 = y - triple_idx
        if n0 < 2 or n0 > 120:
            continue

        triple_close = float(close[triple_idx])
        triple_volume = float(volume[triple_idx])
        if triple_close <= 0 or triple_volume <= 0:
            continue
        if not (vol_ma5[y] < triple_volume and vol_ma10[y] < triple_volume):
            continue

        wash_start = triple_idx + 1
        wash_end = y + 1
        if wash_start >= wash_end:
            continue
        max_close = float(np.nanmax(close[wash_start:wash_end]))
        washed = bool(np.any(close[wash_start:wash_end] < ma20[wash_start:wash_end]))
        if not (max_close < triple_close * 1.15 and washed):
            continue
        if not (dif_arr[y] > dea_arr[y] and dif_arr[y] > dif_arr[y - 1] and dea_arr[y] > dea_arr[y - 1]):
            continue
        if not (close[y - 1] < triple_close and close[y] >= triple_close and close[y] > open_[y]):
            continue
        if not strict_mainboard_limit_up(float(close[y]), float(close[y - 1])):
            continue

        buy_idx = y + 1 if buy_next_trading_day else y
        if buy_idx >= len(hist):
            continue
        buy_date = pd.Timestamp(dates.iloc[buy_idx])
        if buy_date < start or buy_date > end:
            continue
        if not valid_price[buy_idx]:
            continue

        volume_vs_prev = float(volume[triple_idx] / volume[triple_idx - 1]) if triple_idx > 0 and volume[triple_idx - 1] > 0 else 0.0
        vol_ma5_vs_triple = float(vol_ma5[y] / triple_volume) if triple_volume > 0 else float("nan")
        vol_ma10_vs_triple = float(vol_ma10[y] / triple_volume) if triple_volume > 0 else float("nan")
        breakout_gap_pct = (float(close[y]) / triple_close - 1.0) * 100.0
        prior_close_gap_pct = (float(close[y - 1]) / triple_close - 1.0) * 100.0
        max_close_vs_triple = (max_close / triple_close - 1.0) * 100.0
        score = score_signal(
            n0,
            volume_vs_prev,
            vol_ma10_vs_triple,
            float(dif_arr[y]),
            float(dea_arr[y]),
            triple_close,
            breakout_gap_pct,
            max_close_vs_triple,
        )
        rows.append(
            {
                "snapshot_date": date_text[buy_idx],
                "signal_date": date_text[y],
                "code": normalize_code(code),
                "name": name,
                "rank": None,
                "buy_open": round(float(open_[buy_idx]), 4),
                "buy_high": round(float(high[buy_idx]), 4),
                "buy_low": round(float(low[buy_idx]), 4),
                "buy_close": round(float(close[buy_idx]), 4),
                "signal_close": round(float(close[y]), 4),
                "signal_open": round(float(open_[y]), 4),
                "signal_pct_chg": round(float(pct_chg[y]), 4),
                "amount": round(float(amount[y]), 2),
                "volume": round(float(volume[y]), 2),
                "formula_score": score,
                "triple_date": date_text[triple_idx],
                "triple_close": round(triple_close, 4),
                "triple_volume": round(triple_volume, 2),
                "bars_since_triple": int(n0),
                "volume_vs_prev": round(volume_vs_prev, 4),
                "vol_ma5_vs_triple": round(vol_ma5_vs_triple, 4),
                "vol_ma10_vs_triple": round(vol_ma10_vs_triple, 4),
                "max_close_vs_triple": round(max_close_vs_triple, 4),
                "breakout_gap_pct": round(breakout_gap_pct, 4),
                "prior_close_gap_pct": round(prior_close_gap_pct, 4),
                "macd_dif": round(float(dif_arr[y]), 6),
                "macd_dea": round(float(dea_arr[y]), 6),
                "reason": "当日涨停且满足三倍阳缩量回调突破公式，次日开盘买入" if buy_next_trading_day else "当日涨停且满足三倍阳缩量回调突破公式，假设涨停收盘可买入",
            }
        )
    return rows, market_dates


def build_signals_and_histories(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str], int]:
    universe = load_universe(Path(args.universe_file))
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    all_rows: List[Dict[str, object]] = []
    histories: Dict[str, pd.DataFrame] = {}
    market_dates: set[str] = set()
    for idx, item in enumerate(universe.itertuples(index=False), start=1):
        code = normalize_code(item.code)
        path = Path(args.history_dir) / f"{code}.csv"
        if not path.exists():
            continue
        hist = prepare_history(path)
        if hist is None or hist.empty:
            continue
        rows, dates = evaluate_yesterday_limitup_formula(
            code,
            str(item.name),
            hist,
            start,
            end,
            buy_next_trading_day=bool(getattr(args, "buy_next_trading_day", False)),
        )
        histories[code] = hist
        all_rows.extend(rows)
        market_dates.update(dates)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"[{now_text()}] yesterday-limit signals {idx}/{len(universe)}, signals={len(all_rows)}")
    signals = pd.DataFrame(all_rows)
    if not signals.empty:
        signals = signals.sort_values(["snapshot_date", "formula_score", "amount"], ascending=[True, False, False]).reset_index(drop=True)
        signals["rank"] = signals.groupby("snapshot_date").cumcount() + 1
    return signals, histories, sorted(market_dates), int(len(universe))


def summarize_operations(items: Sequence[Dict[str, object]], max_items: int = 8) -> str:
    labels: List[str] = []
    for item in items[:max_items]:
        action = item.get("action")
        if action == "buy":
            labels.append(f"买{item.get('code')}{item.get('name')}{item.get('shares')}股@{item.get('price')}")
        elif action == "sell":
            labels.append(f"卖{item.get('code')}{item.get('name')}{item.get('shares')}股@{item.get('price')}({item.get('reason')})")
    if len(items) > max_items:
        labels.append(f"...另{len(items) - max_items}笔")
    return "；".join(labels)


def simulate(args: argparse.Namespace) -> Dict[str, object]:
    signals, histories, market_dates, universe_count = build_signals_and_histories(args)
    signals_by_date = {date: df.sort_values("rank") for date, df in signals.groupby("snapshot_date")} if not signals.empty else {}
    cash = float(args.initial_cash)
    lot_id = 0
    holdings: List[Holding] = []
    daily_rows: List[Dict[str, object]] = []
    operations_by_day: List[Dict[str, object]] = []
    closed_returns: List[float] = []
    realized_pnl = 0.0
    total_fees = 0.0
    total_commission = 0.0
    total_stamp_tax = 0.0
    total_transfer_fee = 0.0
    total_buys = 0
    total_sells = 0
    total_limit_up_skips = 0
    prev_equity = cash

    for trade_index, date_text in enumerate(market_dates):
        cash_start = cash
        day_ops: List[Dict[str, object]] = []
        day_fees = 0.0
        sold_codes: set[str] = set()
        remaining: List[Holding] = []
        for holding in holdings:
            if date_text <= holding.entry_date:
                remaining.append(holding)
                continue
            hist = histories.get(holding.code)
            row = bar_for_date(hist, date_text) if hist is not None else None
            if row is None:
                remaining.append(holding)
                continue
            decision = sell_decision(row, holding, trade_index)
            reasons = list(decision["reasons"])
            if not reasons:
                remaining.append(holding)
                continue
            price = float(decision["price"])
            gross = price * holding.shares
            fees = fee_breakdown(gross, "sell", args)
            proceeds = gross - fees["total_fee"]
            basis = holding.entry_gross + holding.entry_fee
            pnl = proceeds - basis
            ret_pct = pnl / basis * 100.0 if basis > 0 else 0.0
            cash += proceeds
            realized_pnl += pnl
            total_fees += fees["total_fee"]
            total_commission += fees["commission"]
            total_stamp_tax += fees["stamp_tax"]
            total_transfer_fee += fees["transfer_fee"]
            day_fees += fees["total_fee"]
            total_sells += 1
            sold_codes.add(holding.code)
            closed_returns.append(ret_pct)
            day_ops.append(
                {
                    "action": "sell",
                    "date": date_text,
                    "slot": holding.slot,
                    "lot_id": holding.lot_id,
                    "code": holding.code,
                    "name": holding.name,
                    "shares": holding.shares,
                    "price": round(price, 4),
                    "gross_amount": round(gross, 2),
                    "fee": round(fees["total_fee"], 2),
                    "commission": round(fees["commission"], 2),
                    "stamp_tax": round(fees["stamp_tax"], 2),
                    "transfer_fee": round(fees["transfer_fee"], 4),
                    "amount": round(proceeds, 2),
                    "entry_date": holding.entry_date,
                    "signal_date": holding.signal_date,
                    "entry_price": round(holding.entry_price, 4),
                    "entry_open": round(holding.entry_open, 4),
                    "entry_fee": round(holding.entry_fee, 2),
                    "pnl": round(pnl, 2),
                    "ret_pct": round(ret_pct, 4),
                    "reason": "；".join(reasons),
                    "execution_time": decision["execution_time"],
                    "trigger_price": round(float(decision["trigger_price"]), 4) if decision["trigger_price"] is not None else None,
                }
            )
        holdings = remaining

        raw_candidates = signals_by_date.get(date_text, pd.DataFrame())
        selected_count = int(len(raw_candidates)) if not raw_candidates.empty else 0
        empty_slots = [slot for slot in range(1, int(args.slots) + 1) if all(h.slot != slot for h in holdings)]
        held_codes = {h.code for h in holdings}
        bought_codes = set(held_codes)
        skipped = 0
        limit_up_skips = 0
        first_skipped_rank: Optional[int] = None
        for slot in empty_slots:
            remaining_slots = int(args.slots) - len(holdings)
            if remaining_slots <= 0 or cash <= 0 or raw_candidates.empty:
                break
            budget = cash / remaining_slots
            bought = False
            for signal in raw_candidates.itertuples(index=False):
                code = normalize_code(signal.code)
                if code in bought_codes or code in sold_codes:
                    continue
                if getattr(args, "block_limit_up_buys", False):
                    skipped += 1
                    limit_up_skips += 1
                    total_limit_up_skips += 1
                    first_skipped_rank = first_skipped_rank or int(signal.rank)
                    day_ops.append(
                        {
                            "action": "skip",
                            "date": date_text,
                            "slot": slot,
                            "code": code,
                            "name": str(signal.name),
                            "rank": int(signal.rank),
                            "price": round(float(signal.buy_close), 4),
                            "signal_date": str(signal.signal_date),
                            "formula_score": round(float(signal.formula_score), 3),
                            "reason": "涨停不可买入",
                        }
                    )
                    continue
                price = float(signal.buy_open) if getattr(args, "buy_next_trading_day", False) else float(signal.buy_close)
                shares, cost, fees = max_affordable_shares(price, budget, args)
                if shares <= 0:
                    skipped += 1
                    first_skipped_rank = first_skipped_rank or int(signal.rank)
                    continue
                gross = price * shares
                lot_id += 1
                cash -= cost
                total_fees += fees["total_fee"]
                total_commission += fees["commission"]
                total_transfer_fee += fees["transfer_fee"]
                day_fees += fees["total_fee"]
                holding = Holding(
                    lot_id=lot_id,
                    slot=slot,
                    code=code,
                    name=str(signal.name),
                    shares=shares,
                    entry_date=date_text,
                    signal_date=str(signal.signal_date),
                    entry_trade_index=trade_index,
                    entry_price=price,
                    entry_open=float(signal.buy_open),
                    entry_fee=fees["total_fee"],
                    entry_gross=gross,
                    entry_rank=int(signal.rank),
                    entry_score=float(signal.formula_score),
                )
                holdings.append(holding)
                bought_codes.add(code)
                total_buys += 1
                bought = True
                day_ops.append(
                    {
                        "action": "buy",
                        "date": date_text,
                        "slot": slot,
                        "lot_id": holding.lot_id,
                        "code": holding.code,
                        "name": holding.name,
                        "shares": holding.shares,
                        "price": round(price, 4),
                        "gross_amount": round(gross, 2),
                        "fee": round(fees["total_fee"], 2),
                        "commission": round(fees["commission"], 2),
                        "stamp_tax": 0.0,
                        "transfer_fee": round(fees["transfer_fee"], 4),
                        "amount": round(cost, 2),
                        "rank": holding.entry_rank,
                        "formula_score": round(holding.entry_score, 3),
                        "signal_date": holding.signal_date,
                        "buy_open": round(float(signal.buy_open), 4),
                        "buy_close": round(float(signal.buy_close), 4),
                        "slot_budget": round(budget, 2),
                        "execution_time": "次日开盘" if getattr(args, "buy_next_trading_day", False) else "涨停日收盘",
                    }
                )
                break
            if not bought and not raw_candidates.empty:
                skipped += 1

        market_value = 0.0
        unrealized_pnl = 0.0
        estimated_open_sell_fees = 0.0
        open_holdings: List[Dict[str, object]] = []
        for holding in holdings:
            hist = histories.get(holding.code)
            close = asof_close(hist, date_text) if hist is not None else None
            close = close if close is not None else holding.entry_price
            value = close * holding.shares
            sell_fees = fee_breakdown(value, "sell", args)
            pnl = value - (holding.entry_gross + holding.entry_fee)
            market_value += value
            unrealized_pnl += pnl
            estimated_open_sell_fees += sell_fees["total_fee"]
            open_holdings.append(
                {
                    **asdict(holding),
                    "last_close": round(float(close), 4),
                    "market_value": round(value, 2),
                    "estimated_sell_fee": round(sell_fees["total_fee"], 2),
                    "unrealized_pnl": round(pnl, 2),
                    "unrealized_pnl_after_sell_fee": round(pnl - sell_fees["total_fee"], 2),
                }
            )
        equity = cash + market_value
        daily_ret = (equity / prev_equity - 1.0) * 100.0 if prev_equity > 0 else 0.0
        prev_equity = equity
        daily_row = {
            "date": date_text,
            "selected_count": selected_count,
            "buy_count": sum(1 for op in day_ops if op["action"] == "buy"),
            "sell_count": sum(1 for op in day_ops if op["action"] == "sell"),
            "skipped_count": skipped,
            "limit_up_skip_count": limit_up_skips,
            "first_skipped_rank": first_skipped_rank,
            "cash_start": round(cash_start, 2),
            "cash_end": round(cash, 2),
            "market_value": round(market_value, 2),
            "equity": round(equity, 2),
            "daily_ret_pct": round(daily_ret, 4),
            "holdings_count": len(holdings),
            "fees_paid_day": round(day_fees, 2),
            "fees_paid_total": round(total_fees, 2),
            "realized_pnl_total": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "operations": summarize_operations(day_ops),
        }
        daily_rows.append(daily_row)
        if day_ops:
            operations_by_day.append({**daily_row, "items": day_ops})

    equity_series = pd.Series([row["equity"] for row in daily_rows], dtype=float)
    final_equity = float(equity_series.iloc[-1]) if not equity_series.empty else cash
    summary = {
        "initial_cash": float(args.initial_cash),
        "final_equity": round(final_equity, 2),
        "final_liquidation_equity": round(final_equity - estimated_open_sell_fees, 2),
        "final_cash": round(cash, 2),
        "final_market_value": round(final_equity - cash, 2),
        "profit": round(final_equity - float(args.initial_cash), 2),
        "liquidation_profit": round(final_equity - estimated_open_sell_fees - float(args.initial_cash), 2),
        "return_pct": round((final_equity / float(args.initial_cash) - 1.0) * 100.0, 4),
        "liquidation_return_pct": round(((final_equity - estimated_open_sell_fees) / float(args.initial_cash) - 1.0) * 100.0, 4),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pnl_after_sell_fee": round(unrealized_pnl - estimated_open_sell_fees, 2),
        "total_fees": round(total_fees, 2),
        "total_commission": round(total_commission, 2),
        "total_stamp_tax": round(total_stamp_tax, 2),
        "total_transfer_fee": round(total_transfer_fee, 2),
        "estimated_open_sell_fees": round(estimated_open_sell_fees, 2),
        "max_drawdown_pct": max_drawdown(equity_series),
        "total_buys": total_buys,
        "total_sells": total_sells,
        "blocked_limit_up_buys": total_limit_up_skips,
        "open_lots": len(holdings),
        "closed_win_rate_pct": round(float(np.mean(np.array(closed_returns) > 0) * 100.0), 4) if closed_returns else None,
        "avg_closed_ret_pct": round(float(np.mean(closed_returns)), 4) if closed_returns else None,
        "best_closed_ret_pct": round(float(np.max(closed_returns)), 4) if closed_returns else None,
        "worst_closed_ret_pct": round(float(np.min(closed_returns)), 4) if closed_returns else None,
        "best_equity": round(float(equity_series.max()), 2) if not equity_series.empty else round(cash, 2),
        "worst_equity": round(float(equity_series.min()), 2) if not equity_series.empty else round(cash, 2),
        "trading_days": len(market_dates),
        "signal_days": int(sum(1 for row in daily_rows if row["selected_count"] > 0)),
        "buy_days": int(sum(1 for row in daily_rows if row["buy_count"] > 0)),
        "sell_days": int(sum(1 for row in daily_rows if row["sell_count"] > 0)),
        "avg_holdings_count": round(float(np.mean([row["holdings_count"] for row in daily_rows])), 4) if daily_rows else 0.0,
    }
    buy_next_day = bool(getattr(args, "buy_next_trading_day", False))
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "port": "limitup_formula_signal_next_open_backtest" if buy_next_day else "limitup_formula_signal_close_backtest",
        "strategy_name": "涨停公式次日开盘买入Top3" if buy_next_day else "涨停当天公式收盘买入Top3",
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "universe_count": universe_count,
        "initial_cash": float(args.initial_cash),
        "slots": int(args.slots),
        "lot_size": int(args.lot_size),
        "assumption": (
            "运行日T使用当日收盘后数据选股；T日必须严格涨停并完整满足三倍阳缩量回调突破公式；"
            + (
                "假设涨停日不可买入，信号顺延到T+1交易日并按T+1开盘价成交；"
                if buy_next_day
                else ("假设涨停不可买入，所有涨停信号均跳过；" if getattr(args, "block_limit_up_buys", False) else "假设涨停当天收盘可以买入，按T日收盘价成交；")
            )
            + "买入后第二个交易日若收盘价低于买入当天开盘价则按收盘价止损；从第三个交易日开始，若收盘价低于买入价则按收盘价止损；未止损时，放量阴线、阴线十字星、连续两根阴线任一出现即按收盘价卖出。"
        ),
        "buy_block_rule": {
            "block_limit_up_buys": bool(getattr(args, "block_limit_up_buys", False)),
            "buy_next_trading_day": buy_next_day,
            "description": "涨停日不可买入时，信号顺延到次一交易日开盘买入。" if buy_next_day else "涨停不可买入时，本策略所有信号都因严格涨停条件被跳过。",
        },
        "sell_rules": {
            "t_plus_one": "T+1交易；买入日当日不卖出。",
            "stop_loss": "最高优先级：买入后第二个交易日收盘价低于买入当天开盘价时止损；从第三个交易日开始，收盘价低于买入价时止损；均按触发日收盘价卖出。",
            "volume_bearish": "放量阴线：C<O且V>REF(V,1)，按收盘价卖出。",
            "bearish_doji": "阴线十字星：C<O且实体占当日高低振幅比例小于20%，按收盘价卖出。",
            "two_bearish": "连续两根阴线：当日C<O且前一交易日C<O，按收盘价卖出。",
        },
        "fee_model": {
            "commission_rate": args.commission_rate,
            "min_commission": args.min_commission,
            "stamp_tax_rate_sell_only": args.stamp_tax_rate,
            "transfer_fee_rate_both_sides": args.transfer_fee_rate,
        },
        "summary": summary,
        "daily": daily_rows,
        "operations": operations_by_day,
        "open_holdings": open_holdings,
        "signals_total": int(len(signals)),
        "signals": clean_records(signals.head(int(args.signal_display_limit))) if not signals.empty else [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.daily_csv:
        path = Path(args.daily_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(daily_rows).to_csv(path, index=False, encoding="utf-8-sig")
    if args.operations_csv:
        path = Path(args.operations_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat_ops = [item for day in operations_by_day for item in day["items"]]
        pd.DataFrame(flat_ops).to_csv(path, index=False, encoding="utf-8-sig")
    if args.signals_csv:
        path = Path(args.signals_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[{now_text()}] wrote {output}, return={summary['return_pct']}%, profit={summary['profit']}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-07-08")
    parser.add_argument("--initial-cash", type=float, default=50000.0)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--block-limit-up-buys", action="store_true")
    parser.add_argument("--buy-next-trading-day", action="store_true")
    parser.add_argument("--universe-file", default="data_cache/volume_contraction_screen_20260701_mainboard_entry_close/refresh_status.csv")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--output", default="static/reports/limitup_formula_signal_close_backtest_2026.json")
    parser.add_argument("--daily-csv", default="data_cache/formula_breakout_backtests/limitup_formula_signal_close_2026_daily.csv")
    parser.add_argument("--operations-csv", default="data_cache/formula_breakout_backtests/limitup_formula_signal_close_2026_operations.csv")
    parser.add_argument("--signals-csv", default="data_cache/formula_breakout_backtests/limitup_formula_signal_close_2026_signals.csv")
    parser.add_argument("--signal-display-limit", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    simulate(parse_args())


if __name__ == "__main__":
    main()
