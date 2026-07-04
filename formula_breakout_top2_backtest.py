#!/usr/bin/env python
"""N-slot full-position backtest for the formula breakout selector."""

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

from formula_breakout_backtest import round_or_none
from formula_breakout_cash_backtest import (
    bar_for_date,
    build_signals_and_histories,
    fee_breakdown,
    sell_decision,
)
from formula_breakout_pipeline import clean_records, normalize_code, now_text

print = partial(builtins.print, flush=True)


@dataclass
class SlotHolding:
    lot_id: int
    slot: int
    code: str
    name: str
    shares: int
    entry_date: str
    entry_price: float
    entry_open: float
    entry_fee: float
    entry_gross: float
    entry_rank: int
    entry_score: float


def asof_close(history: pd.DataFrame, date_text: str) -> Optional[float]:
    date = pd.Timestamp(date_text)
    dates = history["date"].to_numpy(dtype="datetime64[ns]")
    idx = int(np.searchsorted(dates, np.datetime64(date), side="right") - 1)
    if idx < 0:
        return None
    close = float(history.iloc[idx]["收盘"])
    return close if math.isfinite(close) else None


def max_affordable_shares(price: float, budget: float, args: argparse.Namespace) -> Tuple[int, float, Dict[str, float]]:
    if price <= 0 or budget <= 0:
        return 0, 0.0, fee_breakdown(0.0, "buy", args)
    max_lots = int(budget // (price * args.lot_size))
    while max_lots > 0:
        shares = max_lots * args.lot_size
        gross = price * shares
        fees = fee_breakdown(gross, "buy", args)
        cost = gross + fees["total_fee"]
        if cost <= budget + 1e-9:
            return shares, cost, fees
        max_lots -= 1
    return 0, 0.0, fee_breakdown(0.0, "buy", args)


def signal_float(signal: object, field: str) -> float:
    try:
        value = getattr(signal, field)
    except AttributeError:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_zt_pool_date(date_text: str) -> str:
    return str(date_text)[:10].replace("-", "")


def load_limit_up_pool_for_date(date_text: str, args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    data_start = str(getattr(args, "limit_up_data_start_date", "") or "").strip()
    if data_start and pd.Timestamp(date_text) < pd.Timestamp(data_start):
        return {}
    cache_dir = Path(args.limit_up_pool_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    date_key = normalize_zt_pool_date(date_text)
    path = cache_dir / f"{date_key}.csv"
    if path.exists() and not getattr(args, "refresh_limit_up_pool", False):
        try:
            df = pd.read_csv(path, dtype={"代码": str})
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
    else:
        import akshare as ak  # Lazy import so normal local backtests do not require network data.

        df = ak.stock_zt_pool_em(date=date_key)
        if df.empty:
            df = pd.DataFrame(columns=["代码", "名称", "封板资金", "首次封板时间", "最后封板时间", "炸板次数", "涨停统计", "连板数"])
        df.to_csv(path, index=False, encoding="utf-8-sig")

    pool: Dict[str, Dict[str, object]] = {}
    if df.empty:
        return pool
    for row in df.to_dict(orient="records"):
        code = normalize_code(row.get("代码"))
        if not code:
            continue
        seal_amount = float(pd.to_numeric(pd.Series([row.get("封板资金")]), errors="coerce").iloc[0] or 0.0)
        if seal_amount < float(args.min_seal_amount):
            continue
        pool[code] = {
            "seal_amount": seal_amount,
            "first_limit_time": row.get("首次封板时间"),
            "last_limit_time": row.get("最后封板时间"),
            "break_count": row.get("炸板次数"),
            "limit_stat": row.get("涨停统计"),
            "limit_boards": row.get("连板数"),
            "source": "eastmoney_zt_pool_em",
        }
    return pool


def load_limit_up_pools(market_dates: Sequence[str], args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, object]]]:
    if not getattr(args, "block_limit_up_buys", False) or args.limit_up_source != "zt_pool":
        return {}
    pools: Dict[str, Dict[str, Dict[str, object]]] = {}
    for idx, date_text in enumerate(market_dates, start=1):
        pools[date_text] = load_limit_up_pool_for_date(date_text, args)
        if args.progress_every > 0 and idx % 40 == 0:
            print(f"[{now_text()}] loaded zt pools {idx}/{len(market_dates)}")
    return pools


def is_limit_up_buy_blocked(
    signal: object,
    args: argparse.Namespace,
    day_limit_up_pool: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[bool, Dict[str, object]]:
    if not getattr(args, "block_limit_up_buys", False):
        return False, {}
    code = normalize_code(getattr(signal, "code", ""))
    if args.limit_up_source == "zt_pool":
        meta = (day_limit_up_pool or {}).get(code)
        return bool(meta), dict(meta or {})

    close = signal_float(signal, "close")
    high = signal_float(signal, "high")
    pct_chg = signal_float(signal, "pct_chg")
    if not (math.isfinite(close) and math.isfinite(high) and math.isfinite(pct_chg)):
        return False, {}
    if close <= 0 or high <= 0:
        return False, {}
    near_day_high = close >= high * float(args.limit_close_high_ratio)
    blocked = pct_chg >= float(args.limit_up_pct) and near_day_high
    return blocked, {"source": "daily_ohlcv_limit_proxy"} if blocked else {}


def summarize_operations(items: Sequence[Dict[str, object]], max_items: int = 8) -> str:
    if not items:
        return ""
    labels: List[str] = []
    for item in items[:max_items]:
        action = item.get("action")
        code = item.get("code")
        name = item.get("name")
        shares = item.get("shares")
        price = item.get("price")
        if action == "buy":
            labels.append(f"买{code}{name}{shares}股@{price}")
        elif action == "sell":
            labels.append(f"卖{code}{name}{shares}股@{price}({item.get('reason')})")
        elif action == "skip":
            labels.append(f"跳{code}{name}({item.get('reason')})")
    if len(items) > max_items:
        labels.append(f"...另{len(items) - max_items}笔")
    return "；".join(labels)


def current_market_value(holdings: Sequence[SlotHolding], histories: Dict[str, pd.DataFrame], date_text: str) -> Tuple[float, float]:
    market_value = 0.0
    unrealized = 0.0
    for holding in holdings:
        hist = histories.get(holding.code)
        close = asof_close(hist, date_text) if hist is not None else None
        close = close if close is not None else holding.entry_price
        market_value += close * holding.shares
        unrealized += close * holding.shares - (holding.entry_gross + holding.entry_fee)
    return market_value, unrealized


def simulate(args: argparse.Namespace) -> Dict[str, object]:
    prebuilt = getattr(args, "_prebuilt_data", None)
    if prebuilt is None:
        signals, histories, market_dates, start, end, universe_count = build_signals_and_histories(args)
    else:
        signals, histories, market_dates, start, end, universe_count = prebuilt
    if not market_dates:
        raise ValueError("no trading dates available for top2 backtest")
    signals_by_date = {date: df.sort_values(["rank"]) for date, df in signals.groupby("snapshot_date")} if not signals.empty else {}
    limit_up_pools = getattr(args, "_limit_up_pools", None)
    if limit_up_pools is None:
        limit_up_pools = load_limit_up_pools(market_dates, args)
    data_start = str(getattr(args, "limit_up_data_start_date", "") or "").strip()
    skipped_limit_source_days = (
        int(sum(1 for date_text in market_dates if pd.Timestamp(date_text) < pd.Timestamp(data_start)))
        if data_start and getattr(args, "block_limit_up_buys", False) and args.limit_up_source == "zt_pool"
        else 0
    )
    requested_limit_source_days = (
        len(market_dates) - skipped_limit_source_days
        if getattr(args, "block_limit_up_buys", False) and args.limit_up_source == "zt_pool"
        else 0
    )
    nonempty_limit_source_days = int(sum(1 for pool in limit_up_pools.values() if pool)) if limit_up_pools else 0

    cash = float(args.initial_cash)
    holdings: List[SlotHolding] = []
    lot_id = 0
    realized_pnl = 0.0
    total_fees = 0.0
    total_commission = 0.0
    total_stamp_tax = 0.0
    total_transfer_fee = 0.0
    total_buys = 0
    total_sells = 0
    total_limit_up_skips = 0
    closed_returns: List[float] = []
    daily_rows: List[Dict[str, object]] = []
    operations_by_day: List[Dict[str, object]] = []
    prev_equity = cash

    for date_text in market_dates:
        cash_start = cash
        day_ops: List[Dict[str, object]] = []
        day_fees = 0.0
        sold_codes: set[str] = set()

        remaining: List[SlotHolding] = []
        for holding in holdings:
            hist = histories.get(holding.code)
            row = bar_for_date(hist, date_text) if hist is not None else None
            if row is None:
                remaining.append(holding)
                continue
            decision = sell_decision(row, holding)  # type: ignore[arg-type]
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
            cash += proceeds
            realized_pnl += pnl
            total_fees += fees["total_fee"]
            total_commission += fees["commission"]
            total_stamp_tax += fees["stamp_tax"]
            total_transfer_fee += fees["transfer_fee"]
            day_fees += fees["total_fee"]
            total_sells += 1
            sold_codes.add(holding.code)
            ret_pct = pnl / basis * 100.0 if basis > 0 else 0.0
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

        selected_count = int(len(signals_by_date.get(date_text, []))) if date_text in signals_by_date else 0
        empty_slots = [slot for slot in range(1, args.slots + 1) if all(h.slot != slot for h in holdings)]
        skipped = 0
        limit_up_skips = 0
        first_skipped_rank: Optional[int] = None
        bought_codes = {holding.code for holding in holdings}
        blocked_buy_codes: set[str] = set()
        limit_up_blocked_labels: List[str] = []
        candidates = signals_by_date.get(date_text, pd.DataFrame())
        day_limit_up_pool = limit_up_pools.get(date_text, {}) if limit_up_pools else {}

        for slot in empty_slots:
            remaining_slots = args.slots - len(holdings)
            if remaining_slots <= 0 or cash <= 0:
                break
            budget = cash / remaining_slots
            bought = False
            for signal in candidates.itertuples(index=False):
                code = normalize_code(signal.code)
                if code in bought_codes or code in sold_codes or code in blocked_buy_codes:
                    continue
                blocked, block_meta = is_limit_up_buy_blocked(signal, args, day_limit_up_pool)
                if blocked:
                    skipped += 1
                    limit_up_skips += 1
                    total_limit_up_skips += 1
                    blocked_buy_codes.add(code)
                    first_skipped_rank = first_skipped_rank or int(signal.rank)
                    seal_amount = float(block_meta.get("seal_amount") or 0.0)
                    label = f"{int(signal.rank)}.{code}{signal.name}"
                    if seal_amount > 0:
                        label += f" 封单{seal_amount / 10000:.0f}万"
                    limit_up_blocked_labels.append(label)
                    day_ops.append(
                        {
                            "action": "skip",
                            "date": date_text,
                            "slot": slot,
                            "code": code,
                            "name": str(signal.name),
                            "rank": int(signal.rank),
                            "price": round(float(signal.close), 4),
                            "pct_chg": round(signal_float(signal, "pct_chg"), 4),
                            "seal_amount": round(seal_amount, 2),
                            "first_limit_time": block_meta.get("first_limit_time"),
                            "last_limit_time": block_meta.get("last_limit_time"),
                            "break_count": block_meta.get("break_count"),
                            "reason": "尾盘涨停不可买入",
                        }
                    )
                    continue
                price = float(signal.close)
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
                holding = SlotHolding(
                    lot_id=lot_id,
                    slot=slot,
                    code=code,
                    name=str(signal.name),
                    shares=shares,
                    entry_date=date_text,
                    entry_price=price,
                    entry_open=float(signal.open),
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
                        "entry_open": round(holding.entry_open, 4),
                        "slot_budget": round(budget, 2),
                        "execution_time": "收盘集合竞价/15:00",
                    }
                )
                break
            if not bought and not candidates.empty:
                skipped += 1

        market_value, unrealized_pnl = current_market_value(holdings, histories, date_text)
        equity = cash + market_value
        daily_ret = (equity / prev_equity - 1.0) * 100.0 if prev_equity > 0 else 0.0
        prev_equity = equity
        buys = [item for item in day_ops if item["action"] == "buy"]
        sells = [item for item in day_ops if item["action"] == "sell"]
        day = {
            "date": date_text,
            "selected_count": selected_count,
            "buy_count": len(buys),
            "sell_count": len(sells),
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
            "limit_up_blocked": "；".join(limit_up_blocked_labels[:8]),
            "operations": summarize_operations(day_ops),
        }
        daily_rows.append(day)
        operations_by_day.append({**day, "items": day_ops})

    final_equity = float(daily_rows[-1]["equity"])
    equity_series = pd.Series([row["equity"] for row in daily_rows], dtype=float)
    peak = equity_series.cummax()
    drawdown = equity_series / peak - 1.0
    closed = pd.Series(closed_returns, dtype=float)

    liquidation_value = cash
    liquidation_fees = 0.0
    open_holdings = []
    for holding in holdings:
        hist = histories.get(holding.code)
        close = asof_close(hist, market_dates[-1]) if hist is not None else None
        close = close if close is not None else holding.entry_price
        gross = close * holding.shares
        fees = fee_breakdown(gross, "sell", args)
        liquidation_value += gross - fees["total_fee"]
        liquidation_fees += fees["total_fee"]
        open_holdings.append(
            {
                **asdict(holding),
                "last_close": round(close, 4),
                "market_value": round(gross, 2),
                "estimated_sell_fee": round(fees["total_fee"], 2),
                "net_liquidation_value": round(gross - fees["total_fee"], 2),
                "unrealized_pnl": round(gross - (holding.entry_gross + holding.entry_fee), 2),
                "unrealized_pnl_after_sell_fee": round(gross - fees["total_fee"] - (holding.entry_gross + holding.entry_fee), 2),
                "unrealized_ret_pct": round((gross / (holding.entry_gross + holding.entry_fee) - 1.0) * 100.0, 4)
                if holding.entry_gross + holding.entry_fee > 0
                else 0.0,
            }
        )

    daily_df = pd.DataFrame(daily_rows)
    daily_df["month"] = daily_df["date"].str.slice(0, 7)
    monthly = daily_df.groupby("month", as_index=False).agg(
        trading_days=("date", "count"),
        buy_count=("buy_count", "sum"),
        sell_count=("sell_count", "sum"),
        selected_count=("selected_count", "sum"),
        ending_equity=("equity", "last"),
        avg_daily_ret_pct=("daily_ret_pct", "mean"),
    )
    monthly["month_ret_pct"] = daily_df.groupby("month")["daily_ret_pct"].apply(lambda x: ((1.0 + x / 100.0).prod() - 1.0) * 100.0).values
    for col in ["ending_equity", "avg_daily_ret_pct", "month_ret_pct"]:
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce").round(4)

    daily_df["year"] = daily_df["date"].str.slice(0, 4)
    yearly = daily_df.groupby("year", as_index=False).agg(
        trading_days=("date", "count"),
        buy_count=("buy_count", "sum"),
        sell_count=("sell_count", "sum"),
        selected_count=("selected_count", "sum"),
        limit_up_skip_count=("limit_up_skip_count", "sum"),
        fees_paid=("fees_paid_day", "sum"),
        ending_equity=("equity", "last"),
        avg_daily_ret_pct=("daily_ret_pct", "mean"),
        min_equity=("equity", "min"),
        max_equity=("equity", "max"),
    )
    yearly["year_ret_pct"] = (
        daily_df.groupby("year")["daily_ret_pct"]
        .apply(lambda x: ((1.0 + x / 100.0).prod() - 1.0) * 100.0)
        .values
    )
    yearly["starting_equity"] = yearly["ending_equity"] / (1.0 + yearly["year_ret_pct"] / 100.0)
    yearly["profit"] = yearly["ending_equity"] - yearly["starting_equity"]
    for col in [
        "fees_paid",
        "ending_equity",
        "avg_daily_ret_pct",
        "min_equity",
        "max_equity",
        "year_ret_pct",
        "starting_equity",
        "profit",
    ]:
        yearly[col] = pd.to_numeric(yearly[col], errors="coerce").round(4)

    full_slot_days = int(sum(1 for row in daily_rows if row["holdings_count"] >= args.slots))
    summary = {
        "initial_cash": round(float(args.initial_cash), 2),
        "final_equity": round(final_equity, 2),
        "final_liquidation_equity": round(liquidation_value, 2),
        "final_cash": round(cash, 2),
        "final_market_value": round(final_equity - cash, 2),
        "profit": round(final_equity - float(args.initial_cash), 2),
        "liquidation_profit": round(liquidation_value - float(args.initial_cash), 2),
        "return_pct": round((final_equity / float(args.initial_cash) - 1.0) * 100.0, 4),
        "liquidation_return_pct": round((liquidation_value / float(args.initial_cash) - 1.0) * 100.0, 4),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(sum(item["unrealized_pnl"] for item in open_holdings), 2),
        "unrealized_pnl_after_sell_fee": round(sum(item["unrealized_pnl_after_sell_fee"] for item in open_holdings), 2),
        "total_fees": round(total_fees, 2),
        "total_costs": round(total_fees, 2),
        "total_commission": round(total_commission, 2),
        "total_stamp_tax": round(total_stamp_tax, 2),
        "total_transfer_fee": round(total_transfer_fee, 2),
        "estimated_open_sell_fees": round(liquidation_fees, 2),
        "total_costs_with_estimated_exit": round(total_fees + liquidation_fees, 2),
        "max_drawdown_pct": round(float(drawdown.min() * 100.0), 4),
        "total_buys": total_buys,
        "total_sells": total_sells,
        "blocked_limit_up_buys": total_limit_up_skips,
        "limit_up_pool_requested_days": requested_limit_source_days,
        "limit_up_pool_nonempty_days": nonempty_limit_source_days,
        "limit_up_pool_skipped_no_source_days": skipped_limit_source_days,
        "open_lots": len(open_holdings),
        "closed_win_rate_pct": round_or_none((closed > 0).mean() * 100.0 if not closed.empty else None, 2),
        "avg_closed_ret_pct": round_or_none(closed.mean() if not closed.empty else None, 4),
        "best_closed_ret_pct": round_or_none(closed.max() if not closed.empty else None, 4),
        "worst_closed_ret_pct": round_or_none(closed.min() if not closed.empty else None, 4),
        "best_equity": round(float(equity_series.max()), 2),
        "worst_equity": round(float(equity_series.min()), 2),
        "trading_days": len(market_dates),
        "signal_days": int(sum(1 for row in daily_rows if row["selected_count"] > 0)),
        "buy_days": int(sum(1 for row in daily_rows if row["buy_count"] > 0)),
        "sell_days": int(sum(1 for row in daily_rows if row["sell_count"] > 0)),
        "full_slot_days": full_slot_days,
        "two_holding_days": full_slot_days,
        "avg_holdings_count": round(float(pd.Series([row["holdings_count"] for row in daily_rows]).mean()), 4),
    }

    slot_label = f"Top{int(args.slots)}"
    limit_block_enabled = bool(getattr(args, "block_limit_up_buys", False))
    if limit_block_enabled and args.limit_up_source == "zt_pool":
        buy_block_text = (
            f"若候选股出现在东方财富当日涨停股池且封板资金>={args.min_seal_amount:.0f}，"
            "则视为尾盘涨停无法买入并顺延下一名。"
        )
    elif limit_block_enabled:
        buy_block_text = f"若候选股涨跌幅>={args.limit_up_pct}%且收盘接近当日最高价，则视为尾盘涨停无法买入并顺延下一名。"
    else:
        buy_block_text = "未模拟涨跌停无法成交影响。"
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "port": f"formula_breakout_top{int(args.slots)}" + ("_limit" if limit_block_enabled else "") + "_backtest",
        "strategy_name": f"公式选股{slot_label}满仓轮动" + ("（封单过滤）" if limit_block_enabled else ""),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "universe_count": universe_count,
        "initial_cash": float(args.initial_cash),
        "lot_size": int(args.lot_size),
        "slots": int(args.slots),
        "assumption": (
            "每天先按当日K线处理已有仓位；若盘中最低价严格低于实际买入价，则视为触发止损，"
            "并按买入价卖出；其他卖出规则按收盘价卖出。"
            "空出的仓位槽用当日公式评分从高到低补齐，买入按收盘集合竞价/15:00的收盘价近似。"
            "买入日不回看买入前盘中低点。"
            "多个空槽时现金按剩余槽位均分，各槽尽量满额买入整手；已持有或当天刚卖出的股票不重复买回。"
            + buy_block_text
        ),
        "sell_rules": {
            "stop_loss": "后续交易日盘中最低价严格低于实际买入价时触发止损，并按买入价卖出。",
            "volume_bearish": "放量阴线：C<O 且 V>REF(V,1)，按收盘价卖出。",
            "doji": "十字星：实体不超过当日高低振幅的10%，不区分阴阳，按收盘价卖出。",
            "long_upper_bearish": "上长阴线：阴线且上影线至少为实体1.5倍，并不短于下影线，按收盘价卖出。",
            "bearish_engulfing": "阴包阳实体吞没：当日阴线实体完全吞没前一交易日阳线实体，按收盘价卖出。",
        },
        "buy_block_rule": {
            "block_limit_up_buys": bool(getattr(args, "block_limit_up_buys", False)),
            "limit_up_source": args.limit_up_source,
            "limit_up_pool_dir": args.limit_up_pool_dir,
            "limit_up_data_start_date": str(getattr(args, "limit_up_data_start_date", "") or ""),
            "min_seal_amount": float(args.min_seal_amount),
            "limit_up_pct": float(args.limit_up_pct),
            "limit_close_high_ratio": float(args.limit_close_high_ratio),
            "description": (
                "东方财富历史涨停股池：出现于当日涨停股池且封板资金达到阈值，判定为尾盘涨停不可买入。"
                if args.limit_up_source == "zt_pool"
                else "日线近似：涨跌幅达到阈值且收盘接近最高价，判定为尾盘涨停不可买入。"
            ),
        },
        "fee_model": {
            "commission_rate": args.commission_rate,
            "min_commission": args.min_commission,
            "stamp_tax_rate_sell_only": args.stamp_tax_rate,
            "transfer_fee_rate_both_sides": args.transfer_fee_rate,
        },
        "summary": summary,
        "daily": daily_rows,
        "monthly": clean_records(monthly),
        "yearly": clean_records(yearly),
        "operations": operations_by_day,
        "open_holdings": open_holdings,
        "signals_total": int(len(signals)),
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
    print(f"[{now_text()}] wrote {output}, return={summary['return_pct']}%, profit={summary['profit']}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--initial-cash", type=float, default=50000.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--block-limit-up-buys", action="store_true")
    parser.add_argument("--limit-up-source", choices=["zt_pool", "daily"], default="zt_pool")
    parser.add_argument("--limit-up-pool-dir", default="data_cache/formula_breakout_backtests/zt_pool_em")
    parser.add_argument("--limit-up-data-start-date", default="")
    parser.add_argument("--refresh-limit-up-pool", action="store_true")
    parser.add_argument("--min-seal-amount", type=float, default=1.0)
    parser.add_argument("--limit-up-pct", type=float, default=9.8)
    parser.add_argument("--limit-close-high-ratio", type=float, default=0.999)
    parser.add_argument("--universe-file", default="data_cache/volume_contraction_screen_20260701_mainboard_entry_close/refresh_status.csv")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--output", default="static/reports/formula_breakout_top2_backtest_1y.json")
    parser.add_argument("--daily-csv", default="data_cache/formula_breakout_backtests/formula_breakout_top2_backtest_1y_daily.csv")
    parser.add_argument("--operations-csv", default="data_cache/formula_breakout_backtests/formula_breakout_top2_backtest_1y_operations.csv")
    parser.add_argument("--progress-every", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    simulate(parse_args())


if __name__ == "__main__":
    main()
