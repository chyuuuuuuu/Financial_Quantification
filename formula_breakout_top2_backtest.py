#!/usr/bin/env python
"""Two-slot full-position backtest for the formula breakout selector."""

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
    sell_reasons,
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
    signals, histories, market_dates, start, end, universe_count = build_signals_and_histories(args)
    if not market_dates:
        raise ValueError("no trading dates available for top2 backtest")
    signals_by_date = {date: df.sort_values(["rank"]) for date, df in signals.groupby("snapshot_date")} if not signals.empty else {}

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
            reasons = sell_reasons(row, holding)  # type: ignore[arg-type]
            if not reasons:
                remaining.append(holding)
                continue
            price = float(row["收盘"])
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
                }
            )
        holdings = remaining

        selected_count = int(len(signals_by_date.get(date_text, []))) if date_text in signals_by_date else 0
        empty_slots = [slot for slot in range(1, args.slots + 1) if all(h.slot != slot for h in holdings)]
        skipped = 0
        first_skipped_rank: Optional[int] = None
        bought_codes = {holding.code for holding in holdings}
        candidates = signals_by_date.get(date_text, pd.DataFrame())

        for slot in empty_slots:
            remaining_slots = args.slots - len(holdings)
            if remaining_slots <= 0 or cash <= 0:
                break
            budget = cash / remaining_slots
            bought = False
            for signal in candidates.itertuples(index=False):
                code = normalize_code(signal.code)
                if code in bought_codes or code in sold_codes:
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
        "total_commission": round(total_commission, 2),
        "total_stamp_tax": round(total_stamp_tax, 2),
        "total_transfer_fee": round(total_transfer_fee, 2),
        "estimated_open_sell_fees": round(liquidation_fees, 2),
        "max_drawdown_pct": round(float(drawdown.min() * 100.0), 4),
        "total_buys": total_buys,
        "total_sells": total_sells,
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
        "two_holding_days": int(sum(1 for row in daily_rows if row["holdings_count"] >= args.slots)),
        "avg_holdings_count": round(float(pd.Series([row["holdings_count"] for row in daily_rows]).mean()), 4),
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "port": "formula_breakout_top2_backtest",
        "strategy_name": "公式选股Top2满仓轮动",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "universe_count": universe_count,
        "initial_cash": float(args.initial_cash),
        "lot_size": int(args.lot_size),
        "slots": int(args.slots),
        "assumption": "每天收盘先按卖出规则处理已有仓位；空出的仓位槽用当日公式评分从高到低补齐。两个空槽时现金均分，各槽尽量满额买入整手；已持有或当天刚卖出的股票不重复买回。",
        "fee_model": {
            "commission_rate": args.commission_rate,
            "min_commission": args.min_commission,
            "stamp_tax_rate_sell_only": args.stamp_tax_rate,
            "transfer_fee_rate_both_sides": args.transfer_fee_rate,
        },
        "summary": summary,
        "daily": daily_rows,
        "monthly": clean_records(monthly),
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
