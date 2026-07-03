#!/usr/bin/env python
"""Cash-based rolling backtest for the formula breakout selector."""

from __future__ import annotations

import argparse
import builtins
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from formula_breakout_backtest import evaluate_history, rank_signals, resolve_dates, round_or_none
from formula_breakout_pipeline import clean_records, load_universe, normalize_code, now_text
from main_uptrend_model import normalize_hist

print = partial(builtins.print, flush=True)


@dataclass
class Holding:
    lot_id: int
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


def prepare_history(path: Path) -> Optional[pd.DataFrame]:
    try:
        hist = normalize_hist(pd.read_csv(path))
    except Exception:
        return None
    if hist.empty:
        return None
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["日期"], errors="coerce").dt.normalize()
    hist = hist.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    if hist.empty:
        return None
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅"]:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist["date_text"] = hist["date"].dt.strftime("%Y-%m-%d")
    hist["prev_open"] = hist["开盘"].shift(1)
    hist["prev_close"] = hist["收盘"].shift(1)
    hist["prev_volume"] = hist["成交量"].shift(1)
    return hist.reset_index(drop=True)


def is_doji(row: pd.Series) -> bool:
    high = float(row.get("最高") or 0.0)
    low = float(row.get("最低") or 0.0)
    open_ = float(row.get("开盘") or 0.0)
    close = float(row.get("收盘") or 0.0)
    span = high - low
    if span <= 0:
        return False
    return abs(close - open_) <= span * 0.10


def is_volume_bearish(row: pd.Series) -> bool:
    open_ = float(row.get("开盘") or 0.0)
    close = float(row.get("收盘") or 0.0)
    volume = float(row.get("成交量") or 0.0)
    prev_volume = float(row.get("prev_volume") or 0.0)
    return close < open_ and prev_volume > 0 and volume > prev_volume


def is_long_upper_bearish(row: pd.Series) -> bool:
    open_ = float(row.get("开盘") or 0.0)
    close = float(row.get("收盘") or 0.0)
    high = float(row.get("最高") or 0.0)
    low = float(row.get("最低") or 0.0)
    if close >= open_:
        return False
    body = abs(close - open_)
    if body <= 0:
        return False
    upper = high - max(open_, close)
    lower = min(open_, close) - low
    return upper >= body * 1.5 and upper >= lower


def is_bearish_engulfing(row: pd.Series) -> bool:
    open_ = float(row.get("开盘") or 0.0)
    close = float(row.get("收盘") or 0.0)
    prev_open = float(row.get("prev_open") or 0.0)
    prev_close = float(row.get("prev_close") or 0.0)
    return close < open_ and prev_close > prev_open and open_ >= prev_close and close <= prev_open


def sell_reasons(row: pd.Series, holding: Holding) -> List[str]:
    close = float(row.get("收盘") or 0.0)
    reasons: List[str] = []
    if close < holding.entry_open:
        reasons.append("收盘跌破买入阳线开盘价")
    if is_doji(row):
        reasons.append("十字星")
    if is_volume_bearish(row):
        reasons.append("放量阴线")
    if is_long_upper_bearish(row):
        reasons.append("上长阴线")
    if is_bearish_engulfing(row):
        reasons.append("阴包阳实体吞没")
    return reasons


def fee_breakdown(amount: float, side: str, args: argparse.Namespace) -> Dict[str, float]:
    commission = 0.0
    if amount > 0 and args.commission_rate > 0:
        commission = max(amount * args.commission_rate, args.min_commission)
    stamp_tax = amount * args.stamp_tax_rate if side == "sell" else 0.0
    transfer_fee = amount * args.transfer_fee_rate
    total = commission + stamp_tax + transfer_fee
    return {
        "commission": commission,
        "stamp_tax": stamp_tax,
        "transfer_fee": transfer_fee,
        "total_fee": total,
    }


def bar_for_date(history: pd.DataFrame, date_text: str) -> Optional[pd.Series]:
    matched = history[history["date_text"] == date_text]
    if matched.empty:
        return None
    return matched.iloc[0]


def asof_close(history: pd.DataFrame, date_text: str) -> Optional[float]:
    date = pd.Timestamp(date_text)
    dates = history["date"].to_numpy(dtype="datetime64[ns]")
    idx = int(np.searchsorted(dates, np.datetime64(date), side="right") - 1)
    if idx < 0:
        return None
    close = float(history.iloc[idx]["收盘"])
    return close if math.isfinite(close) else None


def build_signals_and_histories(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str], pd.Timestamp, pd.Timestamp, int]:
    universe = load_universe(Path(args.universe_file))
    start, end = resolve_dates(args, universe)
    histories: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, object]] = []
    market_dates: set[str] = set()
    history_dir = Path(args.history_dir)
    for idx, item in enumerate(universe.itertuples(index=False), start=1):
        path = history_dir / f"{item.code}.csv"
        if not path.exists():
            continue
        hist = prepare_history(path)
        if hist is None:
            continue
        histories[item.code] = hist
        stock_rows, stock_dates = evaluate_history(item.code, item.name, hist.drop(columns=["date", "date_text"], errors="ignore"), start, end)
        rows.extend(stock_rows)
        market_dates.update(stock_dates)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"[{now_text()}] cash backtest signals {idx}/{len(universe)}, signals={len(rows)}")
    signals = rank_signals(pd.DataFrame(rows))
    return signals, histories, sorted(market_dates), start, end, len(universe)


def summarize_operations(items: Sequence[Dict[str, object]], max_items: int = 8) -> str:
    if not items:
        return ""
    labels: List[str] = []
    for item in items[:max_items]:
        action = item.get("action")
        code = item.get("code")
        name = item.get("name")
        price = item.get("price")
        if action == "buy":
            labels.append(f"买{code}{name}@{price}")
        elif action == "sell":
            labels.append(f"卖{code}{name}@{price}({item.get('reason')})")
    if len(items) > max_items:
        labels.append(f"...另{len(items) - max_items}笔")
    return "；".join(labels)


def simulate(args: argparse.Namespace) -> Dict[str, object]:
    signals, histories, market_dates, start, end, universe_count = build_signals_and_histories(args)
    if not market_dates:
        raise ValueError("no trading dates available for cash backtest")
    signals_by_date = {date: df.sort_values(["rank"]) for date, df in signals.groupby("snapshot_date")} if not signals.empty else {}

    cash = float(args.initial_cash)
    lot_size = int(args.lot_size)
    holdings: List[Holding] = []
    lot_id = 0
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

    prev_equity = cash
    for date_text in market_dates:
        cash_start = cash
        day_ops: List[Dict[str, object]] = []
        day_fees = 0.0

        remaining: List[Holding] = []
        for holding in holdings:
            hist = histories.get(holding.code)
            row = bar_for_date(hist, date_text) if hist is not None else None
            if row is None:
                remaining.append(holding)
                continue
            reasons = sell_reasons(row, holding)
            if not reasons:
                remaining.append(holding)
                continue
            price = float(row["收盘"])
            gross = price * holding.shares
            fees = fee_breakdown(gross, "sell", args)
            proceeds = gross - fees["total_fee"]
            pnl = proceeds - (holding.entry_gross + holding.entry_fee)
            cash += proceeds
            realized_pnl += pnl
            total_fees += fees["total_fee"]
            total_commission += fees["commission"]
            total_stamp_tax += fees["stamp_tax"]
            total_transfer_fee += fees["transfer_fee"]
            day_fees += fees["total_fee"]
            total_sells += 1
            ret_pct = pnl / (holding.entry_gross + holding.entry_fee) * 100.0 if holding.entry_gross + holding.entry_fee > 0 else 0.0
            closed_returns.append(ret_pct)
            day_ops.append(
                {
                    "action": "sell",
                    "date": date_text,
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

        skipped = 0
        first_skipped_rank: Optional[int] = None
        selected_count = int(len(signals_by_date.get(date_text, []))) if date_text in signals_by_date else 0
        if date_text in signals_by_date:
            for signal in signals_by_date[date_text].itertuples(index=False):
                price = float(signal.close)
                if not math.isfinite(price) or price <= 0:
                    skipped += 1
                    first_skipped_rank = first_skipped_rank or int(signal.rank)
                    continue
                gross = price * lot_size
                fees = fee_breakdown(gross, "buy", args)
                cost = gross + fees["total_fee"]
                if cash + 1e-9 < cost:
                    skipped += 1
                    first_skipped_rank = first_skipped_rank or int(signal.rank)
                    continue
                lot_id += 1
                cash -= cost
                total_fees += fees["total_fee"]
                total_commission += fees["commission"]
                total_stamp_tax += fees["stamp_tax"]
                total_transfer_fee += fees["transfer_fee"]
                day_fees += fees["total_fee"]
                holding = Holding(
                    lot_id=lot_id,
                    code=normalize_code(signal.code),
                    name=str(signal.name),
                    shares=lot_size,
                    entry_date=date_text,
                    entry_price=price,
                    entry_open=float(signal.open),
                    entry_fee=fees["total_fee"],
                    entry_gross=gross,
                    entry_rank=int(signal.rank),
                    entry_score=float(signal.formula_score),
                )
                holdings.append(holding)
                total_buys += 1
                day_ops.append(
                    {
                        "action": "buy",
                        "date": date_text,
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
                    }
                )

        market_value = 0.0
        unrealized_pnl = 0.0
        for holding in holdings:
            hist = histories.get(holding.code)
            close = asof_close(hist, date_text) if hist is not None else None
            if close is None:
                close = holding.entry_price
            market_value += close * holding.shares
            unrealized_pnl += close * holding.shares - (holding.entry_gross + holding.entry_fee)
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
    open_holdings = []
    liquidation_value = cash
    liquidation_fees = 0.0
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

    monthly = pd.DataFrame(daily_rows)
    monthly["month"] = monthly["date"].str.slice(0, 7)
    month_rows = monthly.groupby("month", as_index=False).agg(
        trading_days=("date", "count"),
        buy_count=("buy_count", "sum"),
        sell_count=("sell_count", "sum"),
        selected_count=("selected_count", "sum"),
        ending_equity=("equity", "last"),
        avg_daily_ret_pct=("daily_ret_pct", "mean"),
    )
    month_rows["month_ret_pct"] = monthly.groupby("month")["daily_ret_pct"].apply(lambda x: ((1.0 + x / 100.0).prod() - 1.0) * 100.0).values
    for col in ["ending_equity", "avg_daily_ret_pct", "month_ret_pct"]:
        month_rows[col] = pd.to_numeric(month_rows[col], errors="coerce").round(4)

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
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "port": "formula_breakout_cash_backtest",
        "strategy_name": "公式选股五万本金滚动操作",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "universe_count": universe_count,
        "initial_cash": float(args.initial_cash),
        "lot_size": lot_size,
        "assumption": "每天收盘先按当日K线检查已有仓位卖出，再按当日公式评分排名以收盘价依次买入1手；已计入佣金、印花税、过户费；不计滑点和涨跌停无法成交影响。",
        "fee_model": {
            "commission_rate": args.commission_rate,
            "min_commission": args.min_commission,
            "stamp_tax_rate_sell_only": args.stamp_tax_rate,
            "transfer_fee_rate_both_sides": args.transfer_fee_rate,
        },
        "sell_rules": {
            "stop_loss": "后续交易日收盘价严格低于买入信号阳线开盘价时，按收盘价卖出。",
            "volume_bearish": "放量阴线：C<O 且 V>REF(V,1)。",
            "doji": "十字星：实体不超过当日高低振幅的10%，不区分阴阳。",
            "long_upper_bearish": "上长阴线：阴线且上影线至少为实体1.5倍，并不短于下影线。",
            "bearish_engulfing": "阴包阳实体吞没：当日阴线实体完全吞没前一交易日阳线实体。",
        },
        "summary": summary,
        "daily": daily_rows,
        "monthly": clean_records(month_rows),
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
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--universe-file", default="data_cache/volume_contraction_screen_20260701_mainboard_entry_close/refresh_status.csv")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--output", default="static/reports/formula_breakout_cash_backtest_1y.json")
    parser.add_argument("--daily-csv", default="data_cache/formula_breakout_backtests/formula_breakout_cash_backtest_1y_daily.csv")
    parser.add_argument("--operations-csv", default="data_cache/formula_breakout_backtests/formula_breakout_cash_backtest_1y_operations.csv")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    simulate(parse_args())


if __name__ == "__main__":
    main()
