#!/usr/bin/env python
"""Live manual-following plan for the formula breakout Top3 strategy."""

from __future__ import annotations

import argparse
import builtins
import json
import math
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from formula_breakout_cash_backtest import (
    bar_for_date,
    fee_breakdown,
    prepare_history,
    remember_take_profit_trigger,
    sell_decision,
)
from formula_breakout_pipeline import (
    clean_records,
    compact_date,
    run_once as run_formula_once,
)
from formula_breakout_top2_backtest import (
    SlotHolding,
    asof_close,
    is_limit_up_buy_blocked,
    max_affordable_shares,
)

print = partial(builtins.print, flush=True)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def round2(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return round(out, 2)


def normalize_code(value: object) -> str:
    return str(value).strip().zfill(6)[-6:]


def make_fee_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        commission_rate=float(args.commission_rate),
        min_commission=float(args.min_commission),
        stamp_tax_rate=float(args.stamp_tax_rate),
        transfer_fee_rate=float(args.transfer_fee_rate),
        lot_size=int(args.lot_size),
        block_limit_up_buys=True,
        limit_up_source=args.limit_up_source,
        min_seal_amount=float(args.min_seal_amount),
        limit_up_pct=float(args.limit_up_pct),
        limit_close_high_ratio=float(args.limit_close_high_ratio),
    )


def make_formula_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        run_once=True,
        replay=False,
        daemon=False,
        print_cron=False,
        archive_only=False,
        target_date=args.target_date,
        replay_start="",
        replay_end="",
        replay_min_coverage=500,
        schedule_hour=int(args.schedule_hour),
        schedule_minute=int(args.schedule_minute),
        universe_file=args.universe_file,
        history_dir=args.history_dir,
        run_dir=args.formula_run_dir,
        snapshot_dir=args.formula_snapshot_dir,
        feedback_file=args.formula_feedback_file,
        static_dir=args.static_dir,
        template=args.formula_template,
        refresh=bool(args.refresh),
        quote_source=str(args.quote_source),
        workers=int(args.workers),
        retry=int(args.retry),
        progress_every=int(args.progress_every),
        require_target_date=not bool(args.allow_stale_target),
        min_float_market_cap=float(args.min_float_market_cap),
        display_limit=int(args.display_limit),
        feedback_window=200,
        feedback_display_limit=200,
    )


def buy_rule_text(args: argparse.Namespace) -> str:
    min_cap = float(getattr(args, "min_float_market_cap", 0.0) or 0.0)
    if min_cap > 0:
        return f"按公式评分从高到低补齐空槽；候选按成交额/换手率估算流通市值需不低于{min_cap / 100000000:.0f}亿；涨停不可买或单槽资金不足一手时顺延下一名。"
    return "按公式评分从高到低补齐空槽；涨停不可买或单槽资金不足一手时顺延下一名。"


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def initial_state(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "initial_cash": round(float(args.initial_cash), 2),
        "cash": round(float(args.initial_cash), 2),
        "slots": int(args.slots),
        "next_lot_id": 1,
        "last_plan_key": "",
        "last_plan_operations": [],
        "holdings": [],
        "ledger": [],
    }


def load_state(args: argparse.Namespace) -> Dict[str, object]:
    path = Path(args.state_file)
    if args.reset_state or not path.exists():
        return initial_state(args)
    state = load_json(path)
    if not state:
        return initial_state(args)
    state.setdefault("cash", float(args.initial_cash))
    state.setdefault("initial_cash", float(args.initial_cash))
    state.setdefault("slots", int(args.slots))
    state.setdefault("next_lot_id", 1)
    state.setdefault("last_plan_key", "")
    state.setdefault("last_plan_operations", [])
    state.setdefault("holdings", [])
    state.setdefault("ledger", [])
    return state


def holding_from_dict(item: Dict[str, object]) -> SlotHolding:
    return SlotHolding(
        lot_id=int(item.get("lot_id") or 0),
        slot=int(item.get("slot") or 0),
        code=normalize_code(item.get("code")),
        name=str(item.get("name") or ""),
        shares=int(item.get("shares") or 0),
        entry_date=str(item.get("entry_date") or ""),
        entry_price=float(item.get("entry_price") or 0.0),
        entry_open=float(item.get("entry_open") or 0.0),
        entry_fee=float(item.get("entry_fee") or 0.0),
        entry_gross=float(item.get("entry_gross") or 0.0),
        entry_rank=int(item.get("entry_rank") or item.get("rank") or 0),
        entry_score=float(item.get("entry_score") or item.get("formula_score") or 0.0),
        take_profit_armed=bool(item.get("take_profit_armed") or False),
        take_profit_trigger_date=str(item.get("take_profit_trigger_date") or ""),
        take_profit_trigger_reason=str(item.get("take_profit_trigger_reason") or ""),
        take_profit_trigger_close=float(item.get("take_profit_trigger_close") or 0.0),
    )


def serialize_holding(holding: SlotHolding) -> Dict[str, object]:
    out = asdict(holding)
    for key in ["entry_price", "entry_open", "entry_fee", "entry_gross", "entry_score", "take_profit_trigger_close"]:
        out[key] = round(float(out[key]), 4 if key != "entry_fee" and key != "entry_gross" else 2)
    return out


def load_histories(codes: Iterable[str], history_dir: Path) -> Dict[str, pd.DataFrame]:
    histories: Dict[str, pd.DataFrame] = {}
    for code in sorted({normalize_code(c) for c in codes if c}):
        path = history_dir / f"{code}.csv"
        if not path.exists():
            continue
        hist = prepare_history(path)
        if hist is not None and not hist.empty:
            histories[code] = hist
    return histories


def refresh_summary_path(args: argparse.Namespace, run_time: datetime) -> Path:
    return Path(args.formula_run_dir) / f"{run_time:%Y%m%d_%H%M}_refresh" / "refresh_summary.json"


def is_refresh_failure(args: argparse.Namespace, run_time: datetime) -> Tuple[bool, Dict[str, object]]:
    if not args.refresh:
        return False, {}
    summary = load_json(refresh_summary_path(args, run_time))
    if not summary:
        return False, {}
    refresh_ok = int(summary.get("refresh_ok") or 0)
    target_count = int(summary.get("target_date_count") or 0)
    failed = refresh_ok <= 0 or target_count <= 0
    return failed, summary


def candidate_rows(analysis: Dict[str, object]) -> List[Dict[str, object]]:
    rows = analysis.get("selected_rows")
    if not isinstance(rows, list):
        return []
    cleaned = [row for row in rows if isinstance(row, dict)]
    return sorted(cleaned, key=lambda r: int(r.get("rank") or 999999))


def as_signal(row: Dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**row)


def market_value(
    holdings: Sequence[SlotHolding],
    histories: Dict[str, pd.DataFrame],
    date_text: str,
) -> Tuple[float, List[Dict[str, object]]]:
    total = 0.0
    rows: List[Dict[str, object]] = []
    for holding in holdings:
        hist = histories.get(holding.code)
        last_close = asof_close(hist, date_text) if hist is not None else None
        if last_close is None or not math.isfinite(float(last_close)):
            last_close = holding.entry_price
        gross = float(last_close) * holding.shares
        total += gross
        basis = holding.entry_gross + holding.entry_fee
        rows.append(
            {
                **serialize_holding(holding),
                "last_close": round(float(last_close), 4),
                "market_value": round(gross, 2),
                "unrealized_pnl": round(gross - basis, 2),
                "unrealized_ret_pct": round((gross / basis - 1.0) * 100.0, 4) if basis > 0 else 0.0,
            }
        )
    return total, rows


def build_no_trade_plan(
    args: argparse.Namespace,
    state: Dict[str, object],
    run_time: datetime,
    reason: str,
    refresh_summary: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    date_text = run_time.strftime("%Y-%m-%d")
    holdings = [holding_from_dict(item) for item in state.get("holdings", []) if isinstance(item, dict)]
    histories = load_histories([h.code for h in holdings], Path(args.history_dir))
    cash = float(state.get("cash") or float(args.initial_cash))
    market, holding_rows = market_value(holdings, histories, date_text)
    plan = {
        "refresh_failed": bool(refresh_summary),
        "refresh_summary": refresh_summary or {},
        "cash_start": round(cash, 2),
        "cash_after_plan": round(cash, 2),
        "market_value_before": round(market, 2),
        "market_value_after_plan": round(market, 2),
        "equity_before": round(cash + market, 2),
        "equity_after_plan": round(cash + market, 2),
        "planned_buy_count": 0,
        "planned_sell_count": 0,
        "planned_skip_count": 0,
        "holdings_count_after": len(holdings),
        "fees_estimated": 0.0,
        "operations": [{"action": "wait", "date": date_text, "reason": reason}],
        "holdings_before": holding_rows,
        "holdings_after": holding_rows,
    }
    return plan, state


def build_plan(
    args: argparse.Namespace,
    analysis: Dict[str, object],
    state: Dict[str, object],
    run_time: datetime,
    duplicate_applied: bool,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    date_text = run_time.strftime("%Y-%m-%d")
    selected = candidate_rows(analysis)
    fee_args = make_fee_args(args)
    holdings = [holding_from_dict(item) for item in state.get("holdings", []) if isinstance(item, dict)]
    history_codes = [h.code for h in holdings] + [normalize_code(r.get("code")) for r in selected]
    histories = load_histories(history_codes, Path(args.history_dir))
    cash_start = float(state.get("cash") or float(args.initial_cash))
    cash = cash_start
    operations: List[Dict[str, object]] = []
    sold_codes: set[str] = set()
    remaining: List[SlotHolding] = []
    total_fees = 0.0

    if duplicate_applied:
        operations = list(state.get("last_plan_operations") or [])
        market, holding_rows = market_value(holdings, histories, date_text)
        plan = {
            "duplicate_applied": True,
            "cash_start": round(cash_start, 2),
            "cash_after_plan": round(cash_start, 2),
            "market_value_after_plan": round(market, 2),
            "equity_after_plan": round(cash_start + market, 2),
            "operations": operations,
            "holdings_before": holding_rows,
            "holdings_after": holding_rows,
            "fees_estimated": 0.0,
            "note": "同一计划时间已应用到账本，本次仅重新发布上一次操作。若确需重算并覆盖，请使用 --force-apply。",
        }
        return plan, state

    for holding in holdings:
        if date_text <= holding.entry_date:
            remaining.append(holding)
            operations.append(
                {
                    "action": "hold",
                    "slot": holding.slot,
                    "code": holding.code,
                    "name": holding.name,
                    "shares": holding.shares,
                    "reason": "买入日T+1限制或未到卖出检查日",
                }
            )
            continue
        hist = histories.get(holding.code)
        row = bar_for_date(hist, date_text) if hist is not None else None
        if row is None:
            remaining.append(holding)
            operations.append(
                {
                    "action": "hold",
                    "slot": holding.slot,
                    "code": holding.code,
                    "name": holding.name,
                    "shares": holding.shares,
                    "reason": "未取得当日K线",
                }
            )
            continue
        decision = sell_decision(row, holding)  # type: ignore[arg-type]
        reasons = list(decision["reasons"])
        if not reasons:
            remember_take_profit_trigger(holding, date_text, decision)
            remaining.append(holding)
            hold_reason = "未触发止盈止损"
            operations.append(
                {
                    "action": "hold",
                    "slot": holding.slot,
                    "code": holding.code,
                    "name": holding.name,
                    "shares": holding.shares,
                    "price": round(float(row.get("收盘") or holding.entry_price), 4),
                    "ma5_close": round(float(decision["ma5_close"]), 4) if decision.get("ma5_close") is not None else None,
                    "reason": hold_reason,
                }
            )
            continue
        price = float(decision["price"])
        gross = price * holding.shares
        fees = fee_breakdown(gross, "sell", fee_args)
        proceeds = gross - fees["total_fee"]
        basis = holding.entry_gross + holding.entry_fee
        pnl = proceeds - basis
        cash += proceeds
        total_fees += fees["total_fee"]
        sold_codes.add(holding.code)
        operations.append(
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
                "pnl": round(pnl, 2),
                "ret_pct": round(pnl / basis * 100.0, 4) if basis > 0 else 0.0,
                "reason": "；".join(reasons),
                "execution_time": "15:00计划/尾盘手动确认",
                "trigger_price": round(float(decision["trigger_price"]), 4) if decision["trigger_price"] is not None else None,
            }
        )

    holdings = remaining
    empty_slots = [slot for slot in range(1, int(args.slots) + 1) if all(h.slot != slot for h in holdings)]
    bought_codes = {h.code for h in holdings}
    blocked_codes: set[str] = set()
    next_lot_id = int(state.get("next_lot_id") or 1)

    for slot in empty_slots:
        remaining_slots = int(args.slots) - len(holdings)
        if remaining_slots <= 0 or cash <= 0:
            break
        budget = cash / remaining_slots
        bought = False
        for row in selected:
            signal = as_signal(row)
            code = normalize_code(row.get("code"))
            if code in bought_codes or code in sold_codes or code in blocked_codes:
                continue
            blocked, meta = is_limit_up_buy_blocked(signal, fee_args)
            if blocked:
                blocked_codes.add(code)
                operations.append(
                    {
                        "action": "skip",
                        "date": date_text,
                        "slot": slot,
                        "code": code,
                        "name": str(row.get("name") or ""),
                        "rank": int(row.get("rank") or 0),
                        "price": round(float(row.get("close") or 0.0), 4),
                        "pct_chg": round(float(row.get("pct_chg") or 0.0), 4),
                        "reason": "尾盘涨停不可买入",
                        "block_meta": meta,
                    }
                )
                continue
            price = float(row.get("close") or 0.0)
            shares, cost, fees = max_affordable_shares(price, budget, fee_args)
            if shares <= 0:
                operations.append(
                    {
                        "action": "skip",
                        "date": date_text,
                        "slot": slot,
                        "code": code,
                        "name": str(row.get("name") or ""),
                        "rank": int(row.get("rank") or 0),
                        "price": round(price, 4),
                        "slot_budget": round(budget, 2),
                        "reason": "单槽资金不足买入一手",
                    }
                )
                continue
            gross = price * shares
            holding = SlotHolding(
                lot_id=next_lot_id,
                slot=slot,
                code=code,
                name=str(row.get("name") or ""),
                shares=int(shares),
                entry_date=date_text,
                entry_price=price,
                entry_open=float(row.get("open") or price),
                entry_fee=float(fees["total_fee"]),
                entry_gross=float(gross),
                entry_rank=int(row.get("rank") or 0),
                entry_score=float(row.get("formula_score") or 0.0),
            )
            next_lot_id += 1
            cash -= cost
            total_fees += fees["total_fee"]
            holdings.append(holding)
            bought_codes.add(code)
            bought = True
            operations.append(
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
                    "execution_time": "15:00计划/尾盘手动确认",
                }
            )
            break
        if not bought and selected:
            operations.append({"action": "wait", "date": date_text, "slot": slot, "reason": "没有可买入候选"})

    market, holdings_after = market_value(holdings, histories, date_text)
    market_before, holdings_before = market_value([holding_from_dict(item) for item in state.get("holdings", []) if isinstance(item, dict)], histories, date_text)
    state_after = {
        **state,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "cash": round(cash, 2),
        "slots": int(args.slots),
        "next_lot_id": int(next_lot_id),
        "holdings": [serialize_holding(h) for h in holdings],
    }
    ledger = list(state_after.get("ledger") or [])
    ledger.append(
        {
            "plan_key": f"{date_text}_{int(args.schedule_hour):02d}{int(args.schedule_minute):02d}",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "cash_start": round(cash_start, 2),
            "cash_after_plan": round(cash, 2),
            "equity_after_plan": round(cash + market, 2),
            "operations": operations,
        }
    )
    state_after["ledger"] = ledger[-240:]
    state_after["last_plan_operations"] = operations

    buy_count = sum(1 for item in operations if item.get("action") == "buy")
    sell_count = sum(1 for item in operations if item.get("action") == "sell")
    skip_count = sum(1 for item in operations if item.get("action") == "skip")
    plan = {
        "duplicate_applied": False,
        "cash_start": round(cash_start, 2),
        "cash_after_plan": round(cash, 2),
        "market_value_before": round(market_before, 2),
        "market_value_after_plan": round(market, 2),
        "equity_before": round(cash_start + market_before, 2),
        "equity_after_plan": round(cash + market, 2),
        "planned_buy_count": buy_count,
        "planned_sell_count": sell_count,
        "planned_skip_count": skip_count,
        "holdings_count_after": len(holdings),
        "fees_estimated": round(total_fees, 2),
        "operations": operations,
        "holdings_before": holdings_before,
        "holdings_after": holdings_after,
    }
    return plan, state_after


def build_report(
    args: argparse.Namespace,
    run_time: datetime,
    analysis: Dict[str, object],
    state_before: Dict[str, object],
    state_after: Dict[str, object],
    plan: Dict[str, object],
    applied: bool,
) -> Dict[str, object]:
    selected = candidate_rows(analysis)
    date_key = compact_date(run_time.strftime("%Y-%m-%d"))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_date": run_time.strftime("%Y-%m-%d"),
        "snapshot_time": run_time.isoformat(timespec="seconds"),
        "snapshot_key": date_key,
        "port": "live_formula_trading",
        "strategy_name": "公式选股Top3实盘跟单",
        "mode": "manual_following",
        "applied_to_theoretical_state": bool(applied),
        "capital": round(float(args.initial_cash), 2),
        "slots": int(args.slots),
        "slot_target_cash": round(float(args.initial_cash) / int(args.slots), 2),
        "formula_snapshot_date": analysis.get("snapshot_date"),
        "formula_snapshot_time": analysis.get("snapshot_time"),
        "formula_selected_count": int(analysis.get("selected_count") or len(selected)),
        "data_status": "refresh_failed" if plan.get("refresh_failed") else "ok",
        "refresh_summary": plan.get("refresh_summary", {}),
        "summary": {
            "cash_start": plan.get("cash_start"),
            "cash_after_plan": plan.get("cash_after_plan"),
            "equity_before": plan.get("equity_before"),
            "equity_after_plan": plan.get("equity_after_plan"),
            "market_value_after_plan": plan.get("market_value_after_plan"),
            "planned_buy_count": plan.get("planned_buy_count", 0),
            "planned_sell_count": plan.get("planned_sell_count", 0),
            "planned_skip_count": plan.get("planned_skip_count", 0),
            "holdings_count_after": plan.get("holdings_count_after", 0),
            "fees_estimated": plan.get("fees_estimated", 0.0),
        },
        "operations": plan.get("operations", []),
        "holdings_before": plan.get("holdings_before", []),
        "holdings_after": plan.get("holdings_after", []),
        "candidates": selected[: int(args.display_limit)],
        "state_before": {
            "cash": round2(state_before.get("cash")),
            "holdings_count": len(state_before.get("holdings", []) or []),
            "last_plan_key": state_before.get("last_plan_key", ""),
        },
        "state_after": {
            "cash": round2(state_after.get("cash")),
            "holdings_count": len(state_after.get("holdings", []) or []),
            "last_plan_key": state_after.get("last_plan_key", ""),
        },
        "rules": {
            "schedule": "交易日15:00生成当日公式Top3跟单计划。",
            "position": "本金15000，三等分仓位；每个空槽使用剩余现金按剩余槽位均分后买入整手。",
            "buy": buy_rule_text(args),
            "sell": "T+1；买入日当日不卖出；后续交易日优先判断止损，收盘价低于买入价则按收盘价止损；未止损时，放量阴线、阴十字星、连续两阴任一成立即按收盘价止盈卖出。",
            "fees": {
                "commission_rate": args.commission_rate,
                "min_commission": args.min_commission,
                "stamp_tax_rate_sell_only": args.stamp_tax_rate,
                "transfer_fee_rate_both_sides": args.transfer_fee_rate,
            },
        },
        "notes": [
            "页面只生成跟单计划，不连接券商账号，不自动下单。",
            "15:00使用当时可取得的最新日K/行情近似，最终收盘数据可能与盘中快照不同。",
            "理论账本假设你按计划成交；实际成交价不同会造成偏差。",
        ],
    }


def write_static_page(args: argparse.Namespace) -> None:
    template = Path(args.template)
    if not template.exists():
        return
    static_dir = Path(args.static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "live-trading.html").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, object]:
    if args.target_date:
        run_time = datetime.strptime(args.target_date, "%Y-%m-%d").replace(hour=args.schedule_hour, minute=args.schedule_minute)
    else:
        run_time = datetime.now().replace(second=0, microsecond=0)
    plan_key = f"{run_time:%Y-%m-%d}_{int(args.schedule_hour):02d}{int(args.schedule_minute):02d}"
    state = load_state(args)
    duplicate_applied = bool(args.apply_state and not args.force_apply and state.get("last_plan_key") == plan_key)

    formula_args = make_formula_args(args)
    previous_formula_latest = load_json(Path(args.static_dir) / "reports" / "formula_breakout.json")
    analysis = run_formula_once(formula_args, run_time)
    refresh_failed, refresh_summary = is_refresh_failure(args, run_time)
    if refresh_failed:
        if previous_formula_latest:
            write_json(Path(args.static_dir) / "reports" / "formula_breakout.json", previous_formula_latest)
        reason = "行情刷新失败，未生成当日买卖指令"
        plan, state_after = build_no_trade_plan(args, state, run_time, reason, refresh_summary)
    else:
        plan, state_after = build_plan(args, analysis, state, run_time, duplicate_applied)

    applied = bool(args.apply_state and not duplicate_applied and not refresh_failed)
    if applied:
        state_after["last_plan_key"] = plan_key
        write_json(Path(args.state_file), state_after)
    elif duplicate_applied:
        state_after = state

    report = build_report(args, run_time, analysis, state, state_after, plan, applied)
    write_json(Path(args.output), report)
    latest = Path(args.live_dir) / "latest_live_plan.json"
    write_json(latest, report)
    write_static_page(args)
    print(
        f"[{now_text()}] live plan date={report['snapshot_date']} selected={report['formula_selected_count']} "
        f"buys={report['summary']['planned_buy_count']} sells={report['summary']['planned_sell_count']} "
        f"applied={applied} output={args.output}"
    )
    return report


def print_cron(args: argparse.Namespace) -> None:
    python = args.python
    root = Path.cwd()
    print(
        f"0 15 * * 1-5 cd {root} && {python} live_formula_trading.py --run-once --refresh --apply-state "
        f"--target-date $(date +\\%F) >> data_cache/live_trading/cron.log 2>&1"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--print-cron", action="store_true")
    parser.add_argument("--target-date", default="")
    parser.add_argument("--schedule-hour", type=int, default=15)
    parser.add_argument("--schedule-minute", type=int, default=0)
    parser.add_argument("--initial-cash", type=float, default=15000.0)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--limit-up-source", choices=["daily"], default="daily")
    parser.add_argument("--min-seal-amount", type=float, default=1.0)
    parser.add_argument("--limit-up-pct", type=float, default=9.8)
    parser.add_argument("--limit-close-high-ratio", type=float, default=0.999)
    parser.add_argument("--min-float-market-cap", type=float, default=0.0)
    parser.add_argument("--apply-state", action="store_true")
    parser.add_argument("--force-apply", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--quote-source", choices=["tencent", "pytdx", "auto"], default="auto")
    parser.add_argument("--allow-stale-target", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=600)
    parser.add_argument("--display-limit", type=int, default=60)
    parser.add_argument("--universe-file", default="data_cache/volume_contraction_screen_20260701_mainboard_entry_close/refresh_status.csv")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--formula-run-dir", default="data_cache/formula_breakout_runs")
    parser.add_argument("--formula-snapshot-dir", default="data_cache/formula_breakout_snapshots")
    parser.add_argument("--formula-feedback-file", default="data_cache/formula_breakout_snapshots/feedback.csv")
    parser.add_argument("--static-dir", default="static")
    parser.add_argument("--formula-template", default="templates/formula_breakout.html")
    parser.add_argument("--template", default="templates/live_formula_trading.html")
    parser.add_argument("--live-dir", default="data_cache/live_trading")
    parser.add_argument("--state-file", default="data_cache/live_trading/state.json")
    parser.add_argument("--output", default="static/reports/live_formula_trading.json")
    parser.add_argument("--python", default="/home/luochangyu/anaconda3/envs/py310/bin/python")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_cron:
        print_cron(args)
    elif args.run_once:
        run(args)
    else:
        raise SystemExit("choose --run-once or --print-cron")


if __name__ == "__main__":
    main()
