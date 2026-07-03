#!/usr/bin/env python
"""Batch Top1-Top5 full-position formula backtests with limit-up buy blocking."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from formula_breakout_cash_backtest import build_signals_and_histories
from formula_breakout_pipeline import now_text
from formula_breakout_top2_backtest import load_limit_up_pools, simulate


def build_comparison(reports: List[Dict[str, object]]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for report in reports:
        summary = report["summary"]  # type: ignore[index]
        row = {
            "slots": int(report["slots"]),  # type: ignore[index]
            "label": f"Top{int(report['slots'])}",
            "final_equity": summary["final_equity"],  # type: ignore[index]
            "final_liquidation_equity": summary["final_liquidation_equity"],  # type: ignore[index]
            "profit": summary["profit"],  # type: ignore[index]
            "liquidation_profit": summary["liquidation_profit"],  # type: ignore[index]
            "return_pct": summary["return_pct"],  # type: ignore[index]
            "liquidation_return_pct": summary["liquidation_return_pct"],  # type: ignore[index]
            "max_drawdown_pct": summary["max_drawdown_pct"],  # type: ignore[index]
            "total_fees": summary["total_fees"],  # type: ignore[index]
            "total_commission": summary["total_commission"],  # type: ignore[index]
            "total_stamp_tax": summary["total_stamp_tax"],  # type: ignore[index]
            "total_transfer_fee": summary["total_transfer_fee"],  # type: ignore[index]
            "estimated_open_sell_fees": summary["estimated_open_sell_fees"],  # type: ignore[index]
            "total_costs_with_estimated_exit": summary["total_costs_with_estimated_exit"],  # type: ignore[index]
            "total_buys": summary["total_buys"],  # type: ignore[index]
            "total_sells": summary["total_sells"],  # type: ignore[index]
            "blocked_limit_up_buys": summary["blocked_limit_up_buys"],  # type: ignore[index]
            "limit_up_pool_requested_days": summary.get("limit_up_pool_requested_days"),  # type: ignore[attr-defined]
            "limit_up_pool_nonempty_days": summary.get("limit_up_pool_nonempty_days"),  # type: ignore[attr-defined]
            "limit_up_pool_skipped_no_source_days": summary.get("limit_up_pool_skipped_no_source_days"),  # type: ignore[attr-defined]
            "open_lots": summary["open_lots"],  # type: ignore[index]
            "full_slot_days": summary["full_slot_days"],  # type: ignore[index]
            "avg_holdings_count": summary["avg_holdings_count"],  # type: ignore[index]
        }
        rows.append(row)

    best_profit = max(rows, key=lambda r: float(r["profit"])) if rows else None
    best_liquidation = max(rows, key=lambda r: float(r["liquidation_profit"])) if rows else None
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "port": "formula_breakout_top1_to_top5_limit_backtest",
        "strategy_name": "公式选股Top1-Top5满仓轮动（封单过滤）",
        "start_date": reports[0]["start_date"] if reports else None,
        "end_date": reports[0]["end_date"] if reports else None,
        "initial_cash": reports[0]["initial_cash"] if reports else None,
        "assumption": reports[0]["assumption"] if reports else None,
        "buy_block_rule": reports[0]["buy_block_rule"] if reports else None,
        "fee_model": reports[0]["fee_model"] if reports else None,
        "best_profit_slots": best_profit["slots"] if best_profit else None,
        "best_liquidation_slots": best_liquidation["slots"] if best_liquidation else None,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--initial-cash", type=float, default=50000.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--min-slots", type=int, default=1)
    parser.add_argument("--max-slots", type=int, default=5)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--universe-file", default="data_cache/volume_contraction_screen_20260701_mainboard_entry_close/refresh_status.csv")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--limit-up-pool-dir", default="data_cache/formula_breakout_backtests/zt_pool_em")
    parser.add_argument("--limit-up-data-start-date", default="")
    parser.add_argument("--refresh-limit-up-pool", action="store_true")
    parser.add_argument("--min-seal-amount", type=float, default=1.0)
    parser.add_argument("--limit-up-pct", type=float, default=9.8)
    parser.add_argument("--limit-close-high-ratio", type=float, default=0.999)
    parser.add_argument("--progress-every", type=int, default=600)
    parser.add_argument("--output", default="static/reports/formula_breakout_top1_to_top5_limit_backtest_1y.json")
    parser.add_argument("--detail-output-template", default="static/reports/formula_breakout_top{slots}_limit_backtest_1y.json")
    parser.add_argument("--daily-csv-template", default="data_cache/formula_breakout_backtests/formula_breakout_top{slots}_limit_backtest_1y_daily.csv")
    parser.add_argument("--operations-csv-template", default="data_cache/formula_breakout_backtests/formula_breakout_top{slots}_limit_backtest_1y_operations.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = argparse.Namespace(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
        lot_size=args.lot_size,
        slots=args.min_slots,
        commission_rate=args.commission_rate,
        min_commission=args.min_commission,
        stamp_tax_rate=args.stamp_tax_rate,
        transfer_fee_rate=args.transfer_fee_rate,
        universe_file=args.universe_file,
        history_dir=args.history_dir,
        block_limit_up_buys=True,
        limit_up_source="zt_pool",
        limit_up_pool_dir=args.limit_up_pool_dir,
        limit_up_data_start_date=args.limit_up_data_start_date,
        refresh_limit_up_pool=args.refresh_limit_up_pool,
        min_seal_amount=args.min_seal_amount,
        limit_up_pct=args.limit_up_pct,
        limit_close_high_ratio=args.limit_close_high_ratio,
        output="",
        daily_csv="",
        operations_csv="",
        progress_every=args.progress_every,
    )
    prebuilt = build_signals_and_histories(base)
    print(f"[{now_text()}] built shared signals rows={len(prebuilt[0])}, dates={len(prebuilt[2])}")
    limit_up_pools = load_limit_up_pools(prebuilt[2], base)
    print(f"[{now_text()}] loaded shared zt pools dates={len(limit_up_pools)}")

    reports: List[Dict[str, object]] = []
    for slots in range(int(args.min_slots), int(args.max_slots) + 1):
        run_args = copy.copy(base)
        run_args.slots = slots
        run_args.output = args.detail_output_template.format(slots=slots)
        run_args.daily_csv = args.daily_csv_template.format(slots=slots)
        run_args.operations_csv = args.operations_csv_template.format(slots=slots)
        run_args._prebuilt_data = prebuilt
        run_args._limit_up_pools = limit_up_pools
        reports.append(simulate(run_args))

    comparison = build_comparison(reports)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{now_text()}] wrote {output}")


if __name__ == "__main__":
    main()
