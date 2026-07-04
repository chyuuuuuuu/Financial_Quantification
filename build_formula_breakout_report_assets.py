#!/usr/bin/env python
"""Build static report JSON assets for the formula breakout Top3 page."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-report", default="data_cache/formula_breakout_backtests/formula_breakout_top3_limit_backtest_10y_full.json")
    parser.add_argument("--summary-output", default="static/reports/formula_breakout_top3_limit_backtest_10y.json")
    parser.add_argument("--trades-output", default="static/reports/formula_breakout_top3_10y_trades.json")
    parser.add_argument("--minute-dir", default="data_cache/minute_bars")
    parser.add_argument("--publish-revision", default="")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def r2(value: Any) -> Optional[float]:
    return None if value is None else round(float(value), 2)


def r4(value: Any) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def parse_date(value: Optional[str]) -> Optional[datetime]:
    return datetime.strptime(value, "%Y-%m-%d") if value else None


def build_limitations(report: Dict[str, Any], coverage: Dict[str, Any]) -> List[str]:
    universe = coverage.get("universe_count", report.get("universe_count"))
    covered = coverage.get("history_covered_to_2016_count", "-")
    return [
        "止损规则按用户最新指定执行：T+1交易，买入日不卖出；后续交易日收盘价B小于等于买入日开盘价A，即视为收盘触发。",
        "止损成交价为B，即触发日收盘价；该假设不处理真实收盘集合竞价排队和滑点问题。",
        "止盈条件：十字星（不区分阴阳）或放量阴线，均按触发日收盘价卖出。",
        "买入时刻按信号日收盘集合竞价/15:00的收盘价近似。",
        "量能口径：选股三倍阳量为V>3*REF(V,1)且C>O；缩量为MA5/MA10成交量低于三倍阳当日量。",
        "卖出里的放量阴线为C<O且V>REF(V,1)，仅要求今日成交量大于昨日成交量，不要求大于均量或倍量阈值。",
        "当前止损以日线收盘价判断，不使用分钟线触发时间。",
        "封单数据使用东方财富涨停股池；本次十年区间内可用非空封单池仅覆盖到2026-07-03，其他日期按无法确认封单处理，不触发封单买入过滤。",
        f"历史日线覆盖不完整：{universe}只公式股票中{covered}只覆盖到2016年，其他股票从本地可用首日开始计算。",
        "回测计入佣金、印花税、过户费；不计滑点和真实排队成交。",
    ]


def build_summary_payload(report: Dict[str, Any], previous_summary: Dict[str, Any], publish_revision: str) -> Dict[str, Any]:
    coverage = dict(previous_summary.get("coverage") or {})
    coverage["universe_count"] = report.get("universe_count", coverage.get("universe_count"))
    payload = {
        key: report[key]
        for key in [
            "generated_at",
            "port",
            "strategy_name",
            "start_date",
            "end_date",
            "initial_cash",
            "lot_size",
            "slots",
            "assumption",
            "buy_block_rule",
            "fee_model",
            "sell_rules",
        ]
        if key in report
    }
    payload.update(
        {
            "coverage": coverage,
            "limitations": build_limitations(report, coverage),
            "summary": report["summary"],
            "yearly": report.get("yearly", []),
            "monthly": report.get("monthly", []),
            "daily": report.get("daily", []),
            "open_holdings": report.get("open_holdings", []),
            "signals_total": report.get("signals_total"),
            "publish_revision": publish_revision,
        }
    )
    return payload


def operation_items(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for day in report.get("operations", []):
        for item in day.get("items", []):
            yield item


def build_trade_map(report: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    trades: Dict[int, Dict[str, Any]] = {}
    for op in operation_items(report):
        action = op.get("action")
        lot_id = op.get("lot_id")
        if lot_id is None:
            continue
        lot = int(lot_id)
        if action == "buy":
            trades[lot] = {
                "lot_id": lot,
                "slot": op.get("slot"),
                "code": op.get("code"),
                "name": op.get("name"),
                "buy_date": op.get("date"),
                "buy_execution_time": op.get("execution_time"),
                "sell_date": None,
                "sell_execution_time": None,
                "status": "open",
                "holding_days": None,
                "shares": op.get("shares"),
                "buy_price": r4(op.get("price")),
                "sell_price": None,
                "trigger_price": None,
                "buy_gross": r2(op.get("gross_amount")),
                "sell_gross": None,
                "buy_fee": r2(op.get("fee")),
                "sell_fee": 0.0,
                "total_fee": r2(op.get("fee") or 0),
                "pnl": None,
                "ret_pct": None,
                "buy_rank": op.get("rank"),
                "formula_score": op.get("formula_score"),
                "entry_open": r4(op.get("entry_open")),
                "sell_reason": "持仓中",
                "last_close": None,
                "market_value": None,
                "unrealized_pnl": None,
                "unrealized_ret_pct": None,
            }
        elif action == "sell":
            trade = trades.setdefault(lot, {"lot_id": lot})
            buy_date = trade.get("buy_date") or op.get("entry_date")
            sell_date = op.get("date")
            buy_dt = parse_date(buy_date)
            sell_dt = parse_date(sell_date)
            buy_fee = float(trade.get("buy_fee") or op.get("entry_fee") or 0)
            sell_fee = float(op.get("fee") or 0)
            trade.update(
                {
                    "slot": trade.get("slot") or op.get("slot"),
                    "code": trade.get("code") or op.get("code"),
                    "name": trade.get("name") or op.get("name"),
                    "buy_date": buy_date,
                    "sell_date": sell_date,
                    "sell_execution_time": op.get("execution_time"),
                    "status": "closed",
                    "holding_days": (sell_dt - buy_dt).days if buy_dt and sell_dt else None,
                    "shares": trade.get("shares") or op.get("shares"),
                    "buy_price": trade.get("buy_price") if trade.get("buy_price") is not None else r4(op.get("entry_price")),
                    "sell_price": r4(op.get("price")),
                    "trigger_price": r4(op.get("trigger_price")),
                    "buy_gross": trade.get("buy_gross"),
                    "sell_gross": r2(op.get("gross_amount")),
                    "buy_fee": r2(buy_fee),
                    "sell_fee": r2(sell_fee),
                    "total_fee": r2(buy_fee + sell_fee),
                    "pnl": r2(op.get("pnl")),
                    "ret_pct": r4(op.get("ret_pct")),
                    "entry_open": trade.get("entry_open") if trade.get("entry_open") is not None else r4(op.get("entry_open")),
                    "sell_reason": op.get("reason"),
                    "last_close": None,
                    "market_value": None,
                    "unrealized_pnl": None,
                    "unrealized_ret_pct": None,
                }
            )

    for holding in report.get("open_holdings", []):
        trade = trades.get(int(holding["lot_id"]))
        if not trade:
            continue
        trade.update(
            {
                "last_close": r4(holding.get("last_close")),
                "market_value": r2(holding.get("market_value")),
                "unrealized_pnl": r2(holding.get("unrealized_pnl")),
                "unrealized_ret_pct": r4(holding.get("unrealized_ret_pct")),
            }
        )
    return trades


def load_minutes(root: Path, code: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source in ["sina_1m", "eastmoney_1m"]:
        path = root / source / f"{code}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                try:
                    low = float(row.get("low") or row.get("最低") or 0)
                except Exception:
                    continue
                timestamp = row.get("time") or row.get("datetime") or row.get("日期") or ""
                date_text = row.get("date") or (timestamp[:10] if len(timestamp) >= 10 else "")
                rows.append({"source": source, "date": date_text, "time": timestamp, "low": low})
    rows.sort(key=lambda item: (item["date"], item["time"], item["source"]))
    return rows


def annotate_minute_stop_times(trades: Dict[int, Dict[str, Any]], minute_dir: Path) -> Dict[str, int]:
    # Current stop rule is evaluated on daily close, so there is no intraday trigger minute to annotate.
    return {}


def build_stock_summary(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stocks: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "trade_count": 0,
            "closed_count": 0,
            "open_count": 0,
            "win_count": 0,
            "total_pnl": 0.0,
            "total_fee": 0.0,
            "first_buy": None,
            "last_sell": None,
            "name": "",
        }
    )
    for trade in trades:
        code = str(trade.get("code"))
        stock = stocks[code]
        stock["name"] = trade.get("name") or stock["name"]
        stock["trade_count"] += 1
        stock["total_fee"] += float(trade.get("total_fee") or 0)
        buy_date = trade.get("buy_date")
        if buy_date and (stock["first_buy"] is None or buy_date < stock["first_buy"]):
            stock["first_buy"] = buy_date
        if trade.get("status") == "closed":
            stock["closed_count"] += 1
            pnl = float(trade.get("pnl") or 0)
            stock["total_pnl"] += pnl
            if pnl > 0:
                stock["win_count"] += 1
            sell_date = trade.get("sell_date")
            if sell_date and (stock["last_sell"] is None or sell_date > stock["last_sell"]):
                stock["last_sell"] = sell_date
        else:
            stock["open_count"] += 1

    rows: List[Dict[str, Any]] = []
    for code, stock in stocks.items():
        closed_count = int(stock["closed_count"])
        rows.append(
            {
                "code": code,
                "name": stock["name"],
                "trade_count": stock["trade_count"],
                "closed_count": closed_count,
                "open_count": stock["open_count"],
                "win_count": stock["win_count"],
                "total_pnl": r2(stock["total_pnl"]),
                "total_fee": r2(stock["total_fee"]),
                "first_buy": stock["first_buy"],
                "last_sell": stock["last_sell"],
                "win_rate_pct": round(stock["win_count"] / closed_count * 100, 4) if closed_count else None,
            }
        )
    rows.sort(key=lambda row: float(row["total_pnl"] or 0), reverse=True)
    return rows


def build_trade_payload(report: Dict[str, Any], trades: Dict[int, Dict[str, Any]], minute_counts: Dict[str, int], publish_revision: str) -> Dict[str, Any]:
    trade_list = [trades[lot] for lot in sorted(trades)]
    stock_summary = build_stock_summary(trade_list)
    return {
        "generated_at": report.get("generated_at"),
        "port": "formula_breakout_top3_10y_trades",
        "strategy_name": f"{report.get('strategy_name')}交易明细",
        "start_date": report.get("start_date"),
        "end_date": report.get("end_date"),
        "summary": {
            "trade_count": len(trade_list),
            "closed_count": sum(1 for trade in trade_list if trade.get("status") == "closed"),
            "open_count": sum(1 for trade in trade_list if trade.get("status") == "open"),
            "stock_count": len(stock_summary),
            "total_pnl_closed": r2(sum(float(trade.get("pnl") or 0) for trade in trade_list if trade.get("status") == "closed")),
            "total_fee": r2(sum(float(trade.get("total_fee") or 0) for trade in trade_list)),
            "open_unrealized_pnl": r2(sum(float(trade.get("unrealized_pnl") or 0) for trade in trade_list if trade.get("status") == "open")),
        },
        "minute_data": {
            "enabled": bool(minute_counts),
            "sources": ["sina_1m", "eastmoney_1m"],
            "note": "当前止损以日线收盘价B判断，交易明细不再补充盘中首次触发分钟。",
            "match_counts": minute_counts,
        },
        "trades": trade_list,
        "stock_summary": stock_summary,
        "publish_revision": publish_revision,
    }


def main() -> None:
    args = parse_args()
    full_report = Path(args.full_report)
    summary_output = Path(args.summary_output)
    trades_output = Path(args.trades_output)
    publish_revision = args.publish_revision or datetime.now().strftime("%Y-%m-%d-%H%M%S")

    report = read_json(full_report)
    previous_summary = read_json(summary_output) if summary_output.exists() else {}
    summary_payload = build_summary_payload(report, previous_summary, publish_revision)
    write_json(summary_output, summary_payload)

    trades = build_trade_map(report)
    minute_counts = annotate_minute_stop_times(trades, Path(args.minute_dir))
    trade_payload = build_trade_payload(report, trades, minute_counts, publish_revision)
    write_json(trades_output, trade_payload)

    print(f"wrote {summary_output}")
    print(f"wrote {trades_output}")
    print(f"trades={trade_payload['summary']}")
    print(f"minute_matches={minute_counts}")


if __name__ == "__main__":
    main()
