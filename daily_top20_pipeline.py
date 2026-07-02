#!/usr/bin/env python
"""Daily 15:30 Top20 pipeline with immutable snapshots and adaptive ranking."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from main_uptrend_model import normalize_hist


FEATURE_COLS = [
    "base_rank",
    "final_score",
    "rule_score",
    "model_prob",
    "pct_chg",
    "amount_log",
    "entry_age",
    "trigger_age",
    "entry_volume_ratio_20",
    "entry_amount_ratio_20",
    "volume_10_vs_entry",
    "amount_10_vs_entry",
    "trigger_volume_vs_entry",
    "trigger_amount_vs_entry",
    "entry_close_breakout_gap",
    "ret_since_entry",
    "max_ret_since_entry",
    "post_entry_min_ret",
    "close_to_high_60",
    "is_breakout_hold",
    "is_contraction_retest",
    "is_entry_close_breakout",
]


@dataclass
class AdaptiveModel:
    model: object
    feature_cols: List[str]
    trained_at: str
    rows: int
    metrics: Dict[str, float]


def normalize_code(value: object) -> str:
    return str(value).strip().zfill(6)[-6:]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    return df.replace({np.nan: None}).to_dict(orient="records")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def latest_snapshot_time(run_time: datetime) -> str:
    return run_time.strftime("%Y%m%d_%H%M")


def ymd(run_time: datetime) -> str:
    return run_time.strftime("%Y-%m-%d")


def compact_ymd(run_time: datetime) -> str:
    return run_time.strftime("%Y%m%d")


def next_run_time(hour: int, minute: int, now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def run_refresh_and_score(args: argparse.Namespace, run_time: datetime, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        "refresh_and_screen_uptrend.py",
        "--universe-file",
        args.universe_file,
        "--history-dir",
        args.history_dir,
        "--model-path",
        args.model_path,
        "--output-dir",
        str(output_dir),
        "--target-date",
        ymd(run_time),
        "--workers",
        str(args.workers),
        "--retry",
        str(args.retry),
        "--progress-every",
        str(args.progress_every),
        "--candidate-min-amount",
        str(args.candidate_min_amount),
        "--snapshot-top-n",
        str(args.raw_snapshot_top_n),
        "--prediction-log-dir",
        str(args.raw_snapshot_dir),
    ]
    if args.quote_only:
        cmd.append("--quote-only")
    if args.no_skip_current:
        cmd.append("--no-skip-current")
    print(f"[{now_text()}] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        out[col] = 0.0
    out["base_rank"] = pd.to_numeric(df.get("base_rank", df.get("rank", 0)), errors="coerce").fillna(0)
    for col in [
        "final_score",
        "rule_score",
        "model_prob",
        "pct_chg",
        "entry_age",
        "trigger_age",
        "entry_volume_ratio_20",
        "entry_amount_ratio_20",
        "volume_10_vs_entry",
        "amount_10_vs_entry",
        "trigger_volume_vs_entry",
        "trigger_amount_vs_entry",
        "entry_close_breakout_gap",
        "ret_since_entry",
        "max_ret_since_entry",
        "post_entry_min_ret",
        "close_to_high_60",
    ]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    amount = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0).clip(lower=0)
    out["amount_log"] = np.log1p(amount)
    signal = df.get("signal_type", "").fillna("").astype(str)
    out["is_breakout_hold"] = (signal == "breakout_hold").astype(float)
    out["is_contraction_retest"] = (signal == "contraction_retest").astype(float)
    out["is_entry_close_breakout"] = (signal == "entry_close_breakout").astype(float)
    return out[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)


def load_adaptive_model(path: Path) -> Optional[AdaptiveModel]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def apply_adaptive_score(candidates: pd.DataFrame, model_path: Path, weight: float) -> Tuple[pd.DataFrame, Dict[str, object]]:
    out = candidates.copy()
    model = load_adaptive_model(model_path)
    if model is None:
        out["pred_next_day_ret_pct"] = np.nan
        out["adaptive_score"] = pd.to_numeric(out["final_score"], errors="coerce").fillna(0)
        return out, {"enabled": False, "reason": "no adaptive model"}
    features = prepare_features(out)
    pred = model.model.predict(features[model.feature_cols])
    out["pred_next_day_ret_pct"] = np.asarray(pred, dtype=float)
    base = pd.to_numeric(out["final_score"], errors="coerce").fillna(0)
    out["adaptive_score"] = (base + float(weight) * out["pred_next_day_ret_pct"]).clip(lower=0, upper=120)
    return out, {
        "enabled": True,
        "trained_at": model.trained_at,
        "rows": model.rows,
        "metrics": model.metrics,
        "weight": weight,
    }


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str})
    if "rank" not in df.columns:
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    rank_fallback = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    df["base_rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(rank_fallback)
    df["code"] = df["code"].map(normalize_code)
    return df


def save_snapshots(
    candidates: pd.DataFrame,
    args: argparse.Namespace,
    run_time: datetime,
    run_dir: Path,
    model_info: Dict[str, object],
) -> Tuple[Path, Path, pd.DataFrame]:
    snapshots_dir = Path(args.snapshot_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    stamp = latest_snapshot_time(run_time)

    ranked = candidates.sort_values(["adaptive_score", "final_score", "amount"], ascending=False).reset_index(drop=True)
    ranked.insert(0, "snapshot_rank", np.arange(1, len(ranked) + 1))
    ranked.insert(0, "snapshot_time", run_time.isoformat(timespec="seconds"))
    ranked.insert(0, "snapshot_date", ymd(run_time))
    ranked["adaptive_model_enabled"] = bool(model_info.get("enabled"))

    all_path = snapshots_dir / f"candidates_{stamp}.csv"
    top20_path = snapshots_dir / f"top20_{stamp}.csv"
    ranked.head(args.saved_candidates).to_csv(all_path, index=False, encoding="utf-8-sig")
    top20 = ranked.head(args.top_n).copy()
    top20.to_csv(top20_path, index=False, encoding="utf-8-sig")

    latest_top20 = snapshots_dir / "latest_top20.csv"
    latest_candidates = snapshots_dir / "latest_candidates.csv"
    top20.to_csv(latest_top20, index=False, encoding="utf-8-sig")
    ranked.head(args.saved_candidates).to_csv(latest_candidates, index=False, encoding="utf-8-sig")

    run_dir.mkdir(parents=True, exist_ok=True)
    top20.to_csv(run_dir / "top20.csv", index=False, encoding="utf-8-sig")
    ranked.head(args.saved_candidates).to_csv(run_dir / "ranked_candidates.csv", index=False, encoding="utf-8-sig")
    return all_path, top20_path, top20


def next_trade_return(row: pd.Series, history_dir: Path) -> Optional[Dict[str, object]]:
    code = normalize_code(row.get("code"))
    latest_date = pd.Timestamp(str(row.get("latest_date") or row.get("snapshot_date"))[:10])
    base = safe_float(row.get("close"), 0)
    if base <= 0:
        return None
    path = history_dir / f"{code}.csv"
    if not path.exists():
        return None
    try:
        hist = normalize_hist(pd.read_csv(path)).sort_values("日期")
    except Exception:
        return None
    future = hist[pd.to_datetime(hist["日期"]).dt.normalize() > latest_date.normalize()].head(1)
    if future.empty:
        return None
    item = future.iloc[0]
    close = safe_float(item["收盘"], 0)
    if close <= 0:
        return None
    return {
        "next_date": pd.Timestamp(item["日期"]).strftime("%Y-%m-%d"),
        "next_ret_pct": round((close / base - 1.0) * 100.0, 4),
        "next_high_pct": round((safe_float(item["最高"]) / base - 1.0) * 100.0, 4),
        "next_low_pct": round((safe_float(item["最低"]) / base - 1.0) * 100.0, 4),
    }


def update_feedback(args: argparse.Namespace) -> pd.DataFrame:
    snapshots_dir = Path(args.snapshot_dir)
    history_dir = Path(args.history_dir)
    feedback_path = Path(args.feedback_file)
    rows: List[Dict[str, object]] = []
    for path in sorted(snapshots_dir.glob("candidates_*.csv")):
        try:
            df = pd.read_csv(path, dtype={"code": str})
        except Exception:
            continue
        for item in df.to_dict(orient="records"):
            ret = next_trade_return(pd.Series(item), history_dir)
            if ret is None:
                continue
            rows.append({**item, **ret, "snapshot_file": path.name})
    feedback = pd.DataFrame(rows)
    if not feedback.empty:
        feedback = feedback.drop_duplicates(["snapshot_time", "code"], keep="last")
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback.to_csv(feedback_path, index=False, encoding="utf-8-sig")
    return feedback


def train_adaptive_model(args: argparse.Namespace, feedback: pd.DataFrame) -> Dict[str, object]:
    if len(feedback) < args.min_train_rows:
        return {"trained": False, "reason": f"feedback rows {len(feedback)} < {args.min_train_rows}"}
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        return {"trained": False, "reason": f"sklearn unavailable: {exc}"}

    train_df = feedback.dropna(subset=["next_ret_pct"]).copy()
    if len(train_df) < args.min_train_rows:
        return {"trained": False, "reason": f"valid rows {len(train_df)} < {args.min_train_rows}"}
    x = prepare_features(train_df)
    y = pd.to_numeric(train_df["next_ret_pct"], errors="coerce").fillna(0)
    test_size = 0.25 if len(train_df) >= 120 else 0.2
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    model = RandomForestRegressor(
        n_estimators=240,
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    corr = float(np.corrcoef(pred, y_test)[0, 1]) if len(y_test) > 2 and np.std(pred) > 0 and np.std(y_test) > 0 else 0.0
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "corr": round(corr, 4),
        "test_rows": int(len(y_test)),
    }
    payload = AdaptiveModel(
        model=model,
        feature_cols=FEATURE_COLS,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        rows=int(len(train_df)),
        metrics=metrics,
    )
    model_path = Path(args.adaptive_model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return {"trained": True, "rows": int(len(train_df)), "metrics": metrics, "model_path": str(model_path)}


def build_analysis(
    args: argparse.Namespace,
    run_time: datetime,
    top20: pd.DataFrame,
    candidates: pd.DataFrame,
    feedback: pd.DataFrame,
    train_info: Dict[str, object],
    model_info: Dict[str, object],
) -> Dict[str, object]:
    signal_summary = []
    for signal, group in top20.groupby("signal_type", dropna=False):
        signal_summary.append(
            {
                "signal_type": str(signal),
                "count": int(len(group)),
                "avg_score": round(float(pd.to_numeric(group["adaptive_score"], errors="coerce").mean()), 3),
                "avg_base_score": round(float(pd.to_numeric(group["final_score"], errors="coerce").mean()), 3),
            }
        )
    recent_feedback = pd.DataFrame()
    if not feedback.empty:
        recent_feedback = feedback.sort_values("snapshot_time").tail(args.feedback_window)
    analysis = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_date": ymd(run_time),
        "snapshot_time": run_time.isoformat(timespec="seconds"),
        "top_n": args.top_n,
        "candidate_rows": int(len(candidates)),
        "saved_candidates": int(min(args.saved_candidates, len(candidates))),
        "adaptive_model": model_info,
        "training": train_info,
        "signal_summary": signal_summary,
        "top_rows": clean_records(top20),
        "recent_feedback": {
            "rows": int(len(recent_feedback)),
            "avg_next_ret_pct": round(float(recent_feedback["next_ret_pct"].mean()), 4) if not recent_feedback.empty else None,
            "win_rate_pct": round(float((recent_feedback["next_ret_pct"] > 0).mean() * 100.0), 2) if not recent_feedback.empty else None,
        },
    }
    return analysis


def write_static_report(analysis: Dict[str, object], args: argparse.Namespace) -> None:
    static_dir = Path(args.static_dir)
    reports_dir = static_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "daily_top20.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    template = Path(args.template)
    if template.exists():
        html = template.read_text(encoding="utf-8")
        (static_dir / "index.html").write_text(html, encoding="utf-8")
        (static_dir / "daily-top20.html").write_text(html, encoding="utf-8")


def run_once(args: argparse.Namespace, run_time: Optional[datetime] = None) -> Dict[str, object]:
    run_time = run_time or datetime.now().replace(second=0, microsecond=0)
    run_dir = Path(args.run_dir) / latest_snapshot_time(run_time)
    if not args.skip_refresh:
        run_refresh_and_score(args, run_time, run_dir)
    candidates_path = run_dir / "contraction_current_candidates.csv"
    if args.skip_refresh:
        candidates_path = Path(args.candidates_file)
    if not candidates_path.exists():
        raise FileNotFoundError(f"candidate file not found: {candidates_path}")
    candidates = load_candidates(candidates_path)
    feedback = update_feedback(args)
    train_info = train_adaptive_model(args, feedback) if args.train_adaptive else {"trained": False, "reason": "disabled"}
    scored, model_info = apply_adaptive_score(candidates, Path(args.adaptive_model_path), args.adaptive_weight)
    _, _, top20 = save_snapshots(scored, args, run_time, run_dir, model_info)
    feedback = update_feedback(args)
    analysis = build_analysis(args, run_time, top20, scored, feedback, train_info, model_info)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.snapshot_dir, "latest_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    write_static_report(analysis, args)
    print(f"[{now_text()}] saved top20 and analysis under {run_dir}")
    return analysis


def daemon(args: argparse.Namespace) -> None:
    print(f"[{now_text()}] daily scheduler started, target={args.schedule_hour:02d}:{args.schedule_minute:02d}")
    while True:
        target = next_run_time(args.schedule_hour, args.schedule_minute)
        seconds = max(1, int((target - datetime.now()).total_seconds()))
        print(f"[{now_text()}] next run at {target:%Y-%m-%d %H:%M:%S}, sleep={seconds}s")
        time.sleep(seconds)
        try:
            run_once(args, target)
        except Exception as exc:
            print(f"[{now_text()}] run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            time.sleep(60)


def print_cron(args: argparse.Namespace) -> None:
    python = sys.executable
    root = Path.cwd()
    cmd = (
        f"30 15 * * 1-5 cd {root} && {python} daily_top20_pipeline.py --run-once "
        f"--target-date $(date +\\%F) >> data_cache/daily_top20_snapshots/cron.log 2>&1"
    )
    print(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--print-cron", action="store_true")
    parser.add_argument("--target-date", default="")
    parser.add_argument("--schedule-hour", type=int, default=15)
    parser.add_argument("--schedule-minute", type=int, default=30)
    parser.add_argument("--universe-file", default="data_cache/volume_contraction_screen_20260701_mainboard_entry_close/refresh_status.csv")
    parser.add_argument("--history-dir", default="data_cache/main_uptrend/hist")
    parser.add_argument("--model-path", default="model_cache/volume_contraction_breakout_5y_tcn.pt")
    parser.add_argument("--run-dir", default="data_cache/daily_top20_runs")
    parser.add_argument("--snapshot-dir", default="data_cache/daily_top20_snapshots")
    parser.add_argument("--raw-snapshot-dir", default="data_cache/daily_top20_raw_snapshots")
    parser.add_argument("--feedback-file", default="data_cache/daily_top20_snapshots/feedback.csv")
    parser.add_argument("--adaptive-model-path", default="model_cache/adaptive_top20_ranker.pkl")
    parser.add_argument("--static-dir", default="static")
    parser.add_argument("--template", default="templates/daily_top20.html")
    parser.add_argument("--candidates-file", default="")
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--quote-only", action="store_true", default=True)
    parser.add_argument("--full-refresh", dest="quote_only", action="store_false")
    parser.add_argument("--no-skip-current", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--candidate-min-amount", type=float, default=50_000_000)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--saved-candidates", type=int, default=200)
    parser.add_argument("--raw-snapshot-top-n", type=int, default=200)
    parser.add_argument("--adaptive-weight", type=float, default=4.0)
    parser.add_argument("--train-adaptive", action="store_true", default=True)
    parser.add_argument("--no-train-adaptive", dest="train_adaptive", action="store_false")
    parser.add_argument("--min-train-rows", type=int, default=80)
    parser.add_argument("--feedback-window", type=int, default=400)
    args = parser.parse_args()
    if args.target_date:
        target = pd.Timestamp(args.target_date).to_pydatetime()
        args._run_time = target.replace(hour=args.schedule_hour, minute=args.schedule_minute, second=0, microsecond=0)
    else:
        args._run_time = None
    return args


def main() -> None:
    args = parse_args()
    if args.print_cron:
        print_cron(args)
        return
    if args.daemon:
        daemon(args)
        return
    run_once(args, args._run_time)


if __name__ == "__main__":
    main()
