"""
量化因子模型 - 基于历史数据训练，预测未来5日涨跌概率
"""

import json
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _retry(fn, max_retries=3, delay=2.0):
    for i in range(max_retries):
        try:
            return fn()
        except Exception:
            if i < max_retries - 1:
                time.sleep(delay * (i + 1))
    return None


def _symbol_to_ak(symbol: str) -> str:
    """转换为 akshare 股票代码格式"""
    s = str(symbol).strip().upper()
    if s.startswith("SH") or s.startswith("SZ"):
        s = s[2:]
    return s


def _normalize_hist_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一历史数据列名"""
    m = {"date": "日期", "open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量"}
    for k, v in m.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    return df


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算量化因子"""
    if df is None or len(df) < 30:
        return None
    df = _normalize_hist_df(df)
    if "收盘" not in df.columns:
        return None
    df = df.sort_values("日期").reset_index(drop=True)
    close = df["收盘"]
    vol = df["成交量"] if "成交量" in df.columns else pd.Series([1] * len(df))
    # 收益率因子
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_20"] = close.pct_change(20)
    # 波动率
    df["vol_20"] = close.pct_change().rolling(20).std()
    # 成交量比
    df["vol_ratio"] = vol / (vol.rolling(20).mean() + 1e-8)
    # 换手率（若有）
    if "换手率" in df.columns:
        df["turn_5"] = df["换手率"].rolling(5).mean()
    else:
        df["turn_5"] = 0
    return df


def get_label_future_return(df: pd.DataFrame, days: int = 5) -> pd.Series:
    """未来 N 日收益率作为标签"""
    close = df["收盘"]
    future = close.shift(-days) / close - 1
    return future


class StockPredictor:
    """股票涨跌预测器"""

    def __init__(self, model_dir: str = "model_cache"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.scaler = None
        self.feature_cols = [
            "ret_5", "ret_10", "ret_20", "vol_20", "vol_ratio", "turn_5"
        ]

    def _fetch_hist(self, symbol: str, days: int = 500) -> pd.DataFrame:
        end = datetime.now()
        start = end - timedelta(days=days)
        code = _symbol_to_ak(symbol)
        prefix = "sh" if code.startswith(("6", "5")) else "sz"
        sym = f"{prefix}{code}"

        df = _retry(lambda: ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq"
        ))
        if df is not None and not df.empty:
            return df
        df = _retry(lambda: ak.stock_zh_a_daily(
            symbol=sym,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq"
        ))
        if df is not None and not df.empty:
            m = {"date": "日期", "open": "开盘", "close": "收盘", "high": "最高", "low": "最低"}
            if "volume" in df.columns:
                m["volume"] = "成交量"
            df = df.rename(columns=m)
            if "成交量" not in df.columns:
                df["成交量"] = 1
            return df
        df = _retry(lambda: ak.stock_zh_a_hist_tx(
            symbol=sym,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq"
        ))
        if df is not None and not df.empty:
            df = df.rename(columns={"date": "日期", "open": "开盘", "close": "收盘", "high": "最高", "low": "最低"})
            df["成交量"] = df["amount"] if "amount" in df.columns else 1
            return df
        return None

    def build_dataset(
        self,
        stock_list: list,
        lookback_days: int = 500,
        horizon_days: int = 5,
        max_stocks: int = 150,
    ) -> pd.DataFrame:
        """构建训练数据集"""
        rows = []
        for i, item in enumerate(stock_list[:max_stocks]):
            code = item.get("代码", item.get("股票代码", str(item)))
            code = _symbol_to_ak(code)
            name = item.get("名称", item.get("股票名称", code))
            time.sleep(0.3)
            df = self._fetch_hist(code, lookback_days)
            if df is None or len(df) < 60:
                continue
            df = compute_factors(df)
            if df is None:
                continue
            df["label"] = (get_label_future_return(df, horizon_days) > 0).astype(int)
            df["code"] = code
            df["name"] = name
            df = df.dropna(subset=self.feature_cols + ["label"])
            rows.append(df)
            if (i + 1) % 20 == 0:
                time.sleep(2)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def train(
        self,
        stock_list: list = None,
        use_cache: bool = True,
    ) -> dict:
        """训练模型"""
        if not HAS_SKLEARN:
            return {"error": "请安装 scikit-learn: pip install scikit-learn"}

        if stock_list is None or len(stock_list) == 0:
            sh_df = _retry(lambda: ak.stock_info_sh_name_code(symbol="主板A股"))
            sz_df = _retry(lambda: ak.stock_info_sz_name_code(symbol="A股列表"))
            stock_list = []
            if sh_df is not None and not sh_df.empty:
                stock_list.extend([
                    {"代码": str(row["证券代码"]), "名称": row["证券简称"]}
                    for _, row in sh_df.head(100).iterrows()
                ])
            if sz_df is not None and not sz_df.empty:
                stock_list.extend([
                    {"代码": str(row["A股代码"]), "名称": row["A股简称"]}
                    for _, row in sz_df.head(100).iterrows()
                ])
            if not stock_list:
                return {"error": "无法获取股票列表"}

        dataset = self.build_dataset(stock_list, max_stocks=120)
        if dataset.empty or len(dataset) < 100:
            return {"error": f"有效样本不足，仅 {len(dataset)} 条"}

        X = dataset[self.feature_cols].fillna(0)
        y = dataset["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        )
        self.model.fit(X_train_s, y_train)
        acc = self.model.score(X_test_s, y_test)

        self.model_dir.joinpath("model.pkl").write_bytes(
            pickle.dumps({"model": self.model, "scaler": self.scaler})
        )
        return {
            "status": "ok",
            "samples": len(dataset),
            "accuracy": round(acc, 4),
            "feature_importance": dict(zip(
                self.feature_cols,
                [round(float(x), 4) for x in self.model.feature_importances_]
            )),
        }

    def load_model(self) -> bool:
        """加载已训练模型"""
        p = self.model_dir / "model.pkl"
        if not p.exists():
            return False
        try:
            data = pickle.loads(p.read_bytes())
            self.model = data["model"]
            self.scaler = data["scaler"]
            return True
        except Exception:
            return False

    def predict(self, stock_list: list, top_n: int = 20) -> list:
        """预测未来5日涨跌概率"""
        if self.model is None and not self.load_model():
            return []

        results = []
        for item in stock_list[:80]:
            raw = item.get("代码", item.get("股票代码", ""))
            if not raw:
                continue
            code = _symbol_to_ak(raw)
            name = item.get("名称", item.get("股票名称", code))
            time.sleep(0.25)
            df = self._fetch_hist(code, 60)
            if df is None or len(df) < 30:
                continue
            df = compute_factors(df)
            if df is None:
                continue
            last = df.iloc[-1]
            X = last[self.feature_cols].fillna(0).values.reshape(1, -1)
            X_s = self.scaler.transform(X)
            proba = self.model.predict_proba(X_s)[0]
            prob_down = float(proba[0])
            prob_up = float(proba[1])
            results.append({
                "code": code,
                "name": name,
                "prob_up": round(prob_up * 100, 2),
                "prob_down": round(prob_down * 100, 2),
                "suggest": "看涨" if prob_up > 0.55 else ("看跌" if prob_down > 0.55 else "中性"),
            })
            if len(results) >= top_n:
                break

        results.sort(key=lambda x: x["prob_up"], reverse=True)
        return results


def main():
    """训练并预测示例"""
    predictor = StockPredictor()
    print("获取股票列表...")
    hot_df = _retry(lambda: ak.stock_hot_rank_em())
    if hot_df is not None and not hot_df.empty:
        stock_list = [
            {"代码": row["代码"].replace("SH", "").replace("SZ", ""), "名称": row["股票名称"]}
            for _, row in hot_df.head(80).iterrows()
        ]
    else:
        stock_list = None

    print("训练模型...")
    ret = predictor.train(stock_list)
    print(json.dumps(ret, ensure_ascii=False, indent=2))

    if "error" not in ret:
        print("\n预测未来5日涨跌概率...")
        preds = predictor.predict(stock_list or [], top_n=15)
        for p in preds:
            print(f"  {p['name']}({p['code']}): 涨{p['prob_up']}% / 跌{p['prob_down']}% - {p['suggest']}")


if __name__ == "__main__":
    main()
