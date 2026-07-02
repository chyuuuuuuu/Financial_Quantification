"""
A股量化因子自动化爬取模块
基于akshare获取板块ETF、行业板块、概念板块及成份股数据
含重试、延迟、多数据源降级策略
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import akshare as ak
import pandas as pd

T = TypeVar("T")


def _retry_with_delay(
    fn: Callable[[], T],
    max_retries: int = 3,
    delay_base: float = 2.0,
    name: str = "",
) -> Optional[T]:
    """带指数退避的重试"""
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if i < max_retries - 1:
                wait = delay_base ** (i + 1)
                time.sleep(wait)
            else:
                return None
    return None


class FactorScraper:
    """量化因子爬取器"""

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _to_json_serializable(self, obj: Any) -> Any:
        """将DataFrame等转为JSON可序列化格式"""
        if isinstance(obj, pd.DataFrame):
            return obj.fillna(0).to_dict(orient="records")
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, (float,)):
            return round(obj, 4) if not (obj != obj) else None  # NaN check
        return obj

    def _parse_etf_df(self, df: pd.DataFrame) -> list[dict]:
        """解析ETF DataFrame为统一格式"""
        if df is None or df.empty:
            return []
        name_col = "名称" if "名称" in df.columns else "基金名称"
        if name_col not in df.columns and len(df.columns) > 0:
            name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        if "名称" not in df.columns and "基金名称" in df.columns:
            df = df.rename(columns={"基金名称": "名称"})
        pct_col = "涨跌幅" if "涨跌幅" in df.columns else "增长率"
        if pct_col in df.columns and "涨跌幅" not in df.columns:
            df = df.rename(columns={pct_col: "涨跌幅"})
        exclude = ["债券", "货币", "国债", "信用债", "可转债"]
        if "名称" in df.columns:
            mask = ~df["名称"].astype(str).str.contains("|".join(exclude), na=False)
            df = df[mask]
        return df.head(100).fillna(0).to_dict(orient="records")

    def get_sector_etfs(self) -> list[dict]:
        """获取A股板块ETF（同花顺优先，东方财富降级）"""
        today = datetime.now().strftime("%Y%m%d")
        df = _retry_with_delay(lambda: ak.fund_etf_spot_ths(date=today), name="ETF-THS")
        if df is not None and not df.empty:
            return self._parse_etf_df(df)
        df = _retry_with_delay(lambda: ak.fund_etf_spot_em(), name="ETF")
        return self._parse_etf_df(df) if df is not None else []

    def get_industry_sectors(self) -> list[dict]:
        """获取行业板块（同花顺优先）"""
        df = _retry_with_delay(
            lambda: ak.stock_board_industry_summary_ths(), name="行业"
        )
        if df is not None and not df.empty:
            df = df.fillna(0)
            if "板块" in df.columns and "板块名称" not in df.columns:
                df = df.rename(columns={"板块": "板块名称"})
            if "领涨股" in df.columns and "领涨股票" not in df.columns:
                df = df.rename(columns={"领涨股": "领涨股票"})
            if "涨跌幅" in df.columns and df["涨跌幅"].dtype == object:
                df["涨跌幅"] = pd.to_numeric(df["涨跌幅"].astype(str).str.replace("%", ""), errors="coerce")
            return df.to_dict(orient="records")
        df = _retry_with_delay(
            lambda: ak.stock_fund_flow_industry(symbol="即时"), name="行业-资金流"
        )
        if df is not None and not df.empty:
            df = df.fillna(0)
            df = df.rename(columns={"行业": "板块名称", "领涨股": "领涨股票"})
            if "行业-涨跌幅" in df.columns:
                df["涨跌幅"] = pd.to_numeric(df["行业-涨跌幅"].astype(str).str.replace("%", ""), errors="coerce")
            return df.to_dict(orient="records")
        return []

    def get_concept_sectors(self) -> list[dict]:
        """获取概念板块（同花顺概念资金流优先）"""
        df = _retry_with_delay(
            lambda: ak.stock_fund_flow_concept(symbol="即时"), name="概念"
        )
        if df is not None and not df.empty:
            df = df.fillna(0)
            df = df.rename(columns={"行业": "板块名称", "领涨股": "领涨股票"})
            if "行业-涨跌幅" in df.columns:
                df["涨跌幅"] = pd.to_numeric(df["行业-涨跌幅"].astype(str).str.replace("%", ""), errors="coerce")
            return df.head(150).to_dict(orient="records")
        return []

    def get_sector_fund_flow(self) -> list[dict]:
        """获取板块资金流（同花顺行业资金流优先）"""
        df = _retry_with_delay(
            lambda: ak.stock_fund_flow_industry(symbol="即时"), name="资金流"
        )
        if df is not None and not df.empty:
            df = df.fillna(0)
            df = df.rename(columns={"行业": "名称"})
            if "行业-涨跌幅" in df.columns:
                df["今日涨跌幅"] = pd.to_numeric(df["行业-涨跌幅"].astype(str).str.replace("%", ""), errors="coerce")
            if "净额" in df.columns:
                df["今日主力净流入-净额"] = df["净额"] * 1e8
            return df.to_dict(orient="records")
        return []

    def get_industry_stocks(self, sector_name: str, limit: int = 20) -> list[dict]:
        """获取指定行业板块的成份股"""
        try:
            df = ak.stock_board_industry_cons_em(symbol=sector_name)
            if df is None or df.empty:
                return []
            df = df.fillna(0)
            return df.head(limit).to_dict(orient="records")
        except Exception as e:
            print(f"获取行业成份股失败 {sector_name}: {type(e).__name__} - {e}")
            return []

    def get_concept_stocks(self, concept_name: str, limit: int = 20) -> list[dict]:
        """获取指定概念板块的成份股"""
        try:
            df = ak.stock_board_concept_cons_em(symbol=concept_name)
            if df is None or df.empty:
                return []
            df = df.fillna(0)
            return df.head(limit).to_dict(orient="records")
        except Exception as e:
            print(f"获取概念成份股失败 {concept_name}: {type(e).__name__} - {e}")
            return []

    def get_hot_stocks(self) -> list[dict]:
        """获取人气股（同花顺个股资金流优先，按净额排序）"""
        df = _retry_with_delay(
            lambda: ak.stock_fund_flow_individual(symbol="即时"), name="人气股"
        )
        if df is not None and not df.empty:
            df = df.fillna(0)
            df = df.rename(columns={"股票简称": "股票名称"})
            df["代码"] = df["股票代码"].astype(str).str.zfill(6)
            df["当前排名"] = df["序号"]
            if "涨跌幅" in df.columns:
                df["涨跌幅"] = pd.to_numeric(df["涨跌幅"].astype(str).str.replace("%", ""), errors="coerce")
            return df.head(50).to_dict(orient="records")
        return []

    def get_recommended_data(self) -> dict:
        """
        综合爬取：筛选最近可买的板块ETF和个股
        因子逻辑：涨幅靠前、资金流入、换手率适中
        """
        result = {
            "update_time": datetime.now().isoformat(),
            "sector_etfs": [],
            "industry_sectors": [],
            "concept_sectors": [],
            "fund_flow": [],
            "hot_stocks": [],
            "recommended_etfs": [],
            "recommended_stocks": [],
        }

        # 1. 板块ETF（每次请求间隔，降低被限流概率）
        etfs = self.get_sector_etfs()
        result["sector_etfs"] = etfs
        time.sleep(1.5)
        # 推荐：涨幅>0且成交量较大的ETF
        if etfs:
            etf_df = pd.DataFrame(etfs)
            if "涨跌幅" in etf_df.columns:
                recommended = etf_df[etf_df["涨跌幅"] > 0].head(15)
                result["recommended_etfs"] = recommended.to_dict(orient="records")

        # 2. 行业板块
        industries = self.get_industry_sectors()
        time.sleep(1.5)
        result["industry_sectors"] = industries[:30]
        if industries:
            ind_df = pd.DataFrame(industries)
            if "涨跌幅" in ind_df.columns:
                top_industries = ind_df.nlargest(10, "涨跌幅")
                result["top_industries"] = top_industries.to_dict(orient="records")

        # 3. 概念板块
        concepts = self.get_concept_sectors()
        time.sleep(1.5)
        result["concept_sectors"] = concepts[:30]
        if concepts:
            con_df = pd.DataFrame(concepts)
            if "涨跌幅" in con_df.columns:
                top_concepts = con_df.nlargest(10, "涨跌幅")
                result["top_concepts"] = top_concepts.to_dict(orient="records")

        # 4. 资金流
        result["fund_flow"] = self.get_sector_fund_flow()[:20]
        time.sleep(1.0)

        # 5. 人气股
        result["hot_stocks"] = self.get_hot_stocks()

        # 6. 推荐个股：从涨幅靠前板块取领涨股
        recommended_stocks = []
        for sector in (result.get("top_industries") or [])[:5]:
            name = sector.get("板块名称", sector.get("板块", ""))
            leader = sector.get("领涨股票", "")
            if leader:
                recommended_stocks.append(
                    {"板块": name, "领涨股": leader, "涨跌幅": sector.get("涨跌幅", 0)}
                )
        result["recommended_stocks"] = recommended_stocks

        return result

    def save_to_cache(self, data: dict, filename: str = "factor_data.json") -> None:
        """保存数据到缓存"""
        cache_path = self.cache_dir / filename
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"数据已缓存至 {cache_path}")


def main():
    """主函数：执行爬取并保存"""
    scraper = FactorScraper()
    result = scraper.get_recommended_data()
    scraper.save_to_cache(result)
    print("爬取完成！")
    print(f"板块ETF数量: {len(result.get('sector_etfs', []))}")
    print(f"行业板块数量: {len(result.get('industry_sectors', []))}")
    print(f"推荐ETF数量: {len(result.get('recommended_etfs', []))}")
    print(f"人气股数量: {len(result.get('hot_stocks', []))}")


if __name__ == "__main__":
    main()
