"""
接口诊断脚本 - 检查各 akshare 接口是否可用
"""

import time

def test(name, fn):
    try:
        t0 = time.time()
        r = fn()
        elapsed = time.time() - t0
        n = len(r) if hasattr(r, "__len__") else (r.shape[0] if hasattr(r, "shape") else "?")
        print(f"  [OK] {name}: {n} 条, {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__} - {e}")
        return False


def main():
    import akshare as ak

    print("=== A股接口诊断 ===\n")

    tests = [
        ("人气股-同花顺个股资金流 stock_fund_flow_individual", lambda: ak.stock_fund_flow_individual(symbol="即时")),
        ("行业-同花顺 stock_board_industry_summary_ths", lambda: ak.stock_board_industry_summary_ths()),
        ("概念-同花顺概念资金流 stock_fund_flow_concept", lambda: ak.stock_fund_flow_concept(symbol="即时")),
        ("资金流-同花顺行业资金流 stock_fund_flow_industry", lambda: ak.stock_fund_flow_industry(symbol="即时")),
        ("ETF-同花顺 fund_etf_spot_ths", lambda: ak.fund_etf_spot_ths(date=__import__("datetime").datetime.now().strftime("%Y%m%d"))),
        ("股票列表-上证 stock_info_sh_name_code", lambda: ak.stock_info_sh_name_code(symbol="主板A股")),
        ("股票列表-深证 stock_info_sz_name_code", lambda: ak.stock_info_sz_name_code(symbol="A股列表")),
        ("历史行情 stock_zh_a_hist", lambda: ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20240101", end_date="20250318", adjust="qfq")),
    ]

    ok = 0
    for name, fn in tests:
        if test(name, fn):
            ok += 1
        time.sleep(1)

    print(f"\n通过: {ok}/{len(tests)}")


if __name__ == "__main__":
    main()
