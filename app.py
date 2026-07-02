"""
A股量化因子可视化 - FastAPI 后端
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from factor_scraper import FactorScraper
from prediction_model import StockPredictor
from stock_analysis_service import uptrend_service

app = FastAPI(title="A股量化因子可视化", version="1.0.0")
scraper = FactorScraper()
predictor = StockPredictor()

# 静态文件目录
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """返回空 favicon 避免 404"""
    return Response(content=b"", media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse)
async def index():
    """每日 Top20 首页"""
    html_path = Path(__file__).parent / "templates" / "daily_top20.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>请先创建 templates/daily_top20.html</h1>")


@app.get("/daily-top20", response_class=HTMLResponse)
async def daily_top20_page():
    """每日 Top20 选股页"""
    html_path = Path(__file__).parent / "templates" / "daily_top20.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>请先创建 templates/daily_top20.html</h1>")


@app.get("/uptrend", response_class=HTMLResponse)
async def uptrend_page():
    """主升模型单股分析页"""
    html_path = Path(__file__).parent / "templates" / "uptrend.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>请先创建 templates/uptrend.html</h1>")


@app.get("/june-backtest", response_class=HTMLResponse)
async def june_backtest_page():
    """六月每日 Top20 回测前端"""
    html_path = Path(__file__).parent / "templates" / "june_backtest.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>请先创建 templates/june_backtest.html</h1>")


@app.get("/api/june-backtest")
async def june_backtest_data():
    """读取六月 Top20 回测数据"""
    import json

    candidates = [
        STATIC_DIR / "reports" / "june_top20_backtest.json",
        Path(__file__).parent / "data_cache" / "june_top20_backtest" / "june_top20_backtest.json",
    ]
    for path in candidates:
        if path.exists():
            return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))
    return JSONResponse(status_code=404, content={"message": "暂无回测数据，请先运行 june_top20_backtest.py"})


@app.get("/api/daily-top20")
async def daily_top20_data():
    """读取每日 Top20 最新快照分析"""
    import json

    candidates = [
        STATIC_DIR / "reports" / "daily_top20.json",
        Path(__file__).parent / "data_cache" / "daily_top20_snapshots" / "latest_analysis.json",
    ]
    for path in candidates:
        if path.exists():
            return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))
    return JSONResponse(status_code=404, content={"message": "暂无每日 Top20 数据，请先运行 daily_top20_pipeline.py --run-once"})


@app.get("/api/uptrend/search")
async def uptrend_search(q: str = "", exclude_chinext: bool = False, limit: int = 30):
    """搜索股票代码/名称"""
    try:
        return JSONResponse(content={"items": uptrend_service.search(q, exclude_chinext=exclude_chinext, limit=limit)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "message": "搜索失败"})


@app.get("/api/uptrend/analyze/{code}")
async def uptrend_analyze(code: str, refresh: bool = True, start_date: str = ""):
    """使用 5 年主升模型分析单只股票"""
    try:
        result = uptrend_service.analyze(code, refresh=refresh, start_date=start_date or None)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "message": "分析失败"})


@app.get("/api/data")
async def get_factor_data():
    """获取量化因子数据（实时爬取）"""
    try:
        data = scraper.get_recommended_data()
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "数据获取失败，请稍后重试"},
        )


@app.get("/api/refresh")
async def refresh_and_cache():
    """刷新并缓存数据"""
    try:
        data = scraper.get_recommended_data()
        scraper.save_to_cache(data)
        return JSONResponse(content={"status": "ok", "message": "数据已刷新并缓存"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "刷新失败"},
        )


@app.get("/api/train")
async def train_model():
    """训练量化因子预测模型"""
    try:
        data = scraper.get_recommended_data()
        stock_list = (
            data.get("hot_stocks", [])
            or data.get("recommended_stocks", [])
            or []
        )
        if stock_list:
            stock_list = [
                {"代码": s.get("代码", s.get("股票代码", "")), "名称": s.get("股票名称", s.get("名称", ""))}
                for s in stock_list if s.get("代码") or s.get("股票代码")
            ]
        result = predictor.train(stock_list)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "模型训练失败"},
        )


@app.get("/api/predict")
async def get_predictions():
    """获取未来5日涨跌概率预测"""
    try:
        if predictor.model is None and not predictor.load_model():
            return JSONResponse(
                status_code=400,
                content={"message": "请先访问 /api/train 训练模型"},
            )
        data = scraper.get_recommended_data()
        stock_list = data.get("hot_stocks", []) or []
        if stock_list:
            stock_list = [
                {"代码": s.get("代码", s.get("股票代码", "")), "名称": s.get("股票名称", s.get("名称", ""))}
                for s in stock_list if s.get("代码") or s.get("股票代码")
            ]
        preds = predictor.predict(stock_list, top_n=20)
        return JSONResponse(
            content={
                "update_time": __import__("datetime").datetime.now().isoformat(),
                "predictions": preds,
                "horizon_days": 5,
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "预测失败"},
        )


@app.get("/api/cached")
async def get_cached_data():
    """获取缓存数据（不爬取）"""
    cache_path = scraper.cache_dir / "factor_data.json"
    if not cache_path.exists():
        return JSONResponse(
            status_code=404,
            content={"message": "暂无缓存数据，请先访问 /api/refresh 或 /api/data"},
        )
    import json

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return JSONResponse(content=data)


if __name__ == "__main__":
    import socket
    import uvicorn

    def _find_free_port(start: int = 8002, max_tries: int = 10) -> int:
        for p in range(start, start + max_tries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", p))
                    return p
            except OSError:
                continue
        return start

    port = _find_free_port()
    print(f"访问 http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
