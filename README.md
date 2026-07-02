# A股量化因子爬取与可视化系统

基于 [vnpy](https://github.com/vnpy/vnpy) 与 [akshare](https://github.com/akfamily/akshare) 的 A 股量化因子自动化爬取、机器学习预测与前端可视化展示。

## 功能

- **板块 ETF**：获取 A 股可购买的板块 ETF 实时行情，筛选涨幅 > 0 的推荐标的
- **行业板块**：东方财富/同花顺行业板块实时行情，涨幅靠前板块及领涨股
- **概念板块**：概念板块行情与领涨股
- **资金流**：行业资金流排名（今日主力净流入）
- **人气股**：A 股人气榜个股（东方财富→新浪降级）
- **领涨股推荐**：从涨幅靠前板块提取领涨股
- **5日涨跌预测**：基于量化因子训练的随机森林模型，输出未来5日涨/跌概率

## 项目结构

```
Financial_Quantification/
├── vnpy/                 # vnpy 框架（已 clone）
├── factor_scraper.py     # 量化因子爬取模块
├── app.py                # FastAPI 后端
├── templates/
│   └── index.html        # 前端可视化页面
├── data_cache/           # 数据缓存目录（自动创建）
├── requirements.txt
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

### 1. 启动 Web 服务

```bash
python app.py
```

或使用 uvicorn：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. 访问界面

浏览器打开：http://localhost:8000

### 3. 仅爬取数据（不启动服务）

```bash
python factor_scraper.py
```

数据将保存至 `data_cache/factor_data.json`。

## API 接口

| 接口 | 说明 |
|------|------|
| `GET /` | 前端首页 |
| `GET /api/data` | 实时爬取并返回因子数据 |
| `GET /api/refresh` | 刷新数据并写入缓存 |
| `GET /api/cached` | 读取缓存数据（不爬取） |
| `GET /api/train` | 训练量化因子预测模型 |
| `GET /api/predict` | 获取未来5日涨跌概率预测 |

## 使用流程

1. **刷新数据**：先点击「刷新数据」获取板块、人气股等
2. **训练模型**：点击「训练模型」（首次需 2–5 分钟，取决于网络）
3. **5日预测**：点击「5日预测」查看涨跌概率推荐

### 主升浪/爆量大阳线样本模型

`main_uptrend_model.py` 用 A 股日线识别「120 个交易日内最高收盘价达到起点 5 倍」的历史案例，并只把其中出现爆量大阳线或放量突破的启动日作为正样本；推理时可筛出当前成交额、成交量和大阳线形态同时满足的候选股。

建议使用 Python 3.8+ 环境运行，例如本机 `py310`：

```bash
/home/luochangyu/anaconda3/envs/py310/bin/python main_uptrend_model.py \
  --output-dir data_cache/main_uptrend \
  --min-amount 0 \
  --candidate-min-amount 50000000 \
  --candidate-ready-only \
  --max-stocks 0 \
  --include-bj \
  --lookback-years 6 \
  --multiple 5 \
  --workers 10
```

主要输出：

- `data_cache/main_uptrend/detected_5x_events.csv`：历史半年 5 倍且爆量启动样本
- `data_cache/main_uptrend/training_samples.csv`：训练样本
- `data_cache/main_uptrend/current_main_uptrend_candidates.csv`：当前主升候选
- `model_cache/main_uptrend_model.pkl`：训练后的模型

### 潜伏主升模型：爆量进场后 30-100 日再启动

`latent_uptrend_model.py` 针对「不是已经启动，而是即将进入主升」的场景：

- 历史标签：先出现爆量进场信号，之后第 30-100 个交易日内出现快速拉升，且 200 个交易日内达到 5 倍
- 正样本：快速拉升日前 1-15 个交易日的潜伏窗口
- 当前候选：最近 30-100 个交易日内有过爆量进场，当前尚未明显主升
- 模型：PyTorch Transformer + TCN 时序模型，默认使用 CUDA；本机建议单张 A100 更稳定

```bash
/home/luochangyu/anaconda3/envs/py310/bin/python latent_uptrend_model.py \
  --output-dir data_cache/latent_uptrend \
  --history-dir data_cache/main_uptrend/hist \
  --spot-cache-dir data_cache/main_uptrend \
  --max-stocks 0 \
  --include-bj \
  --epochs 80 \
  --patience 12 \
  --device cuda \
  --single-gpu \
  --max-pre-run-multiple 1.6 \
  --max-sample-to-entry-multiple 1.5 \
  --max-sample-ret20 0.35
```

主要输出：

- `data_cache/latent_uptrend/latent_5x_entry_events.csv`：历史爆量进场后 30-100 日再启动且 200 日 5 倍事件
- `data_cache/latent_uptrend/latent_training_samples.csv`：潜伏窗口训练样本
- `data_cache/latent_uptrend/latent_current_candidates.csv`：当前即将主升候选
- `model_cache/latent_uptrend_tcn.pt`：深度时序模型

### 五年内爆量进场、缩量后主升模型

`volume_contraction_breakout_model.py` 是当前最新策略版本：

- 标签扫描近 5 年历史，额外拉取约 6 年日线，保证早期样本也有足够上下文
- 原始时序输入使用最近 240 个交易日，并叠加多年高低位、长期缩量/缩额、长期收益和波动等压缩特征
- 先找「三倍爆量阳」作为主力进场信号
- 爆量之后要求成交量、成交额进入缩量状态，且股价尚未明显主升
- 默认只筛沪深主板并剔除 ST；爆量缩量后，新增「MACD DIF/DEA 始终在 0 轴上方，后续阳线收盘突破三倍量阳线收盘价」的严格信号
- 未来最多 250 个交易日内，若出现快速拉升或第二次爆量阳，并且主升开始后 20 个交易日内达到 2 倍，则定义为正事件
- 正样本不是主升日，而是主升前仍处于缩量潜伏状态的交易日
- 模型为 PyTorch 时序深度模型，结合长短周期量价结构和爆量后缩量上下文特征

```bash
/home/luochangyu/anaconda3/envs/py310/bin/python volume_contraction_breakout_model.py \
  --output-dir data_cache/volume_contraction_5y \
  --history-dir data_cache/main_uptrend/hist \
  --spot-cache-dir data_cache/main_uptrend \
  --max-stocks 0 \
  --include-bj \
  --epochs 80 \
  --patience 12 \
  --progress-every 400 \
  --print-top 60 \
  --device cuda \
  --main-board-only \
  --strategy-days 1825 \
  --lookback-years 6 \
  --seq-len 240 \
  --max-entry-to-run-days 250
```

主要输出：

- `data_cache/volume_contraction_5y/contraction_entry_run_events.csv`：历史「三倍爆量阳 -> 缩量 -> 主升」事件
- `data_cache/volume_contraction_5y/contraction_training_samples.csv`：主升前缩量潜伏训练样本
- `data_cache/volume_contraction_5y/contraction_current_candidates.csv`：当前可能迎来主升候选
- `data_cache/volume_contraction_5y/contraction_summary.json`：训练指标、参数和样本统计
- `model_cache/volume_contraction_breakout_5y_tcn.pt`：训练后的深度时序模型

### 单股主升模型前端

启动服务后访问 `/uptrend`，可以输入股票代码或名称，刷新最新日线，并用 5 年主升模型输出：

- 当前是否满足爆量后缩量潜伏模型
- 最新量价、均线、成交密集区、支撑压力
- 历史三倍爆量阳和主升确认日
- 反弹/失败路径推演和风控提示

```bash
/home/luochangyu/anaconda3/envs/py310/bin/python app.py
```

页面地址示例：

- `http://localhost:8002/uptrend`

### 六月 Top20 回测前端与公开发布

本项目新增了六月每日 Top20 回测看板，入口如下：

- 本地 FastAPI：`http://localhost:8002/june-backtest`
- GitHub Pages：`https://chyuuuuuuu.github.io/Financial_Quantification/`

生成静态报告：

```bash
/home/luochangyu/anaconda3/envs/py310/bin/python refresh_and_screen_uptrend.py \
  --universe-file data_cache/volume_contraction_screen_20260701_mainboard_entry_close/refresh_status.csv \
  --history-dir data_cache/main_uptrend/hist \
  --output-dir data_cache/refresh_june_backtest \
  --begin 20260601 \
  --end 20260701 \
  --target-date 2026-07-01 \
  --workers 32 \
  --retry 2 \
  --progress-every 500 \
  --eastmoney-only \
  --refresh-only

/home/luochangyu/anaconda3/envs/py310/bin/python june_top20_backtest.py \
  --start 2026-06-01 \
  --end 2026-06-30 \
  --top-n 20 \
  --forward-days 3 \
  --device cpu \
  --fast-rule-only
```

输出文件：

- `static/index.html`：GitHub Pages 首页
- `static/june-backtest.html`：同一份回测页面
- `static/reports/june_top20_backtest.json`：前端读取的数据
- `data_cache/june_top20_backtest/*.csv`：本地分析用明细

公开访问需要在 GitHub 仓库的 `Settings -> Pages` 中选择 `GitHub Actions`。仓库已包含 `.github/workflows/pages-report.yml`，每天 16:40（Asia/Shanghai，交易日）会发布 `static/` 到 GitHub Pages。若要让服务器 A100 每天生成最新结果并推送，服务器上可设置 cron 执行上述刷新和回测命令，然后提交 `static/` 下的静态文件：

```bash
git add static/index.html static/june-backtest.html static/reports/june_top20_backtest.json
git commit -m "Update daily stock report"
git push
```

## 数据来源

- **akshare**：东方财富、同花顺等免费数据接口
- 数据为实时爬取，非交易日可能返回上一交易日数据

## 注意事项

1. 数据仅供学习研究，不构成投资建议
2. 请合理控制请求频率，避免对数据源造成压力
3. 部分接口在非交易时段可能返回空数据或延迟
