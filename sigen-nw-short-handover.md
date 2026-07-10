# sigen-nw-short 超短期光伏预测 - 工作交接说明

更新时间：2026-07-09  
项目仓库：`sigen-nw-short` / GitLab `sigen-plant-ushort-pv-forecast`  
主开发分支：`5station`

## 1. 项目定位

本项目用于多电站实时超短期光伏功率预测，预测未来 4 小时功率，时间步长为 15 分钟，即每次输出未来 16 个预测点。

实时链路为：

1. 拉取电站 PV 历史功率与 Open-Meteo 气象数据。
2. 生成实时训练 / 推理 CSV。
3. 运行 PatchTST、CSFformer 15m、CSFformer 5m、InertiaTrendLGBM 四套模型。
4. 将预测结果写入 Kafka topic `plant_models_prediction`。
5. 下游消费 Kafka 并入库。

默认部署形态为容器长驻，四模型并行调度。镜像不内置权重，运行时需要挂载电站配置和 checkpoint。

## 2. 本轮工作的核心结论

本轮工作的核心不是新加一个模型，而是把实时预测系统从「能跑」推进到「多模型稳定调度 + LGBM 可解释改造 + 统一气象口径 + Kafka 实时输出」。

主要改进集中在三个方面：

1. **InertiaTrendLGBM 是算法改造重点**：重做了预测目标、混合锚点、四源气象、夜间置零、推理后处理和特征安全填充。
2. **PatchTST / CSFformer 主要是工程实时化**：模型结构基本沿用，重点是接入日训、15 分钟推理、Kafka 输出、缺权重容错和 5 分钟 CSF 独立调度。
3. **LGBM 特征体系做了口径修正**：统一 Open-Meteo 四源平均、修正 NWP 与实测辐照对齐方式、明确 Band A / Band B 特征边界，并将训练标签改为相对混合锚的残差。

## 3. 模型清单

| 模型 | 代码 | 权重路径 `checkpoints/realtime/{station_id}/` | Kafka `model_name` | 本轮改动性质 |
| --- | --- | --- | --- | --- |
| PatchTST | `models/patchtst.py` | `PatchTST.pt` | `PatchTST` | 工程接入、实时调度、Kafka 输出 |
| CSFformer 15m | `models/csfformer.py` | `CSFformer.pt` | `CSFformer` | 工程接入、实时调度、Kafka 输出 |
| CSFformer 5m | `models/csfformer.py` | `CSFformer_grid5min.pt` | `CSFformer_5m` | 独立 5 分钟模型、独立 checkpoint、auto 启停 |
| InertiaTrendLGBM | `models/inertia_trend_lgbm.py` | `InertiaTrendLGBM_openmeteo4mean/` | `InertiaTrendLGBM` | 算法重点改造 |

注意：

- 权重不打进镜像，需要挂载 PVC 到 `/app/checkpoints/realtime`。
- `PatchTST` 当前实现是 MLP 展平输入，未做标准 Patch 切分，见 `layers/patchtst/__init__.py`。
- `CSFformer_grid5min.pt` 不存在时，`ENABLE_CSF_5M_SCHEDULER=auto` 会自动不启 5 分钟 CSF 进程。

## 4. 模型分别改进了什么

### 4.1 InertiaTrendLGBM

InertiaTrendLGBM 是本轮最主要的算法改造对象。

#### 4.1.1 预测目标改造

改造前：

- Band A，即 lead 1-4，主要学习 `未来功率 - 当前功率`。
- Band B，即 lead 5-16，学习相对混合锚点的残差。
- Band A 和 Band B 的标签口径不一致，近端容易在下行趋势时残差预测过大，出现负功率或过冲。

改造后：

- Band A / Band B 统一学习：

```text
target_blended_delta = target_power - anchor
```

- 推理时统一还原：

```text
prediction = anchor + predicted_delta
prediction = clip(prediction, 0, capacity)
```

这样训练目标和推理还原口径一致，避免 Band A 仍围绕当前功率硬学差值导致的下跌段过冲。

#### 4.1.2 混合锚点改造

混合锚点：

```text
anchor = alpha * nwp_proxy + (1 - alpha) * current_power
nwp_proxy = GHI / 1000 * capacity * 0.85
```

设计原则：

- lead 1-4：`alpha` 偏小，更信当前功率和短期惯性。
- lead 5-16：`alpha` 加大，更信未来 NWP 辐照代理功率。

这样做的原因是项目当前功率历史只使用最近 4 个 15 分钟点，即约 1 小时。对于 2-4 小时远端预测，继续依赖功率惯性意义有限，未来 GHI、云量、温度和风速等 NWP 特征更重要。

#### 4.1.3 气象源改造

改造前：

- LGBM 可受 Open-Meteo 单源 / 四源开关影响，和深度学习模型的数据口径可能不一致。

改造后：

- LGBM 固定使用 Open-Meteo 四源平均。
- 四源包括 ECMWF、DWD / ICON、GFS、MeteoFrance 等 Open-Meteo 可用源。
- 四源平均后做 15 分钟时间插值。
- LGBM 不再受 `OPENMETEO_FOUR_SOURCES_MEAN` 环境变量影响，该变量主要影响 PatchTST / CSFformer。

对应 checkpoint 目录使用：

```text
InertiaTrendLGBM_openmeteo4mean/
```

旧版 `InertiaTrendLGBM/` 权重与新逻辑不兼容，需要重训。

#### 4.1.4 夜间与负功率处理

训练阶段：

- 当目标时刻 `nwp_ghi < 10 W/m2`，强制 `target_power = 0`。
- 对应残差大致变成 `0 - anchor`，让模型学习夜间回零逻辑。

推理阶段：

- 预测结果先 `clip(0, capacity)`。
- 再结合未来 GHI 做 `force_night_zero`。
- 默认开启夜间置零。

解决的问题：

- 夜间误报功率。
- 辐照很低时残差还原后出现负值。
- 下行趋势中 Band A 预测残差过大导致的异常。

#### 4.1.5 稳定性修正

修复和增强项：

- `nwp_trust` 缺列时安全填充，避免 `fillna` 报错。
- 缺 checkpoint 时 scheduler 只打 WARN 并跳过该模型推理，不把整站刷成 ERROR。
- LGBM 权重目录和旧版隔离，避免误加载旧模型。

### 4.2 PatchTST

PatchTST 本轮主要是工程实时化，不是结构性算法升级。

完成内容：

- 接入实时建表与推理链路。
- 参与 15 分钟墙钟调度。
- 每日按 `TRAIN_AT` 执行日训。
- 推理结果写入 Kafka。
- 缺权重时 WARN 跳过。

当前注意点：

- 代码实现上是全序列展平 MLP，未实现标准 PatchTST 的 patch 切分结构。
- 接手人如果要继续提升 PatchTST，需要先确认是否要替换为真实 patch embedding 版本。

### 4.3 CSFformer 15m

CSFformer 15m 本轮也是工程接入为主。

完成内容：

- 接入实时 15 分钟预测。
- 参与日训和 Kafka 输出。
- 与 PatchTST 使用相同实时建表链路。
- 缺权重容错跳过。

### 4.4 CSFformer 5m

CSFformer 5m 是独立 5 分钟粒度模型。

关键点：

- checkpoint 独立为 `CSFformer_grid5min.pt`。
- 调度由 `ENABLE_CSF_5M_SCHEDULER` 控制。
- 默认 `auto`：没有 5m 权重时不启动 5m 进程。
- Kafka `model_name` 为 `CSFformer_5m`。

风险：

- 5 分钟数据量更大，训练慢。
- 首次部署如果没有权重，建议先关闭 5m 或等待日训完成后再自动启用。

## 5. LGBM 特征体系修正

### 5.1 输入数据口径

| 数据 | 修正 / 约定 |
| --- | --- |
| PV 功率 | `sigen_device.station_statistics_min` -> `filtered_pv_total_power` -> 15 分钟 mean -> `power` |
| 气象源 | Open-Meteo 四源平均，再做 15 分钟插值 |
| `lmd_totalirrad` | 来自 `shortwave_radiation_instant` |
| `Output_ghi` / `nwp_globalirrad` | 同样来自 `shortwave_radiation_instant` |
| `lmd_diffuseirrad` | 来自 `diffuse_radiation_instant`，用于构造云量代理 |
| `actual_irradiance` | 优先使用对齐后的实测 / 近实时辐照，缺失时按现有逻辑回退 |

重点是：LGBM 的训练、推理、标签、锚点全部围绕同一套 NWP GHI 口径，减少「特征来自一套气象、标签锚点来自另一套气象」的问题。

### 5.2 Base 特征

Base 特征由 `InertiaTrendLGBMFeatureBuilder` 构造，主要包含当前时刻 `t0` 的历史功率、时间和近实时气象状态。

主要字段：

| 类别 | 特征 |
| --- | --- |
| 时间 | `minute_of_day_sin`、`minute_of_day_cos`、`doy_sin`、`doy_cos` |
| 当前功率 | `current_power`、`power_cap_ratio` |
| 历史功率 | `power_lag_1`、`power_lag_2`、`power_lag_3`、`power_lag_4` |
| 功率变化 | `power_diff_1`、`power_diff_2` |
| 晴空 / 辐照比例 | `power_spf1`、`power_spf2`、`power_spf3`、`power_spf4` |
| 滚动统计 | `power_trend_std_3`、`power_roll_mean_4`、`power_roll_std_4` |
| NWP 当前状态 | `nwp_ghi_l1`、`nwp_dni_l1`、`nwp_t2m_l1`、`nwp_ws10_l1`、`nwp_tcc_l1` |
| NWP 可信度 | `nwp_trust` |

`power_lag_1~4` 的含义：

```text
power_lag_1 = 当前时刻往前 1 个 15min 点
power_lag_2 = 当前时刻往前 2 个 15min 点
power_lag_3 = 当前时刻往前 3 个 15min 点
power_lag_4 = 当前时刻往前 4 个 15min 点
```

也就是说，当前 LGBM 只显式使用过去 1 小时功率历史。

### 5.3 按 lead 拼接的未来特征

对未来 16 个 lead 分别构造样本。

来自 `t0` 的特征：

- 当前功率。
- 历史 lag。
- 滚动统计。
- `nwp_trust`。
- 当前日历特征。

来自目标时刻 `target_ts` 的特征：

- `nwp_ghi`
- `nwp_dni`
- `nwp_tcc`
- `nwp_t2m`
- `nwp_ws10`
- `target_minute_of_day_sin`
- `target_minute_of_day_cos`
- `target_doy_sin`
- `target_doy_cos`
- `lead_time`

衍生特征：

- `nwp_ghi_delta_l1`
- `nwp_ghi_delta_current`
- `nwp_ghi_ratio_l1`
- `nwp_ghi_ratio_current`

### 5.4 Band A / Band B 特征边界

Band A：lead 1-4，对应未来 15-60 分钟。

```python
BAND_A_FEATURES = [
    "minute_of_day_sin", "minute_of_day_cos", "doy_sin", "doy_cos",
    "target_minute_of_day_sin", "target_minute_of_day_cos", "target_doy_sin", "target_doy_cos",
    "current_power", "power_lag_1", "power_lag_2", "power_lag_3", "power_lag_4",
    "power_diff_1", "power_spf1", "power_spf2", "power_spf3", "power_spf4",
    "power_trend_std_3", "power_roll_mean_4", "power_roll_std_4", "power_cap_ratio",
    "actual_irradiance", "lead_time", "nwp_ghi", "nwp_ghi_delta_l1",
    "nwp_ghi_delta_current", "nwp_t2m", "nwp_ws10", "nwp_trust",
]
```

Band B：lead 5-16，对应未来 75-240 分钟。

```python
BAND_B_FEATURES = [
    "minute_of_day_sin", "minute_of_day_cos", "doy_sin", "doy_cos",
    "target_minute_of_day_sin", "target_minute_of_day_cos", "target_doy_sin", "target_doy_cos",
    "current_power", "power_lag_1", "power_lag_2", "power_lag_3", "power_lag_4",
    "power_diff_1", "power_spf1", "power_spf2", "power_spf3", "power_spf4",
    "power_trend_std_3", "power_diff_2", "power_roll_mean_4", "power_roll_std_4",
    "power_cap_ratio", "actual_irradiance", "lead_time", "nwp_ghi", "nwp_ghi_delta_l1",
    "nwp_ghi_delta_current", "nwp_ghi_ratio_l1", "nwp_ghi_ratio_current",
    "nwp_dni", "nwp_tcc", "nwp_t2m", "nwp_ws10",
]
```

差异：

- Band B 多了 `power_diff_2`。
- Band B 多了 `nwp_ghi_ratio_l1`、`nwp_ghi_ratio_current`。
- Band B 多了 `nwp_dni`、`nwp_tcc`。
- Band A 保留 `nwp_trust`，更强调近端惯性和 NWP 可信度。
- Band B 更强调未来气象，因为 1-4 小时远端对云量和辐照更敏感。

### 5.5 `nwp_trust` 修正

`nwp_trust` 用于衡量近期 NWP GHI 和实测 / 近实时辐照的一致性。

逻辑：

```text
rolling_rmse = 最近约 24 个 15min 点的 NWP GHI 与实测辐照 RMSE
nwp_trust = exp(-rolling_rmse / 180)
```

本轮修正：

- 缺列时安全填充。
- 避免某些站点或某些时间段没有 `nwp_trust` 时训练 / 推理报错。

### 5.6 训练样本修正

训练样本生成规则：

- `target_power` 必须和 `target_ts` 严格对齐。
- 对不齐的样本丢弃。
- `power_lag_1~4` 必须非空。
- 目标时刻 `nwp_ghi < 10` 时，`target_power = 0`。
- 标签统一为 `target_power - anchor`。

这样保证：

- 标签和未来目标时刻一致。
- 近端 / 远端标签口径一致。
- 夜间标签不会把噪声功率当作真实功率学习。

## 6. LGBM 改造前后对比

| 维度 | 改造前 | 改造后 |
| --- | --- | --- |
| Band A 目标 | 未来功率 - 当前功率 | 未来功率 - 混合锚 |
| Band B 目标 | 未来功率 - 混合锚 | 未来功率 - 混合锚 |
| 训练 / 推理对称性 | Band A / B 不完全一致 | 统一 `anchor + delta` |
| 气象源 | 可能随开关走单源 / 四源 | LGBM 固定四源平均 |
| 远端预测 | 仍较依赖当前功率 | 更依赖未来 NWP 和混合锚 |
| 夜间处理 | 主要靠后处理 | 标签置零 + 推理置零 + clip |
| 负功率风险 | 下行趋势时较明显 | clip 与锚点残差共同缓解 |
| 权重目录 | 旧 `InertiaTrendLGBM/` | 新 `InertiaTrendLGBM_openmeteo4mean/` |

## 7. 实时数据流

### 7.1 实时建表入口

入口：

```text
scripts/run_daily_realtime_openmeteo_train_infer.py
```

核心函数：

```text
build_train_test_csv_realtime_openmeteo
```

PV 数据：

```text
station_statistics_min -> filtered_pv_total_power -> 15min mean
```

气象数据：

```text
Open-Meteo -> 四源平均 / 单源 -> 15min 插值 -> 模型输入
```

LGBM 固定四源平均，PatchTST / CSFformer 可受配置开关影响。

### 7.2 调度入口

脚本：

```text
scripts/scheduled_realtime_train_infer.py
```

调度逻辑：

- 上海时区墙钟 `:00`、`:15`、`:30`、`:45` 进入新槽。
- 新槽触发各站 `infer_once`。
- 推理完成后写 Kafka。
- 每日 `TRAIN_AT` 后，每站每模型最多训练一次。
- 默认训练窗口：90 天。
- 默认训练 epoch：10。
- 顺序为先推理，再日训，避免训练阻塞准点推理。

缺权重策略：

- scheduler 下 infer 缺权重只 WARN 跳过。
- 不把整个站点或整个调度进程刷成 ERROR。

## 8. Docker / Jenkins / K8s

### 8.1 Docker

关键文件：

```text
Dockerfile
docker/entrypoint.sh
docker/entrypoint-quad-scheduler.sh
docker/entrypoint-triple-scheduler.sh
docker/verify_models_import.py
requirements-docker.txt
```

说明：

- 默认入口为四模型并行：PatchTST + CSFformer 15m + CSFformer 5m + InertiaTrendLGBM。
- 可通过 `SCHEDULER_ENTRYPOINT=/app/docker/entrypoint-triple-scheduler.sh` 启动三模型模式，即不跑 LGBM。
- Dockerfile 使用 BuildKit pip / apt 缓存。
- 构建期会跑模型 import 校验。

### 8.2 Jenkins 构建示例

```bash
export DOCKER_BUILDKIT=1
docker build -t code-oss.sigenpower.com:8090/sigen-ai2/sigen-plant-ushort-pv-forecast:${BUILD_NUMBER} .
docker push code-oss.sigenpower.com:8090/sigen-ai2/sigen-plant-ushort-pv-forecast:${BUILD_NUMBER}
```

### 8.3 运行必挂

```text
plants_config.yaml        -> /app/plants_config.yaml
checkpoints/realtime      -> /app/checkpoints/realtime
results                   -> /app/results
```

### 8.4 K8s

示例文件：

```text
k8s/data-platform/deployment.yaml
```

关键环境变量：

- `TRAIN_DAYS_BACK`
- `EPOCHS`
- `TRAIN_AT`
- `KAFKA_MODEL_NAME_LGBM`
- `ENABLE_CSF_5M_SCHEDULER`
- `SCHEDULER_ENTRYPOINT`

## 9. 本地常用命令

### 9.1 五站一次性训练 LGBM

```bash
python -u scripts/scheduled_realtime_train_infer.py \
  --mode train \
  --config plants_config.yaml \
  --station_id all \
  --model InertiaTrendLGBM \
  --train_days_back 90 \
  --resample_minutes 15
```

### 9.2 常驻调度 LGBM

```bash
python -u scripts/scheduled_realtime_train_infer.py \
  --mode scheduler \
  --config plants_config.yaml \
  --station_id all \
  --model InertiaTrendLGBM \
  --train_days_back 90 \
  --kafka_bootstrap_servers <host:9092> \
  --kafka_topic plant_models_prediction \
  --kafka_model_name InertiaTrendLGBM \
  --force_night_zero true
```

### 9.3 批量训练三套深度学习模型

```bash
python -u scripts/train_all_realtime_models.py
```

训练顺序：

```text
PatchTST -> CSFformer 15m -> CSFformer 5m
```

## 10. 测试与调研材料摘要

### 10.1 前期调研方向

已调研 / 对比的方向包括：

- 超短期光伏预测。
- MCloudNet。
- LSTM + Transformer。
- CSFformer。
- 注意力残差结构。
- PVOD v1.0 西北地区数据。
- 卫星云图数据。

卫星云图数据源：

- 风云卫星 / 中国气象数据网：`https://www.nmic.cn/data/online/t/3`
- NOAA 卫星系列。

当前判断：

- 卫星云图公开数据可获得，但分辨率偏低。
- 对当前 15 分钟 / 4 小时站点级预测，短期更现实的收益仍在 NWP 对齐、实时功率惯性、残差锚点和模型工程稳定性。

### 10.2 多站点测试结果摘要

#### 犀牛日用品站 `82026030500002`

| 模型 | 气象源 | ACC_4h(%) | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| CSFformer | `ecmwf_ifs025` | 68.58 | 24782.63 | 81.56 |
| DAG | `gfs_global` | 72.60 | 23417.70 | 82.13 |
| PatchTST | `icon_global` | 69.51 | 25698.24 | 83.66 |
| CSFformer | `icon_global` | 71.42 | 21255.09 | 83.74 |
| DAG | `meteofrance_arpege_world` | 71.90 | 22365.73 | 85.56 |

#### 新岚环境站 `82026021000012`

装机容量：1077.10 kW  
经纬度：北纬 22.78，东经 114.41

| 模型 | 气象源 | ACC_4h(%) | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| CSFformer | `ecmwf_ifs025` | 67.40 | 8077 | 48.27 |
| PatchTST | `gfs_global` | 69.87 | 7594 | 45.91 |
| PatchTST | `ecmwf_ifs025` | 69.34 | 8108 | 47.98 |
| PatchTST | `icon_global` | 69.33 | 8571 | 48.66 |
| DAG | `meteofrance_arpege_world` | 67.32 | 8507 | 47.25 |

#### 深镭纺织站 `82025121300002`

装机容量：2300.00 kW  
经纬度：北纬 24.68，东经 113.51

| 模型 | 气象源 | ACC_4h(%) | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| CSFformer | `ecmwf_ifs025` | 72.03 | 34943 | 95.46 |
| DAG | `gfs_global` | 68.52 | 36256 | 101 |
| PatchTST | `icon_global` | 72.58 | 31917 | 93 |
| CSFformer | `icon_global` | 68.99 | 35686 | 102 |
| DAG | `ecmwf_ifs025` | 71.44 | 33801 | 94 |

#### 韶关乳源鑫中胜汽车配件 1 站 `82026022800001`

装机容量：520.00 kW  
经纬度：北纬 24.77，东经 113.30

| 模型 | 气象源 | ACC_4h(%) | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| CSFformer | `ecmwf_ifs025` | 77.13 | 1896 | 18.73 |
| DAG | `ecmwf_ifs025` | 77.48 | 1363 | 18.73 |
| PatchTST | `gem_global` | 77.02 | 1375 | 19.21 |
| CSFformer | `gfs_global` | 77.07 | 1404 | 19.18 |
| DAG | `gem_global` | 76.98 | 1447 | 19.29 |

#### 韶关乳源鑫中胜汽车配件 2 站 `82026022800002`

| 模型 | 气象源 | ACC_4h(%) | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| CSFformer | `ecmwf_ifs025` | 78.70 | 1526 | 19.66 |
| DAG | `ecmwf_ifs025` | 78.89 | 1570 | 19.78 |
| PatchTST | `gem_global` | 78.63 | 1413 | 19.38 |
| CSFformer | `gem_global` | 79.05 | 1498 | 19.41 |
| PatchTST | `ecmwf_ifs025` | 78.98 | 1447 | 19.29 |

### 10.3 测试结论

- 大部分测试结果接近或超过 ACC_4h 70%。
- 不同站点最优模型不一致，说明模型和气象源存在明显站点适配差异。
- DAG 在部分站点和气象源上表现较好，但并非全站稳定最优。
- PatchTST、CSFformer、DAG 对不同气象源敏感。
- 当前工程化交付更适合保留多模型并行输出，让下游或后续融合模块选择更优模型。

## 11. 已知问题

| 问题 | 说明 | 建议 |
| --- | --- | --- |
| 首次部署无权重 | scheduler 会 WARN 跳过推理 | 预挂载 checkpoint，或等日训完成 |
| CSF 5m 训练慢 | 5 分钟粒度数据量大 | 可先设置 `ENABLE_CSF_5M_SCHEDULER=false` |
| LGBM 历史功率短 | 只显式使用最近 1 小时功率 | 远端主要依赖 NWP，后续可加长 lag |
| LGBM 旧权重不兼容 | 新标签和锚点已变化 | 必须重训 |
| 日训失败不重试 | 为避免打爆 DB / API，当日不重复训练 | 次日自动再训，或人工触发 |
| 卫星云图分辨率低 | 对站点级 15m 预测收益不确定 | 暂不作为主线依赖 |

## 12. 接手后验证清单

1. 确认 `plants_config.yaml` 已挂载到 `/app/plants_config.yaml`。
2. 确认 `/app/checkpoints/realtime/{station_id}/` 下四模型权重目录齐全。
3. 特别确认是否存在 `CSFformer_grid5min.pt`。
4. 启动容器后确认四个 scheduler 进程是否正常。
5. 等待一个 15 分钟槽，确认 Kafka 中是否出现 4 路 `model_name`。
6. 对 LGBM 使用新逻辑重训一轮，确认权重生成在 `InertiaTrendLGBM_openmeteo4mean/`。
7. 抽查白天、傍晚、夜间三个时间段预测，确认夜间置零和 clip 生效。
8. 对比 LGBM Band A 下行趋势样本，确认负功率和过冲是否缓解。
9. 对比 2-4 小时远端预测，重点看 NWP 变化明显日的误差。
10. 检查 Kafka 下游入库字段和 `model_name` 是否与消费端约定一致。

## 13. 关键文件索引

| 文件 | 说明 |
| --- | --- |
| `scripts/scheduled_realtime_train_infer.py` | 调度 / 训练 / 推理主入口 |
| `scripts/run_daily_realtime_openmeteo_train_infer.py` | 实时 CSV 建表 |
| `scripts/train_all_realtime_models.py` | 批量训练 PatchTST / CSFformer |
| `models/inertia_trend_lgbm.py` | LGBM 特征、双 Band 模型、锚点和残差逻辑 |
| `models/patchtst.py` | PatchTST 模型 |
| `models/csfformer.py` | CSFformer 模型 |
| `utils/openmeteo_nwp.py` | Open-Meteo 四源平均和列映射 |
| `utils/night_zero.py` | 夜间置零逻辑 |
| `docker/entrypoint-quad-scheduler.sh` | 四模型容器入口 |
| `Dockerfile` | 镜像构建 |
| `plants_config.yaml` | 电站配置 |
| `k8s/data-platform/deployment.yaml` | K8s 部署示例 |

## 14. 一句话交接

已完成五站超短期光伏预测的多模型实时调度、Kafka 输出、Docker 四进程入口、缺权重容错，以及 InertiaTrendLGBM 的四源气象、混合锚残差、夜间置零和特征安全填充改造。接手重点是先重训 LGBM 新权重，再验证 15 分钟槽 Kafka 输出、夜间置零、CSF 5m 权重是否齐全，以及不同站点的最优模型差异。
