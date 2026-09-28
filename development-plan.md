# DeepCAVE-X 插件开发计划（开发指南 v1.1）

> 基座：DeepCAVE v1.4.1（`pip install -e .`，editable 模式，改源码即时生效）
> 目标环境：Linux 服务器 / 云容器（Ubuntu 22.04，Python 3.9 或 3.10）
> v1.1 说明：按最终确定的功能范围校准；删除多优化器对比、交互效应、一致性矩阵、搜索空间建议等超出范围的条目；补齐数据口径、算法伪代码、缓存键、输入输出契约与测试细节。

---

## 0. 目标与范围

### 0.1 一句话目标

在 DeepCAVE 上新增 **4 个可交互插件**，全部遵循官方插件机制
（evaluator 算法层 + plugin 展示层 + `Config.PLUGINS` 注册），既有功能全部保留；
算法层在 Web 与 API 双入口下同步可用。

### 0.2 本次开发范围（最终版，4 项）

| #    | 功能                   | 一句话                                                       | 基类                            |
| ---- | ---------------------- | ------------------------------------------------------------ | ------------------------------- |
| P1   | **加权 Sobol 重要性**  | 在 fANOVA/LPI 之外加入“以经验采样分布替换均匀分布”的一阶重要性估计；只输出每个超参一个数（默认 S1，ST 为可选开关）；**不做交叉/成对重要性** | StaticPlugin                    |
| P2   | **采样密度展示**       | 选 1 个超参画一维边际密度，选 2 个超参画二维联合密度         | DynamicPlugin                   |
| P3   | **Footprint 时序回放** | 在 Configuration Footprint 点云上加入 trial 序号滑块，回放配置探索过程；MDS 只算一次、坐标冻结 | StaticPlugin（滑块作为 filter） |
| P4   | **瓶颈检测**           | 在 incumbent 曲线上以阴影带标注停滞区间，附 banner 提示；对运行中的 run 支持轻量实时刷新 | DynamicPlugin                   |

### 0.3 明确不做（写进报告 Future Work）

1. **Sobol 交叉/成对重要性**：P1 只输出单参数口径（S1，ST 可选），不做二阶/参数对图。
2. **多优化器专用对比视图**：DeepCAVE 已支持多 run 选择与 Group Analysis，
   现有 Cost over Time 等插件可叠加多条曲线；本项目不再新增专用对比插件。
   P3/P4 的 run selection 与 Group 兼容，但本期的交互与验收按“单 run / 单 Group”设计。
3. **搜索空间收缩建议**：P2 只做密度展示，不输出建议区间、ForbiddenClause 或空间导出。
4. **独立 Regret 曲线插件**：不作为交付功能；仅在离线验证停滞检测器时，
   用“事后全局最优”构造 regret 平台作为弱 ground truth。

> 范围原则：先把“单 run 诊断四件套”做成可运行、可测试、可演示的完整闭环；
> 研究向能力（多 run 对齐、成对交互、决策输出）全部后置。

### 0.4 开发环境与已知问题（M1 必须先解决）

1. **DeepCAVE 版本**：v1.4.1，`python_requires=">=3.9, <3.11"`；
   建议使用 conda 环境 + `pip install -e .` 从源码安装，方便改插件并即时生效。

2. **Redis + worker**：Web 模式的静态插件（P1/P3）依赖 RQ + Redis；
   开发时先跑 `deepcave --open --n_workers=1`，确认 worker 在线。

3. **history 加载 bug（必须打补丁）**：v1.4.1 的 `deepcave/runs/run.py`
   在读取 `Recorder` 新写入的 9 字段 history 时会插入错误字段，报
   `Trial.__init__() takes 10 positional arguments but 11 were given`。
   修复方式（备份原文件后改）：

   ```python
   # deepcave/runs/run.py, in Run.load()
   # 旧代码：if len(obj) != 11: obj.insert(6, 0.0)
   # 正确兼容：旧 8 字段格式缺的是 cpu_time，应插到 index 4；新 9 字段格式不动
   if len(obj) == 8:
       obj.insert(4, 0.0)
   ```

4. **macOS 兼容性**：DeepCAVE 官方仅正式支持 Linux。本机实测 Mac 上
   原生 `Importances`（fANOVA/`pyrfr`）会报
   `swig_runtime_data5.SwigPyObject has no attribute 'mean'`。
   因此：Web/插件联调与验收在 Linux 容器/服务器上做；
   P1 的代理模型优先用 sklearn `RandomForestRegressor`（见 §2），
   避免把开发阻塞在 `pyrfr` 上。

5. **依赖范围**：P1/P2 只使用 `numpy / scipy / scikit-learn / plotly`
   （均为 DeepCAVE 现有依赖），不需要引入 SALib；若后续改用 SALib，
   必须同步更新 `requirements.txt` 与容器镜像。

---

## 1. 总体架构与代码落位

### 1.1 三条硬约束（v1.4.1 源码行为）

1. **双入口同包**：`deepcave/__init__.py` 判断是否以 `server.py` / `worker.py` 执行；
   否则 `_api_mode=True`，`@interactive` 装饰的 GUI 方法空转。
   → 新增 **evaluator 层**（纯 Python 算法）Web/API 都能用；plugin 层的回调只在 Web 模式执行。
2. **重活走 RQ，回调别重算**：`StaticPlugin.process` 通过 `_process` 投递到 Redis Queue；
   `DynamicPlugin` 的回调直接执行，必须控制在数百毫秒内
   （`config.REFRESH_RATE = 500`，`layouts/main.py` 的 `dcc.Interval` 每 500ms 触发）。
3. **注册生效点唯一**：`config.py` 的 `Config.PLUGINS` property；
   新插件被 import 并实例化进对应类目列表后，才会出现在侧边栏与
   `/plugins/<id>` 路由。

### 1.2 文件规划

```text
deepcave/
├── config.py                                   # 改：PLUGINS 中 import + 实例化 4 个插件
├── evaluators/
│   ├── weighted_sobol.py       # 新增 P1：算法层（纯 numpy/sklearn，无 pyrfr 依赖）
│   ├── sampling_density.py     # 新增 P2：1D/2D 密度计算
│   ├── footprint_replay.py     # 新增 P3：复用 Footprint 的 MDS/距离，追加 first_seen 与 incumbent 前缀
│   └── stagnation.py           # 新增 P4：停滞检测器（离线全量 + 流式增量）
└── plugins/
    ├── hyperparameter/
    │   ├── weighted_sobol.py    # P1 展示层
    │   └── sampling_density.py  # P2 展示层
    ├── summary/
    │   └── footprint_replay.py   # P3 展示层
    └── objective/
        └── bottleneck_diagnosis.py  # P4 展示层
```

### 1.3 插件生命周期（每个插件必须实现的 5 类方法）

以 `plugins/summary/footprint.py` 为模板。准确签名如下：

| 方法                     | 签名（含 self 省略说明）                                     | 职责                                          |
| ------------------------ | ------------------------------------------------------------ | --------------------------------------------- |
| `get_input_layout`       | `@staticmethod def get_input_layout(register)`               | 声明输入控件；会触发计算                      |
| `get_filter_layout`      | `@staticmethod def get_filter_layout(register)`              | 声明过滤器；**不触发计算**，适合放滑块        |
| `load_inputs`            | `def load_inputs(self)`                                      | 首次加载时的默认选项                          |
| `load_dependency_inputs` | `def load_dependency_inputs(self, run, previous_inputs, inputs)` | run / 上游输入变化后回填 options/value        |
| `process`                | `@staticmethod def process(run, inputs)`（StaticPlugin）     | 算法计算；返回值必须 JSON 可序列化            |
| `get_output_layout`      | `@staticmethod def get_output_layout(register)`              | 声明输出组件                                  |
| `load_outputs`           | `@staticmethod def load_outputs(run, inputs, outputs)`       | 把原始 outputs 变成 Plotly figure / Dash 组件 |

关键细节：

- `register("key", ["value", "options"], type=int)` 会返回带插件前缀的内部 id；
  `inputs["key"]["value"]` 才是真实值。
- 过滤器的值是字符串/布尔值，`process` 前会被 `_clean_inputs` 清洗；
  在 `load_outputs` 中拿到的是清洗后的 `inputs`。
- **P3 的滑块必须放在 filter block**：这样拖动滑块只重跑 `load_outputs`，
  不会重新走 RQ，也不会重算 MDS。
- StaticPlugin 的缓存键在计算时会移除 filter 输入
  （`_dict_as_key(..., remove_filters=True)`），所以滑块变化不会命中新的计算任务，
  只会用已有 outputs 重新渲染；这正是 P3 想要的语义。

### 1.4 数据取用统一约定（四个插件共用）

#### 1.4.1 编码数据

```python
from deepcave.runs import Status

df = run.get_encoded_data(
    objectives=objective,          # Objective 或 [Objective]
    budget=budget_value,           # 真实 budget 值（由 budget_id 解析得到）
    statuses=[Status.SUCCESS],     # 重要性/密度只统计成功 trial；需要看失败搜索行为时另行说明
    include_config_ids=True,       # 需要按 config 聚合时必开
)
```

返回 DataFrame 的列为：`CONFIG_ID`（若开启）、各超参编码列、目标列（一列或多列），
可选 `COMBINED_COST`。

编码空间的真实语义（实现前必须知道）：

- `run.encode_config()` / `get_encoded_data()` 返回的数值列已经是
  ConfigSpace 的**归一化向量**：连续/整数列在 [0,1]（log 参数在 log 尺度上归一化），
  类别列是 `类别下标 / (类别数-1)`，Constant 列固定为 1.0；
- 因此“在编码空间做均匀采样”等价于 ConfigSpace 先验分布采样，
  不需要再手动 min-max 归一化；
- 反向显示原始值时使用 `hp.to_value(encoded)`（如把 alpha 的 0.5 映射回 0.1），
  正向编码使用 `hp.to_vector(value)`；类别轴直接用 `hp.choices` 的标签；
- `specific` 参数选择：P1/P2 默认 `specific=False`（单位立方体，类别列已缩放）；
  若搜索空间含条件参数，v1 改用 `specific=True`，直接使用 DeepCAVE 已完成的
  inactive 维度插补，并在返回值的 `notes` 中标注“条件维度为近似处理”；
  两条路径不要混用。

#### 1.4.2 多 seed 折叠（重要修正）

`get_encoded_data(seed=None)` **不会跨 seed 平均**，它会为同一 config 的每个 seed
各生成一行。直接使用会导致多 seed 权重翻倍。

统一做法二选一：

```python
# 方案 A：按 config 聚合成一行（推荐）
df = run.get_encoded_data(..., include_config_ids=True)
df = df.groupby("CONFIG_ID", as_index=False).mean(numeric_only=True)

# 方案 B：显式使用跨 seed 均值 API
avg, std = run.get_avg_costs(config_id, budget=budget_value)
```

在 P1/P2 的实现里必须显式写这一步，并在文档/论文中说明折叠规则。

#### 1.4.3 budget 语义

- UI 下拉里的 `budget_id` 是**索引**，不是真实 budget；
  `process` 中必须解析：`budget_value = run.get_budget(budget_id)`。
- 本期 P1/P2/P3/P4 默认使用**最高预算**（`run.get_highest_budget()`）；
  如需多预算对比，后续版本再扩展。
- `COMBINED_BUDGET` 的语义沿用 DeepCAVE：取每个 config 实际评估过的最高预算 trial。

#### 1.4.4 目标方向与代价

DeepCAVE 支持 maximize / minimize。做距离、停滞、regret、Sobol 代理训练前，
统一转成“越小越好”的 cost：

```python
cost = -value if objective.optimize == "upper" else value
```

#### 1.4.5 条件超参数与常量

- 条件超参数（conditional HP）在非活跃时值为 NaN；P1/P2 必须先处理：
  - **v1 完整支持不含条件子句的搜索空间**；
  - 若存在条件参数：优先 `specific=True`（DeepCAVE 已用 −1 / 类别数等哨兵值插补
    inactive 维度），或直接排除条件维度并在 `notes` 中说明；
  - 禁止把 `specific=False` 下的 NaN 直接喂给 sklearn RF；
- Constant 超参数不参与重要性/密度分析，直接跳过，并在返回值中标注排除项。

#### 1.4.6 JSON 序列化契约

`process` 的返回值只能包含 Python 原生类型：
`int / float / str / bool / list / dict / None`；
所有 `numpy` 标量用 `float()` / `int()` 转换，`NaN` 用 `None` 或省略。
建议每个 evaluator 提供一个 `_to_jsonable` 工具函数统一处理。

---

## 2. P1 加权 Sobol 重要性

### 2.1 用户故事与界面

在 Hyperparameter Analysis 类目下新增 **Weighted Sobol** 页签：

输入区：

| key                    | 控件              | 说明                                      |
| ---------------------- | ----------------- | ----------------------------------------- |
| `objective_id`         | Select            | 目标                                      |
| `budget_id`            | Select            | 预算                                      |
| `seed`                 | Select/Input      | 可选；默认合并全部 seed（按 §1.4.2 折叠） |
| `n_samples`            | Input（高级折叠） | 默认 20000                                |
| `n_trees`              | Input（高级折叠） | 默认 100                                  |
| `show_uniform_control` | Select            | 默认 True：叠加均匀 Sobol 灰条            |
| `show_total_effect`    | Select            | 默认 False：显示 ST^w（仍是单参数口径）   |

输出：

- 每个超参一根柱，按加权 S1 降序；
- 可选叠加均匀测度 S1 作为对照；
- 可选数值表：`hp_name / S1_weighted / S1_uniform / ST_weighted / ST_uniform`。

### 2.2 算法（可直接照此实现）

**输入数据**：

```python
X_df = run.get_encoded_data(objective, budget_value,
                            statuses=[Status.SUCCESS],
                            include_config_ids=True)
X_df = X_df.groupby("CONFIG_ID", as_index=False).mean(numeric_only=True)
hp_names = list(run.configspace.keys())
X = X_df[hp_names].to_numpy(dtype=float)
y = X_df[objective.name].to_numpy(dtype=float)
# y 统一转成 lower-is-better（若 objective.optimize == "upper"，取负）
```

若 §1.4.5 判定为含条件子句的空间：改用 `specific=True` 取数（使用 DeepCAVE 的
inactive 插补）或删除条件维度；**禁止把 NaN 直接传给 sklearn**。

**代理模型**：使用 sklearn 代理，避免 Mac 上 pyrfr 问题：

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3,
                           random_state=seed, n_jobs=-1).fit(X, y)
f_hat = rf.predict
```

**采样器（必须按列类型区分）**：

```python
def sample_weighted(m, rng):
    """经验边际乘积采样；连续列 bootstrap + 抖动，类别/整数列 bootstrap。"""
    out = np.empty((m, len(hp_names)))
    for j, hp in enumerate(configspace.values()):
        if is_categorical(hp):
            # 只在观测到的类别中按频率抽样，不叠加高斯噪声
            out[:, j] = rng.choice(X[:, j], size=m, replace=True)
        else:
            idx = rng.integers(0, len(X), size=m)
            bw = max(1.06 * np.std(X[:, j]) * len(X) ** (-0.2), 1e-9)
            out[:, j] = np.clip(X[idx, j] + rng.normal(0, bw, m), lo[j], hi[j])
    return out

def sample_uniform(m, rng):
    """均匀对照：连续列在 [lo,hi] 均匀；类别列等概率取观测类别。"""
    ...
```

若使用 `specific=True` 产生了 inactive 哨兵值（例如 −1），这些列应按
“观察值 bootstrap”处理，不参与连续列的抖动与 clip。

**Saltelli 估计器（修正版）**：记 `A` 为基线矩阵，`B` 为独立矩阵，
`AB_i` 表示“把 A 的第 i 列替换为 B 的第 i 列”。

```python
A = sampler(N, rng)
B = sampler(N, rng)
yA, yB = f_hat(A), f_hat(B)
V = np.var(np.concatenate([yA, yB]), ddof=1)
V = max(V, 1e-12)          # variance floor，避免除零

S1 = np.zeros(d)
ST = np.zeros(d)
for i in range(d):
    AB = A.copy()
    AB[:, i] = B[:, i]
    yAB = f_hat(AB)
    S1[i] = np.mean(yB * (yAB - yA)) / V
    ST[i] = np.mean((yA - yAB) ** 2) / (2 * V)

# 截断负值只为展示；同时返回原始负值用于诊断，不要静默丢弃
S1_display = np.clip(S1, 0, None)
ST_display = np.clip(ST, 0, None)
```

实现注意：

- 上述估计量中的 `AB` 必须是 “A 的第 i 列被 B 替换”，不要写反；
- 若 `V` 极小（目标在观测区近似常数），直接返回错误提示，不要输出全 0 柱子；
- `S1` 的负值是估计噪声，保留原始值并在表格中标注，不要假装 Sobol 一定非负；
- 加权版与均匀版必须使用同一 `rf`、同一 `N`、同一 `rng` 种子，仅采样器不同；
- 不对 S1/ST 做强制归一化（它们本来就不满足“求和恰好为 1”）。

### 2.3 为什么是 StaticPlugin

代理拟合 + `N(d+2)` 次 RF predict 是秒级到十秒级开销，必须走 RQ worker，
不能放在 500ms 回调里。

### 2.4 `process` 返回值 schema

```python
{
    "hp_names": ["alpha", "learning_rate_init", ...],
    "s1_weighted": [0.41, 0.22, ...],
    "s1_uniform": [0.37, 0.25, ...],
    "st_weighted": [0.55, 0.30, ...],   # show_total_effect 关闭时可省略
    "st_uniform": [...],
    "n_configs": 120,
    "budget": 100.0,
    "seed_mode": "folded" | "seed=0",
    "notes": ["excluded constant hp: ...", "conditional hp: imputed"],
}
```

### 2.5 实现步骤 Checklist

1. `evaluators/weighted_sobol.py`：
   `calculate(run, objective, budget, n_samples, n_trees, seed, sampler_kind) -> dict`；
   内部完成取数、多 seed 折叠、方差地板、负值诊断。
2. `plugins/hyperparameter/weighted_sobol.py`：
   `id="weighted_sobol"`、`name="Weighted Sobol"`；
   五类方法齐全；输出条形图 + 表格。
3. `config.py` 的 `"Hyperparameter Analysis"` 追加 `WeightedSobol()`。
4. 验收：
   - 用构造函数（如 `y = x1 + 2*x2 + 噪声`）验证排序正确；
   - 用仓库示例 run 对比均匀/加权两列，记录差异与解释；
   - 不把“加权后必然下降”当作验收标准，差异方向是研究结果。

### 2.6 风险与边界

- 乘积边际测度假设忽略超参相关；相关强的搜索空间下，加权 Sobol 的解释需谨慎；
- KDE 带宽/分位数是超参，N < 1e4 时估计方差大；实现需在 UI 提示；
- 条件超参数的插补是近似，需在返回值 notes 和论文 limitation 中写明。

---

## 3. P2 采样密度展示

### 3.1 用户故事与界面

在 Hyperparameter Analysis 类目下新增 **Sampling Density**：

- `mode`：Single（1D）/ Pair（2D）；
- 1D：目标超参下拉 → 密度曲线 + 均匀基线 + rug + incumbent 标记；
- 2D：两个超参下拉 → 联合密度热图 + 已评估点 + incumbent 星标；
- log 参数在数值轴上按对数刻度显示，计算仍在编码空间。

输入 keys：

`objective_id`、`budget_id`、`mode`、`hp1`、`hp2`（pair 时启用）、
`show_uniform`、`show_rug`、`show_points`。

### 3.2 算法要点（性能修正）

取数与 P1 相同（成功 trial、按 CONFIG_ID 折叠多 seed）。

1D：

- 连续参数：`scipy.stats.gaussian_kde` 或直方图平滑；网格默认 200 点；
- 类别参数：直接画 frequency bar，不套 KDE；
- 均匀基线：连续为 `1/(hi-lo)`，类别为 `1/类别数`；
- incumbent 值：`run.get_incumbent(objective, budget)`，映射到同一坐标轴。

2D：

- **默认使用 `np.histogram2d`**（或 `numpy` 计数、Plotly heatmap），
  网格默认 ≤ 80×80；
- 仅当两维都是连续且 n 较大时才考虑二维 KDE；
- 不要对 256×256 网格做二维 KDE：n=10⁴ 时是 ~6.5×10⁸ 次核计算，
  无法满足 500ms 回调要求；
- 类别/整数维度用分箱计数，边界要对齐实际取值。

### 3.3 基类选择与逃生舱

默认 `DynamicPlugin`：

- 1D 直接计算；
- 2D 用 histogram，并在 `process` 内做耗时保护；
- 若某 run 的 n 很大，切换为 StaticPlugin 或把网格降到 40×40；
- 计算前检查 `run.hash` 变化，避免对未变化的 run 重复计算。

### 3.4 `process` 返回值 schema

1D：

```python
{"mode": "single", "hp": "alpha", "grid": [...], "pdf": [...],
 "uniform": [...], "rug": [...], "incumbent": 0.03, "is_categorical": False}
```

2D：

```python
{"mode": "pair", "hp1": "alpha", "hp2": "beta",
 "x_edges": [...], "y_edges": [...], "counts": [[...], ...],
 "points": [[x, y], ...], "incumbent": [x, y]}
```

### 3.5 实现步骤 Checklist

1. evaluator：`marginal(...)`、`joint(...)` 两个纯函数 + 类型判断工具；
2. plugin：五类方法；`load_dependency_inputs` 根据 `mode` 控制 hp 下拉；
3. `config.py` 注册进 `"Hyperparameter Analysis"`；
4. 验收：均匀采样近似平线；明显偏斜的 run 能看出峰；2D 图在 500ms 内返回；
5. 不实现任何区间建议 / ForbiddenClause 输出。

---

## 4. P3 Footprint 时序回放

### 4.1 用户故事与界面

Summary 类目下新增 **Footprint Replay**：

- 图：MDS 2D 点云 + 性能热图；
- trial 滑块（0 → 当前最大 trial）；t 时刻显示已评估点、incumbent 前缀折线与当前 incumbent；
- 未评估点在早期可淡灰显示（可开关）；
- 拖动滑块不重算 MDS，只做数组切片与重绘。

### 4.2 关键设计决策（v1.1 修正）

**P3 必须是 StaticPlugin，滑块作为 filter。** 理由：

- MDS / 热图是重计算，必须走 RQ；
- 滑块如果放在 input block，每次拖动都会触发 `process` → 新计算任务；
- 放在 filter block 后，拖动只触发 `load_outputs`，从缓存 outputs 中切片即可。

具体流程：

1. `get_input_layout`：`objective_id`、`budget_id`、`details`（触发计算）；
2. `get_filter_layout`：`trial`（Slider，min/max 由 `load_dependency_inputs` 设置）、
   `show_unvisited`；
3. `process`：
   - 调用 `Footprint(run)` evaluator 拿 MDS 坐标、热图、四类点；
   - 追加 `first_seen`：遍历 `run.history`，按 `end_time` 排序，记录每个
     `config_id` 第一次出现的 history id；
   - 追加 `incumbent_prefix`：按时间顺序遍历 trial，维护当前最优（按目标方向），
     记录 `(trial_id, config_id, cost)`；
   - 返回 JSON 安全的点云 + 序列；
4. `load_outputs`：根据 `inputs["trial"]` 切分点云，生成 figure。

缓存：

- 静态插件本身由 DeepCAVE 的 `run.hash` 缓存机制管理；
- 若再加进程内缓存，键必须包含
  `(run.id, run.hash, objective_id, budget_id, details)`；
- **不要**用单独 `run.id` 做键：`run.id` 只反映路径，不会随新 trial 改变。

### 4.3 数据细节

- MDS 复用 `evaluators/footprint.py` 的距离与 `MDS(n_components=2,
  dissimilarity="precomputed", random_state=0)`；坐标一次算完并冻结；
- 4000 config cap 沿用现有 footprint 行为；
- `first_seen` 基于完整 `run.history`，不能只从 `get_trajectory()` 的
  improvement ids 推导；
- `trial` 滑块最大值来自“该 run 在目标 budget 下的有效 trial 数”；
  运行中 run 新 trial 到来时，需要重新触发一次 process 或手动刷新。

### 4.4 `process` 返回值 schema

```python
{
    "coords": [[x, y], ...],          # MDS 坐标（configs/borders/supports 合并或分组）
    "point_meta": [{"category": "configs", "config_id": 3, "first_seen": 12}, ...],
    "surface": {"x": [...], "y": [...], "z": [[...]]},
    "incumbent_prefix": [{"trial_id": 0, "config_id": 2, "cost": 0.8}, ...],
    "t_max": 120,
    "notes": [...],
}
```

### 4.5 实现步骤 Checklist

1. `evaluators/footprint_replay.py`：`build_cloud(...)` + `subset(cloud, t)`；
2. `plugins/summary/footprint_replay.py`：StaticPlugin 五类方法 + filter 滑块；
3. `config.py` 注册进 `"Summary"`；
4. 验收：拖动滑块流畅（只切片）、MDS 不漂移、incumbent 轨迹正确、hover 正确；
5. 运行中 run：明确“手动 Process / 拉滑块到最新”的行为，不做无限自动追平。

---

## 5. P4 瓶颈检测

### 5.1 用户故事与界面

Objective Analysis 类目下新增 **Bottleneck Diagnosis**：

- 曲线：incumbent 目标值 vs trial 序号（可切 time/trials）；
- 停滞段用半透明阴影带标注，按类型着色（局部最优型 / 无效探索型）；
- 图上方 banner 显示最新/选中段的诊断与建议；
- 运行中 run：每 1–2s 轻量刷新，新停滞段自动出现。

输入 keys：

`objective_id`、`budget_id`、`seed`（可选）、`xaxis`、`window_k`、
`eps_noise`、`theta_e`、`tau_q`、`min_trials`。

### 5.2 序列构造（每 trial 的 incumbent）

1. 从 `run.history` 取目标 budget / seed 下的 trial，按 `end_time` 排序；

2. 维护当前最优 cost（目标为 upper 时先转成负值）；

3. 每个 trial 输出：

   ```python
   {
       "trial_id": int,        # history 下标
       "order": int,           # 该 budget 下的第几次评估
       "time": float,          # end_time
       "cost_best": float,     # 截至当前的 incumbent cost
       "config_id": int,
       "is_new_config": bool,  # 该 config 是否第一次出现
       "nn_dist": float,       # 与既往所有 config 的最小归一化距离
   }
   ```

4. 不要直接用 `get_trajectory()` 当逐 trial 序列：它只返回 improvement 点，
   平台期会被压缩掉。可以用它交叉校验 incumbent 跳变点。

### 5.3 信号定义与判定

窗口大小：`k = max(3, min(window_k, floor(0.1 * n_trials)))`。

- `m_t`：窗口内 `cost_best` 发生严格下降的次数；
- `δ_t = (cost_best(t-k) - cost_best(t)) / max(|cost_best(t-k)|, 1e-9)`；
- `e_t`：窗口内 `is_new_config=True` 的 trial 中，
  `nn_dist >= τ` 的比例；`τ` 取既往 `nn_dist` 的 `tau_q` 分位数
  （默认 0.75；样本不足时使用小样本规则）。

判定：

```python
if n_trials < min_trials:
    state = "warming_up"
elif delta_t >= eps_noise:
    state = "ok"
elif e_t < theta_e:
    state = "local_optimum_stall"     # 采样集中、无改进
else:
    state = "ineffective_exploration" # 采样扩散、无改进
```

实现注意：

- `is_new_config` 必须按 config_id 定义，重复评估 incumbent 不算“新配置”；
- 距离按编码空间逐维归一化后计算，复用 footprint 的距离逻辑；
- 计算 `nn_dist` 时，v1 可与最近 M 个历史点比较（M 默认 200），
  避免 t 很大时每次 O(t·d) 增长过快；
- 连续两个窗口判定一致才生成/延长 segment，减少抖动误报；
- `eps_noise` 用相对值，但必须加绝对下限（如 `1e-6`）。

### 5.4 检测器实现

```python
class StagnationDetector:
    def __init__(self, k=20, eps_noise=1e-3, theta_e=0.5,
                 tau_q=0.75, min_trials=20):
        ...

    def update(self, record: dict) -> None:
        """追加一个新 trial 的记录（O(k) + 距离计算）。"""

    def detect(self) -> list[dict]:
        """返回当前全部停滞段：start_order/end_order/type/delta/e。"""
```

同时提供 `detect_full(records, ...)` 作为离线入口，单元测试与首帧都走它。

状态缓存键必须是：

```python
key = (run.id, run.hash, objective_id, budget_id, seed)
```

`run.hash` 变化说明有新 trial；键不同则重建 detector 并全量重放。

### 5.5 实时实现

- 用 DynamicPlugin；在插件内部加一个 `dcc.Interval`（1500ms）
  或复用全局 `global-update`，每次回调：
  1. 检查 `run.latest_change` / `run.hash` 是否变化；
  2. 未变化 → `PreventUpdate` / `no_update`；
  3. 变化 → 读取新增 trial，调用 `detector.update()`，返回新 figure + banner；
- 不做全图每 500ms 重绘；
- 切换 run / budget / objective 时重建 detector；
- 如果 run 很长（>1000 trial），提供“重新检测”按钮触发一次性全量计算。

### 5.6 单 run / Group 行为

- 插件 `activate_run_selection=True`，逻辑上以单 run 为输入；
- 若用户选 Group：优先复用 `Group` 提供的 trajectory；
  若 Group 语义导致 config 映射不明确，直接给出 warning 并退化为“仅显示曲线不检测”。
- 多 run 并排对比不作为本插件职责（见 §0.3）。

### 5.7 返回值 schema

```python
{
    "segments": [
        {"start_order": 80, "end_order": 124, "type": "local_optimum_stall",
         "delta": 0.0004, "e": 0.12, "advice": "..."},
        ...
    ],
    "records": [...],       # 供 figure 使用
    "state": "ok" | "warming_up" | "local_optimum_stall" | "ineffective_exploration",
    "thresholds": {"k": 20, "eps_noise": 1e-3, "theta_e": 0.5, "tau": 0.15},
}
```

### 5.8 实现步骤 Checklist

1. `evaluators/stagnation.py`：`detect_full` + `StagnationDetector`；
2. 单元测试：健康收敛、局部最优型停滞、无效探索型停滞、噪声改进、warming-up；
3. plugin 五类方法 + 实时回调 + banner；
4. `config.py` 注册进 `"Objective Analysis"`；
5. 验收：
   - 离线示例 run 的阴影段与肉眼平台期一致；
   - 边跑 Optuna 边开页面，新段 1–2s 内出现且不卡顿；
   - 健康收敛 run 不误报为停滞；
   - 阈值全部可配，并在报告中标注“经验初值，待用户研究标定”。

---

## 6. 测试与联调

### 6.1 evaluator 单元测试（pytest）

| 模块               | 测试点                                                       |
| ------------------ | ------------------------------------------------------------ |
| `weighted_sobol`   | 已知函数排序；类别列不产生非法取值；V≈0 的报错；负值诊断     |
| `sampling_density` | 均匀采样的 pdf 近似平线；偏斜采样峰值；2D counts 形状与总计数 |
| `footprint_replay` | `first_seen` 正确；`subset(t)` 只含 ≤t 的点；incumbent 前缀单调 |
| `stagnation`       | 四类合成序列（见 §5.8）+ 边界条件（< min_trials、全失败）    |

### 6.2 API 模式冒烟（不启动 server）

```python
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.plugins.hyperparameter.weighted_sobol import WeightedSobol

run = DeepCAVERun.from_path(Path("logs/.../run_1"))
plugin = WeightedSobol()
inputs = plugin.generate_inputs(...)
outputs = plugin.generate_outputs(run, inputs)
figure = plugin.load_outputs(run, inputs, outputs)
```

四个 evaluator 必须都能在 API 模式下直接 import 和计算；四个 plugin 至少
各写一个 API 冒烟样例，作为回归测试脚本。

### 6.3 Web 模式联调清单

- `deepcave --open --n_workers=1`；
- 四个新页签出现在正确类目下；
- 静态插件（P1/P3）在 worker 在线时能出结果，worker 离线时有错误提示；
- 动态插件（P2/P4）拖动控件无明显卡顿；
- 缓存：同一 run 二次打开直接命中；run.hash 变化后能失效重算；
- 运行中 run：P4 能追新，P3 手动刷新行为符合预期。

---

## 7. 工程实施顺序（里程碑）

| 里程碑                   | 内容                                                         | 验收标准                                    | 主要风险                                             |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------- | ---------------------------------------------------- |
| **M1 环境骨架**          | Linux/容器 + Python 3.9/3.10 + redis-server + `pip install -e .`；打 history 加载补丁；用 `logs/` 示例 run 跑通 footprint/importances | Web 全功能可点；API 模式可 import evaluator | pyrfr/numpy/dash 版本；Redis 未装；Mac 不支持 fANOVA |
| **M2 P2 采样密度**       | 建立“新插件全链路”模板（五类方法 + 注册）                    | 新页签出现；1D/2D 图正确；改源码热生效      | 类别列/条件列处理；2D 性能                           |
| **M3 P3 Footprint 回放** | StaticPlugin 全量点云 + filter 滑块 + 缓存键                 | 滑块只切片不重算；MDS 不漂移；hover 正确    | first_seen 定义；运行中刷新策略                      |
| **M4 P1 加权 Sobol**     | evaluator + StaticPlugin + RQ 全流程                         | 已知函数排序正确；均匀/加权对照可解释       | 采样器类型处理；方差地板；条件列                     |
| **M5 P4 瓶颈检测**       | detector + 单测 → 插件 + 阴影带/banner → 实时回路            | §5.8 全部验收项                             | 增量状态/缓存一致；500ms 回调误放重活                |
| **M6 联调收尾**          | evaluator pytest 全绿；API 冒烟；Web 手动清单 + 截图         | 双入口均可用；报告素材齐                    | —                                                    |

**顺序理由**：M2 最简单且建立插件范式 → M3 验证 StaticPlugin + filter 的架构模式
（P1 也会用到静态插件）→ M4 独立走 RQ → M5 最后做（依赖最复杂的实时与增量逻辑）。

---

## 8. 复用资产清单（调试不用先跑实验）

仓库 `logs/` 下已有 SMAC3v1/v2、Optuna、BOHB、AMTK、RayTune、DataFrame 等示例 run：

- P1/P2：选含 ≥100 trial 且后期收敛明显的 run；
- P3：选配置数和 trial 数适中（几百以内）的 run，便于观察 MDS 回放；
- P4 实时演示：单独起一个 Optuna study，边写盘边观察页面。

**注意**：`logs/` 的 run 是旧 8 字段格式，正好可以验证 §0.4 的兼容补丁；
自己用 `Recorder` 生成的新 run 是 9 字段格式，也必须能加载。

---

## 9. 明确不做 / 边界声明汇总（写报告 Future Work 用）

1. 不做 Sobol 成对/纯交互输出（P1 单参数口径）。
2. 不新增多优化器专用对比插件（DeepCAVE Group / 多 run 选择已支持）。
3. 不做搜索空间收缩建议输出（P2 仅展示密度）。
4. 不做独立 Regret 插件；regret 仅用于离线验证停滞检测器。
5. 不做跨进程状态同步：实时增量缓存在 Web 进程内；多 worker 部署需外部化（Redis），留作工程扩展。
6. 阈值/带宽类参数均为经验初值，属于 5–10 人用户研究标定项。
7. 条件超参数的插补是近似；v1 只在无条件子句的空间上提供完整保证。

---

## 附录 A：`config.py` 注册改动

```python
# 文件头 import 区
from deepcave.plugins.hyperparameter.weighted_sobol import WeightedSobol
from deepcave.plugins.hyperparameter.sampling_density import SamplingDensity
from deepcave.plugins.summary.footprint_replay import FootprintReplay
from deepcave.plugins.objective.bottleneck_diagnosis import BottleneckDiagnosis

# Config.PLUGINS property 内
plugins = {
    "Summary": [
        Overview(),
        Configurations(),
        FootPrint(),
        FootprintReplay(),
    ],
    "Objective Analysis": [
        CostOverTime(),
        ParetoFront(),
        BottleneckDiagnosis(),
    ],
    "Budget Analysis": [
        BudgetCorrelation(),
    ],
    "Hyperparameter Analysis": [
        Importances(),
        WeightedSobol(),
        SamplingDensity(),
        AblationPaths(),
        ConfigurationCube(),
        ParallelCoordinates(),
        PartialDependencies(),
        SymbolicExplanations(),
    ],
}
```

## 附录 B：关键源码 API 速查（v1.4.1）

| 符号                                                         | 位置                                      | 说明                                                         |
| ------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------------------ |
| `run.get_encoded_data(objectives, budget, seed=None, statuses=..., specific=False, include_config_ids=False, include_combined_cost=False)` | `runs/__init__.py:1260`                   | 编码矩阵 DataFrame；**seed=None 时每 seed 一行，不自动平均** |
| `run.get_avg_costs(config_id, budget=None)`                  | `runs/__init__.py:711`                    | (config,budget) 上跨 seed 均值/标准差                        |
| `run.get_trajectory(objective, budget=None, seed=None)` → `(times, costs_mean, costs_std, ids, config_ids)` | `runs/__init__.py:1101`                   | 仅 improvement 点，不适合直接当逐 trial 序列                 |
| `run.get_budget(id, human=False)`                            | `runs/__init__.py:539`                    | budget_id → 真实 budget 值                                   |
| `run.get_highest_budget(config_id=None)`                     | `runs/__init__.py:623`                    | 默认最高预算                                                 |
| `run.get_incumbent(objectives=None, budget=None, seed=None, statuses=None, selected_ids=None)` | `runs/__init__.py:884`                    | incumbent 查询                                               |
| `run.hash` / `run.latest_change`                             | `converters/deepcave.py:48/67`            | `hash` = history.jsonl 文件哈希；缓存/失效键必须带它         |
| `StaticPlugin.process` → RQ                                  | `plugins/static.py`                       | P1/P3 继承                                                   |
| `DynamicPlugin` 回调直算                                     | `plugins/dynamic.py`                      | P2；P4 另加 Interval 轻量刷新                                |
| `Footprint` evaluator                                        | `evaluators/footprint.py:100/246/304/574` | P3 复用 MDS/热图/点云；4000 config cap                       |
| `Config.PLUGINS`                                             | `config.py:107-149`                       | 唯一注册点                                                   |

## 附录 C：编程前最终确认清单

- [ ] Python 版本与 DeepCAVE 安装方式已固定；
- [ ] history 加载补丁已打并可加载新 9 字段 run；
- [ ] Redis + worker 可启动；
- [ ] 四个 evaluator 的函数签名、返回 schema 已按本计划冻结；
- [ ] 每个插件的 input/filter 分布已确认（P3 滑块必须是 filter）；
- [ ] 缓存键包含 `run.hash`；
- [ ] 多 seed 折叠逻辑已统一；
- [ ] 类别/条件/常量超参数处理已明确；
- [ ] 单元测试与 API 冒烟测试已列入 M6；
- [ ] 用户研究阈值标定项已列出。

---

## 附录 D：插件元数据与接口契约（编程前冻结）

### D.1 类属性

| 插件 | 模块                                         | 类名                  | `id`                   | `name`                 | 基类            | `activate_run_selection` | 备注                               |
| ---- | -------------------------------------------- | --------------------- | ---------------------- | ---------------------- | --------------- | ------------------------ | ---------------------------------- |
| P1   | `plugins/hyperparameter/weighted_sobol.py`   | `WeightedSobol`       | `weighted_sobol`       | `Weighted Sobol`       | `StaticPlugin`  | `True`                   | `button_caption` 默认 `Process`    |
| P2   | `plugins/hyperparameter/sampling_density.py` | `SamplingDensity`     | `sampling_density`     | `Sampling Density`     | `DynamicPlugin` | `True`                   | `use_cache=True`；大 run 时降级    |
| P3   | `plugins/summary/footprint_replay.py`        | `FootprintReplay`     | `footprint_replay`     | `Footprint Replay`     | `StaticPlugin`  | `True`                   | 滑块必须放 filter block            |
| P4   | `plugins/objective/bottleneck_diagnosis.py`  | `BottleneckDiagnosis` | `bottleneck_diagnosis` | `Bottleneck Diagnosis` | `DynamicPlugin` | `True`                   | 建议 `use_cache=False`（实时刷新） |

`help` 均指向 `plugins/<对应文档>.html`；`icon` 任选现有语义相近的 FontAwesome 图标。

### D.2 输入 / 输出 key 契约

| 插件 | input block keys                                             | filter block keys                                           | output block keys                   |
| ---- | ------------------------------------------------------------ | ----------------------------------------------------------- | ----------------------------------- |
| P1   | `objective_id`(int)、`budget_id`(int)、`seed`(int/None)、`n_samples`(int)、`n_trees`(int)、`show_uniform_control`(bool)、`show_total_effect`(bool) | 无                                                          | `graph`(figure)、`table`(children)  |
| P2   | `objective_id`(int)、`budget_id`(int)、`mode`(single/pair)、`hp1`(str)、`hp2`(str/None) | `show_uniform`(bool)、`show_rug`(bool)、`show_points`(bool) | `graph`(figure)                     |
| P3   | `objective_id`(int)、`budget_id`(int)、`details`(float)      | `trial`(int)、`show_unvisited`(bool)                        | `graph`(figure)                     |
| P4   | `objective_id`(int)、`budget_id`(int)、`seed`(int/None)、`xaxis`(time/trials)、`window_k`(int)、`eps_noise`(float)、`theta_e`(float)、`tau_q`(float)、`min_trials`(int) | 无                                                          | `graph`(figure)、`banner`(children) |

### D.3 `load_inputs` 返回值模板

```python
# P3 示例（其它插件同理）
return {
    "objective_id": {"options": get_select_options(obj_names, obj_ids),
                     "value": obj_ids[0]},
    "budget_id": {"options": get_select_options(budget_labels, budget_ids),
                  "value": budget_ids[-1]},
    "details": {"value": 0.5},
}
```

`load_dependency_inputs(self, run, previous_inputs, inputs)` 在 run / 上游输入变化后
返回同结构字典；只返回需要更新的 key 即可（框架会 merge）。

### D.4 兼容性检查

- P1/P2/P3/P4 均在 `check_runs_compatibility` 中要求：
  - 至少 3 个成功 trial；
  - 目标可解析、budget 可解析；
- P4 额外要求：若输入是 Group 且无法建立稳定的 `config_id` 轨迹，
  只显示 incumbent 曲线，跳过检测并在 banner 中说明；
- P3 额外要求：config 数超过现有 Footprint 上限（4000）时沿用拒绝策略并提示。

### D.5 错误处理与状态反馈

- StaticPlugin：`process` 抛异常时框架会把 job 标为 failed；
  在 `load_outputs` 中检测失败状态并给出可读提示（参考 Importances 的
  `PluginState.FAILED` 分支）。
- DynamicPlugin：计算超时/异常时返回空 figure + banner 文本，
  不要把异常直接抛到全局回调；
- P4 实时：`run.hash` 未变化时返回 `no_update` / `PreventUpdate`；
- P2/P4：若当前浏览器会话已有同 key 缓存，优先复用。

### D.6 插件实现的最小骨架（以 P3 为例）

```python
class FootprintReplay(StaticPlugin):
    id = "footprint_replay"
    name = "Footprint Replay"
    icon = "fas fa-play-circle"
    help = "plugins/footprint_replay.html"
    activate_run_selection = True

    @staticmethod
    def get_input_layout(register):
        return [ ... ]          # objective_id / budget_id / details

    @staticmethod
    def get_filter_layout(register):
        return [ ... ]          # trial slider / show_unvisited

    def load_inputs(self):
        return { ... }

    def load_dependency_inputs(self, run, previous_inputs, inputs):
        return { ... }          # 更新 objective/budget 选项与 slider max

    @staticmethod
    def process(run, inputs):
        return {...}            # MDS 点云 + first_seen + incumbent_prefix（JSON 安全）

    @staticmethod
    def get_output_layout(register):
        return dcc.Graph(register("graph", "figure"), ...)

    @staticmethod
    def load_outputs(run, inputs, outputs):
        return _figure_from_slice(outputs, t=inputs["trial"])
```