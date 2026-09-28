# DeepCAVE-X：面向超参数优化过程的交互式诊断扩展 —— 选题报告（v1.1）

> 说明：本报告的功能范围以《DeepCAVE-X 插件开发计划 v1.1》为准，只包含
> **P1 加权 Sobol 重要性、P2 采样密度、P3 Footprint 时序回放、P4 瓶颈检测** 四项。
> 与任务书/早期 outline 不一致之处在 §0 集中订正。
> 事实性论断基于 DeepCAVE v1.4.1 源码（GitHub: automl/DeepCAVE）与官方文档交叉核对，
> 关键处标注文件位置。

---

## 0. 范围与订正说明

### 0.1 本报告的功能范围（与开发计划一致）

| #    | 功能                   | 解决什么问题                                                 |
| ---- | ---------------------- | ------------------------------------------------------------ |
| P1   | **加权 Sobol 重要性**  | 均匀测度下的重要性可能被“从未探索的区域”主导；改为按优化器实际采样分布估计单参数重要性 |
| P2   | **采样密度展示**       | Footprint 的 MDS 压缩丢失逐维信息；用 1D 边际 / 2D 联合密度直接展示采样覆盖与偏斜 |
| P3   | **Footprint 时序回放** | Footprint 是终态快照；加 trial 滑块回放“点如何铺开、如何收缩” |
| P4   | **瓶颈检测**           | Cost over Time 只显示自身进度；自动标注停滞区间并给出类型化建议，支持运行中轻量刷新 |

### 0.2 订正说明（相对任务书 / 早期 outline）

| #    | 早期表述                                                     | 核查结论                                                     | 依据                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | “填补 DeepCAVE 仅支持单次运行分析的空白”                     | **不成立**。DeepCAVE 已支持多 run 选择与 Group Analysis；Cost over Time 等插件可叠加多条曲线 | `runs/group.py` 的 `Group` 类；`cost_over_time` 的 Show Runs / Show Groups 选项 |
| 2    | DeepCAVE 的 fANOVA 无法体现交互信息                          | **需精确化**。算法层 `get_importances(depth≥2)` 支持参数对/交互方差，但插件层从未传 `depth>1`，UI 也没有入口，属于“算而不用” | `evaluators/fanova.py`；`plugins/hyperparameter/importances.py` 的调用点 |
| 3    | 把“集成超参数重要性分析”当作从零开发                         | DeepCAVE 已有 fANOVA/LPI 插件；本项目是**扩展**：新增加权 Sobol 单参数口径，并与现有方法做描述性对照 | `plugins/hyperparameter/importances.py`                      |
| 4    | 任务书中的“多优化器并排对比 / 搜索空间收缩建议 / 交互效应视图” | 与开发计划 v1.1 对齐：**本期不做专用多优化器对比**（Group 已支持基础能力，先不改 DeepCAVE 现有部分）；不做成对交互；不做空间建议 | 开发计划 §0.3                                                |

### 0.3 明确不做（Future Work）

1. Sobol 成对 / 纯交互重要性图；
2. 新增多优化器专用对比插件（DeepCAVE 现有 Group / 多 run 选择已覆盖基础场景）；
3. 搜索空间收缩建议、ForbiddenClause、新空间导出；
4. 独立 Regret 曲线插件（regret 只在离线评估中作为停滞检测的弱 ground truth）。

---

## 1. 背景与问题

### 1.1 从机器学习到 AutoML

AutoML 旨在降低机器学习应用门槛，其中**超参数优化（HPO）**是核心环节：
给定搜索空间 $\mathcal{X}$ 与目标函数 $f$（如验证误差），在有限评估预算内寻找
性能最优的配置 $x^\*$。主流方法包括随机搜索、贝叶斯优化（SMAC 等）、
多保真方法（Hyperband/BOHB）以及 Optuna/Ray Tune 等工程框架。

### 1.2 HPO 的“黑箱”问题

即使优化算法不断进步，用户仍难以回答：

1. 哪些超参数真正重要？当前重要性结论是否受“未探索区域”影响？
2. 优化器搜索行为是否健康？它在探索还是在收缩？哪些维度覆盖不足？
3. 搜索是否已经停滞？停滞是“采样太集中”还是“采样太散但无效”？
4. 运行中能否及时发现停滞，而不是等结束后才复盘？

现有工具（TensorBoard/HParams、W&B sweeps、Optuna Dashboard、DeepCAVE 等）
对“结果记录与汇总”覆盖较好，但对**过程行为的可视化与诊断**仍有空间。

### 1.3 研究问题（与 4 个功能一一对应）

- **RQ1（重要性测度）**：把 Sobol 的基线测度从均匀分布换成优化器经验采样分布后，
  单参数重要性排序是否发生变化？这种变化能否用“BO 后期收窄”来解释？
- **RQ2（过程行为）**：采样密度图与时序回放能否让用户更直观地判断
  “探索是否充分、何时开始收缩”？
- **RQ3（瓶颈诊断）**：能否在运行中自动检测停滞区间，并按
  “采样集中 / 采样扩散”两类给出可读建议？

---

## 2. DeepCAVE 基座

### 2.1 架构

DeepCAVE 采用计算层 / 展示层分离的插件架构：

```text
deepcave/
├── runs/            数据层：Run/Trial、(config, budget, seed) 三元组、Group、handler
│   └── converters/  SMAC3v1/v2、Optuna、BOHB、AMTK、RayTune、DataFrame、原生格式
├── evaluators/      计算层：fanova、lpi、footprint（MDS）、ablation、epm 代理模型
├── plugins/         展示层：summary / objective / budget / hyperparameter 下的 Dash 插件
└── layouts/         页面骨架（header、sidebar、全局刷新节拍）
```

**双入口**：进程以 `server.py`/`worker.py` 启动时创建 Dash app 与 Redis 队列；
否则进入 API 模式，`@interactive` 装饰的 GUI 回调空转。因此：

- **evaluator 算法层**在 Web 与 API 两种模式下都可用；
- **plugin 展示层**只在 Web 模式参与回调。

**任务执行模式**：

- `StaticPlugin.process` 通过 RQ 投递到 Redis，worker 计算后缓存结果；
- `DynamicPlugin` 回调内直接计算，必须控制在数百毫秒；
- 全局 `dcc.Interval` 的刷新节拍为 `REFRESH_RATE = 500ms`。

**增量更新**：`runs/handler.py` 周期比对 `run.hash`（结果是结果文件哈希）；
哈希变化才重载 run 并清缓存。这是 P4 实时刷新与 P3 缓存失效的基础。

### 2.2 现有功能速览

| 插件 / 能力                                                 | 回答的问题                               |
| ----------------------------------------------------------- | ---------------------------------------- |
| Overview                                                    | 状态、目标、budget、配置空间是否如预期   |
| Cost over Time                                              | 收敛了吗？谁更快？多 run / Group 可叠加  |
| Configuration Footprint                                     | 搜索空间覆盖是否充分、incumbent 是否可靠 |
| Importances（fANOVA / LPI）                                 | 全局 / 局部参数重要性                    |
| PDP / Parallel Coordinates                                  | 单参数或参数组合与性能的关系             |
| Budget Correlation                                          | 低预算排名能否代表高预算                 |
| Pareto Front                                                | 多目标权衡                               |
| Ablation Paths / Configuration Cube / Symbolic Explanations | 路径、空间切片、显式公式解释             |

### 2.3 多 run 能力的事实说明

DeepCAVE 的 `Group` 可以把多个 run 组合为一个 `AbstractRun`，
所有插件可以像处理单 run 一样处理 Group（如 Cost over Time 显示均值±标准差）。
因此“多 run 支持”不是本项目创新点，也**不是本期开发任务**；
本项目的四项功能按“单 run / 单 Group”设计。

### 2.4 本项目要补的缺口

| 缺口                                                      | 对应功能 |
| --------------------------------------------------------- | -------- |
| fANOVA/LPI 的重要性定义在均匀测度上，与实际偏斜采样不一致 | P1       |
| Footprint 是静态终态，且 MDS 压缩丢失逐维覆盖信息         | P2、P3   |
| Cost over Time 只有自身进度，没有停滞检测与类型化建议     | P4       |

---

## 3. 问题分析与方案动机

### 3.1 P1：重要性测度与实际采样分布的失配

**现状**：DeepCAVE 的 fANOVA 是全局方差分解，LPI 是 incumbent 邻域的局部扰动重要性。
两者都在源码中实现完整，是成熟的现有能力。

**问题**：fANOVA/经典 Sobol 的分解建立在搜索空间上的均匀测度 $\mu$ 上。
但 BO 类优化器的采样是刻意偏斜的：后期会集中在有希望的区域。于是
“均匀测度下重要性高”可能只说明“未探索区域性能波动大”，
对“实际搜索区域内谁重要”没有直接回答。

**方案**：加权 Sobol。把基线测度从均匀分布换成优化器经验采样分布
$\hat{p}$，得到单参数一阶指数：

$$
S_i^{w} = \frac{\mathrm{Var}_{X \sim \hat{p}}\left[\mathbb{E}(M(X) \mid X_i)\right]}
{\mathrm{Var}_{X \sim \hat{p}}\left[M(X)\right]},
\qquad
S_{T_i}^{w} \text{ 同理（可选开关）}.
$$

其中 $M$ 是在 DeepCAVE 编码数据上训练的随机森林代理。
本期**只输出每个超参一个数**（S1；ST 可选），不输出参数对/交互；
交互效应虽然算法层可计算，但明确列为 Future Work。

### 3.2 P2：MDS 压缩丢失逐维覆盖信息

**现状**：Configuration Footprint 用 MDS 把高维配置压到 2D，
再叠加性能热图与四类点（evaluated / border / random support / incumbent）。

**问题**：MDS 的距离是所有维度的混合：

1. 单个超参数的搜索范围完全不可见；
2. 无法区分“哪个维度没采到样”；
3. 非专业用户很难从点云判断 BO 是否已经收缩。

**方案**：采样密度图。

- 1D：单超参密度曲线 + 均匀基线 + rug + incumbent 标记；
- 2D：两个超参的联合密度热图 + 已评估点 + incumbent 星标；
- 与 P1 共享同一个“经验采样分布”概念：密度图是 $\hat{p}$ 的可视化，
  加权 Sobol 是 $\hat{p}$ 的定量利用。

本期只做**展示**，不输出区间建议。

### 3.3 P3：Footprint 的静态快照

**现状**：Footprint 把全部 trial 一次性画在同一张图上。

**问题**：丢失时间维度，无法表达“第 30 个 trial 后开始收缩到某个区域”这类过程叙事；
非专业用户最难理解的恰是“过程”。

**方案**：给 MDS 点云加 trial 滑块，回放配置如何铺开、incumbent 如何移动。
关键约束：**MDS 只在全部点上拟合一次并冻结**；
拖动滑块只做数组切片，不重算 MDS（否则点会跳动、动画不可读）。

### 3.4 P4：Cost over Time 不等于“没有停滞”

**现状**：Cost over Time 显示 incumbent 随时间的改进；曲线变平既可能是“已收敛”，
也可能是“停滞”。

**方案**：在 incumbent 曲线上检测连续无改进区间，并用探索率区分两类停滞：

| 类型           | 判定直觉              | 建议方向                             |
| -------------- | --------------------- | ------------------------------------ |
| 局部最优型停滞 | 无改进 + 采样高度集中 | 提高探索、混合随机采样、重启 sampler |
| 无效探索型停滞 | 无改进 + 采样持续扩散 | 收缩搜索空间、提前止损               |

运行中的 run 通过哈希变化 + 轻量增量更新，1–2s 内把新停滞段显示出来。
离线评估时用“事后全局最优”构造 regret 平台作为弱 ground truth，
但 regret 本身不作为独立交付插件。

---

## 4. DeepCAVE-X 技术设计（摘要）

> 详细实现细节、函数签名、返回值 schema、测试清单见开发计划 v1.1。

### 4.0 数据契约（编程前必须统一）

1. **编码数据**：通过
   `run.get_encoded_data(objectives, budget, statuses=[Status.SUCCESS],
   include_config_ids=True)` 获取；
   返回的数值列是 ConfigSpace 归一化向量（连续/整数在 [0,1]，log 参数在 log 尺度
   归一化，类别为下标/(类别数−1)），因此编码空间内的“均匀采样”即 ConfigSpace 先验采样；
   需要显示原始值时用 `hp.to_value(encoded)` 反变换。
2. **多 seed 折叠**：`get_encoded_data(seed=None)` 会为每个 seed 生成一行，
   **不会自动平均**；必须先按 `CONFIG_ID` 聚合为一行（或使用 `get_avg_costs`），
   再进入 P1/P2；
3. **budget**：UI 的 `budget_id` 是索引，必须用 `run.get_budget(budget_id)`
   解析成真实 budget；本期默认取最高预算；
4. **目标方向**：统一转成 lower-is-better 的 cost
   （`cost = -value if objective.optimize == "upper" else value`）；
5. **类别 / 条件 / 常量超参数**：类别列按频率抽样，不做高斯扰动；
   条件维度需插补或限制在无条件空间；常量维度排除并记录。

### 4.1 P1 实现要点

- 代理：sklearn `RandomForestRegressor`（不依赖 `pyrfr`，Mac 上也能开发）；
- 采样：经验边际乘积；连续列 bootstrap + 小抖动，类别/整数列 bootstrap；
- 估计：Saltelli 列交换估计 S1/ST，负值保留原始值、展示时截断；
- 方差地板：`V = max(V, 1e-12)`，V 过小直接报错；
- 均匀对照：同一代理、同一 N，仅把采样器换成均匀/类别等概率；
- 任务执行：StaticPlugin + RQ。

### 4.2 P2 实现要点

- 1D：连续列 KDE 或直方图（网格 ≤200），类别列 frequency bar；
- 2D：默认 `np.histogram2d`（网格 ≤80×80）；不对 256×256 网格做二维 KDE；
- 均匀基线：连续 `1/(hi-lo)`，类别 `1/类别数`；
- 基类：DynamicPlugin，计算前检查 run 哈希与缓存，必要时降级方案。

### 4.3 P3 实现要点

- **StaticPlugin + filter 滑块**：
  - input block：`objective_id`、`budget_id`、`details`（触发计算）；
  - filter block：`trial` 滑块、`show_unvisited`（只重绘，不重算）；
- process 输出：MDS 坐标 + 热图 + `first_seen` + `incumbent_prefix`；
- `first_seen` 遍历完整 `run.history`（按 end_time 排序），
  不能只看 `get_trajectory()` 的 improvement ids；
- 缓存键包含 `(run.id, run.hash, objective_id, budget_id, details)`；
- 运行中 run：哈希变化后由缓存失效触发重算（手动 Process 或明确的刷新策略）。

### 4.4 P4 实现要点

- 逐 trial 构造 incumbent 序列（不能用 `get_trajectory()` 直接替代，
  因为它只保留 improvement 点）；
- 信号：`m_t`（改进次数）、`δ_t`（相对净改进）、`e_t`（新配置的最近邻距离超过
  τ 的比例）；
- 判定：`δ_t < ε` 前提下，按 `e_t` 区分局部最优型 / 无效探索型；
- 检测器支持离线全量 + 流式增量；状态键为
  `(run.id, run.hash, objective_id, budget_id, seed)`；
- 实时：DynamicPlugin + 1.5s Interval，只在 `run.hash` 变化时增量更新；
- Group：可显示曲线，但若 config 映射不明确则降低检测级别并提示。

### 4.5 代码落位

```text
deepcave/evaluators/
  weighted_sobol.py
  sampling_density.py
  footprint_replay.py
  stagnation.py
deepcave/plugins/
  hyperparameter/weighted_sobol.py
  hyperparameter/sampling_density.py
  summary/footprint_replay.py
  objective/bottleneck_diagnosis.py
```

注册点：`deepcave/config.py` 的 `Config.PLUGINS`。

---

## 5. 评估计划与四页论文

### 5.1 实验数据

- 选取 2–3 个公开分类数据集（OpenML，或 sklearn 内置公开数据作为 fallback）；
- 用统一 ConfigSpace、统一目标（如 balanced accuracy 或 1−balanced accuracy）
  与统一 seed，运行 SMAC 与 Optuna-TPE，生成 DeepCAVE run；
- 这些 run 也可通过 DeepCAVE 现有 Group/多 run 选择查看；
  本项目不新增专用对比视图。

### 5.2 定量评估

| 功能 | 指标                                                         |
| ---- | ------------------------------------------------------------ |
| P1   | 加权 Sobol vs 均匀 Sobol 的排序差异（Spearman / Top-k 重合度）；合成函数上的排序正确性；不同 N 与 n_trees 的稳定性 |
| P2   | 均匀采样的密度应近似平线；偏斜采样的峰值位置符合直觉；2D 图计算延迟 |
| P3   | 滑块单帧延迟；MDS 坐标漂移量（应为 0）；`first_seen` 与 incumbent 前缀正确性 |
| P4   | 与事后 regret 平台的区间重合率；健康收敛 run 的误报率；运行中检测延迟；不同阈值下的敏感性 |

### 5.3 用户研究（任务书要求）

- 招募 5–10 位同学；
- 统一任务剧本：判断采样是否收缩、找出最重要参数、识别停滞区间并解释建议；
- 收集 SUS 量表 + 半结构化访谈；
- 以现有 Cost over Time / Footprint / Importances 为对照，记录任务正确率与用时；
- 用户研究同时用于标定 P4 的经验阈值（ε、θ_e、τ、k）与 P1 的 N/n_trees 默认值。

### 5.4 四页论文结构（建议）

1. Introduction & Motivation（0.5–0.75 页）：HPO 黑箱问题、任务与贡献；
2. Background & Design（1–1.25 页）：DeepCAVE 架构、P1–P4 的设计选择；
3. Case Study / Evaluation（1–1.25 页）：公开数据实验、定量指标、图表；
4. User Feedback & Conclusion（0.5 页）：5–10 人反馈、limitation、Future Work。

必备图：加权 vs 均匀 Sobol 对照、1D/2D 密度图、Footprint 回放帧、停滞阴影带 + banner。

### 5.5 边界与 limitation

- 所有 Sobol/重要性结论以代理模型质量为上限；
- 乘积边际测度忽略超参相关；
- 条件超参数的插补是近似；
- 停滞检测是启发式规则，阈值需要用户研究标定；
- 本期不做成对交互、不做专用多优化器对比、不做搜索空间建议输出。

---

## 参考文献

1. Segel, S., Graf, H., Bergman, E., et al. DeepCAVE: A Visualization and Analysis
   Tool for Automated Machine Learning. arXiv:2206.03493, 2022.
   （如另有 2025 年新论文 arXiv:2512.01810，请在正文中单独列出并核对引用信息。）
2. Hutter, F., Hoos, H., Leyton-Brown, K. An Efficient Approach for Assessing
   Hyperparameter Importance. ICML, 2014.
3. Biedenkapp, A., et al. Efficient Parameter Importance Analysis via Ablation
   with Kernels. 2021.
4. Sobol, I. Global sensitivity indices for nonlinear mathematical models.
   Mathematics and Computers in Simulation, 2001.
5. Saltelli, A., et al. Variance based sensitivity analysis of model output:
   design and estimator for the total sensitivity index. CPC, 2010.
6. Bergstra, J., Bengio, Y. Random Search for Hyper-Parameter Optimization.
   JMLR, 2012.
7. DeepCAVE 源码仓库：github.com/automl/DeepCAVE（本报告基于 v1.4.1 核查）。