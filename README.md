# DeepCAVE-X（deepcave-experiments）

**DeepCAVE-X** is a decision-centered visual analytics framework for hyperparameter optimization.
It treats an HPO run not merely as a collection of trials to be visualized, but as a **sequential decision-making process**.
DeepCAVE-X organizes diagnostic evidence into four layers: **performance sufficiency, parameter attribution, search behavior, and resource fidelity**.
By linking visualization evidence to concrete actions—stop, continue, expand, contract, reparameterize, or adjust fidelity—DeepCAVE-X closes the loop between visual analysis and HPO decision-making.

> **中文简介**：DeepCAVE-X 是一个以决策为中心的超参数优化（HPO）可视化分析框架。它把一次 HPO 运行看作一个序列决策过程，而不仅仅是待可视化的试验集合；将诊断证据组织为"性能充分性、参数归因、搜索行为、资源保真度"四个层面，并把可视化证据与具体行动（停止 / 继续 / 扩展 / 收缩 / 重参数化 / 调整保真度）关联起来，打通"可视化分析 → HPO 决策"的闭环。

本仓库基于 **DeepCAVE v1.4.1**（[automl/DeepCAVE](https://github.com/automl/DeepCAVE)），在**保留其全部原有功能**的基础上，新增了三组实验性诊断能力（对应 `development-plan.md` 中的 P1–P3）：

| # | 新增功能 | 挂载方式 | 代码位置 |
|---|---------|---------|---------|
| **P1** | **加权 Sobol 重要性**：用优化器的*经验采样分布*替代均匀分布来估计一阶 Sobol 重要性（S1，可选总效应 ST 与均匀对照），回答"重要性结论是否被未探索区域主导" | 集成进原有 **Importances** 插件，方法下拉框新增 `Weighted Sobol (empirical)` | `deepcave/evaluators/weighted_sobol.py`、`deepcave/plugins/hyperparameter/importances.py` |
| **P2** | **采样密度展示**：选 1 个超参画一维边际密度（含均匀基线、样本 rug、incumbent 位置），选 2 个超参画二维联合密度直方图，直接展示搜索的覆盖与偏斜 | 全新插件 **Sampling Density**（Hyperparameter Analysis 分类） | `deepcave/plugins/hyperparameter/sampling_density.py`、`deepcave/evaluators/sampling_density.py` |
| **P3** | **Footprint 时序回放**：在 Configuration Footprint 点云上增加 trial 序号滑块，回放"配置点如何铺开、如何收缩"；MDS 投影只计算一次、坐标冻结，红线显示截至当前 trial 的 incumbent 序列 | 全新插件 **Footprint Replay**（Summary 分类） | `deepcave/plugins/summary/footprint_replay.py`、`deepcave/evaluators/footprint_replay.py` |

> 说明：开发计划中的 P4"瓶颈检测"（incumbent 曲线停滞区间标注）在 `development-plan.md` / `proposal.md` 中有完整设计，但当前仓库代码尚未实现。

---

## 目录

1. [项目结构](#1-项目结构)
2. [环境要求](#2-环境要求)
3. [安装（从仓库路径 pip install，已验证）](#3-安装从仓库路径-pip-install已验证)
4. [启动与使用](#4-启动与使用)
5. [插件（Plugin）功能详解](#5-插件plugin功能详解)
6. [Evaluator（算法层）功能详解](#6-evaluator算法层功能详解)
7. [架构说明](#7-架构说明)
8. [示例数据与示例代码](#8-示例数据与示例代码)
9. [安装与启动验证记录](#9-安装与启动验证记录)
10. [插件开发规范（插件实现必须遵守）](#10-插件开发规范插件实现必须遵守)
11. [致谢与引用](#11-致谢与引用)

## 1. 项目结构

```
deepcave-experiments/
├── deepcave/                     # 主包（DeepCAVE-X，v1.4.1）
│   ├── __init__.py               #   应用初始化、Web/API 双模式判定
│   ├── cli.py                    #   `deepcave` 命令行入口（absl flags）
│   ├── start.sh                  #   启动脚本：redis → worker → server
│   ├── server.py / worker.py / open.py
│   ├── config.py                 #   配置类：端口、目录、插件/转换器注册表
│   ├── constants.py / custom_queue.py
│   ├── plugins/                  # 【插件层】Dash 交互展示（仅 Web 模式生效）
│   │   ├── static.py             #   StaticPlugin 基类（重计算走 RQ 队列+缓存）
│   │   ├── dynamic.py            #   DynamicPlugin 基类（轻量计算，回调内直接执行）
│   │   ├── summary/              #   Overview / Configurations / Footprint / Footprint Replay★
│   │   ├── objective/            #   Cost over Time / Pareto Front
│   │   ├── budget/               #   Budget Correlation
│   │   └── hyperparameter/       #   Importances(含★Weighted Sobol) / Ablation Paths /
│   │   │                         #   Configuration Cube / Parallel Coordinates /
│   │   │                         #   Partial Dependencies / Symbolic Explanations /
│   │   │                         #   Sampling Density★
│   ├── evaluators/               # 【算法层】纯 Python 计算，Web 与 API 双入口通用
│   │   ├── fanova.py / lpi.py / ablation.py / footprint.py
│   │   ├── mo_fanova.py / mo_lpi.py / mo_ablation.py   # 多目标变体
│   │   ├── weighted_sobol.py★ / sampling_density.py★ / footprint_replay.py★
│   │   └── epm/                  #   代理模型（pyrfr 随机森林、pyPDP 兼容代理）
│   ├── runs/                     # 【数据层】Run/Group/Trial/Objective/Recorder
│   │   └── converters/           #   smac3v1 / smac3v2 / bohb / optuna / amltk /
│   │                             #   raytune / dataframe / deepcave（原生）
│   ├── layouts/                  #   页面布局（侧边栏、页头、主区域、通知）
│   └── utils/                    #   工具（布局、压缩、日志、样式化绘图等）
├── configs/                      # 配置模板：local.py（DEBUG）/ server.py（生产）
├── logs/                         # 预置示例运行（见 §8）
├── examples/                     # 示例：api/（API 用法）、record/（Recorder 记录）
├── tests/                        # 单元测试：test_evaluators / test_runs / test_utils
├── docs/                         # Sphinx 文档源（含各插件 rst 说明）
├── development-plan.md           # DeepCAVE-X 插件开发计划（中文，v1.1）
├── proposal.md                   # DeepCAVE-X 选题报告（中文，v1.1）
├── CHANGELOG.md / Makefile / setup.py / requirements.txt / pyproject.toml
└── README.md                     # 本文件（★ = 本仓库相对原版 DeepCAVE 新增）
```

## 2. 环境要求

| 依赖 | 要求 | 说明 |
|------|------|------|
| Python | **3.9 或 3.10**（`>=3.9, <3.11`） | 版本过高/过低均无法安装 |
| 操作系统 | Linux（官方仅正式支持） | macOS 上 fANOVA/`pyrfr` 存在已知兼容问题 |
| redis-server | 必需 | Web 模式的任务队列（默认端口 6379） |
| swig | 编译 `pyrfr` 所需 | `conda install -c anaconda swig` 或 `apt install swig` |
| Python 包 | 见 `requirements.txt` | ConfigSpace 1.2.0、dash 2.0.0、pyrfr、plotly、rq 等，安装时自动解决 |

## 3. 安装（从仓库路径 pip install，已验证）

```bash
# 1) 创建并激活 conda 环境
conda create -n DeepCAVE python=3.9 -y
conda activate DeepCAVE

# 2) 安装系统依赖（Ubuntu/Debian；已有可跳过）
apt-get install -y redis-server swig

# 3) 从本地仓库路径安装（editable 模式：修改源码即时生效）
pip install -e /root/deepcave-experiments
```

可选扩展（按需）：

```bash
pip install -e '/root/deepcave-experiments[optuna]'    # 加载 Optuna 运行
pip install -e '/root/deepcave-experiments[bohb]'      # 加载 BOHB 运行
pip install -e '/root/deepcave-experiments[raytune]'   # 加载 RayTune 运行
pip install -e '/root/deepcave-experiments[examples]'  # 运行 MNIST/PyTorch 等示例
pip install -e '/root/deepcave-experiments[dev]'       # 开发与测试工具链
```

安装成功后，`deepcave` 命令会被注册到环境 `bin/` 下（`setup.py` 的 `console_scripts` 入口 → `deepcave.cli:main`）。

## 4. 启动与使用

### 4.1 Web 版（交互式仪表盘）

```bash
conda activate DeepCAVE
deepcave --open            # 启动并自动在浏览器打开（默认 1 个 worker）
deepcave --n_workers=2     # 指定后台 worker 数（StaticPlugin 重计算并行度）
deepcave --config server   # 使用 configs/server.py 配置（生产、非 DEBUG）
```

`deepcave` 命令经由 `deepcave/start.sh` 依次完成：

1. 检查/启动 **redis-server**（默认 6379 端口，任务队列）；
2. 启动 N 个 **RQ worker**（`worker.py`，异步执行 StaticPlugin 的重计算并缓存结果）；
3. 启动 **Dash Web 服务器**（`server.py`，默认 **http://127.0.0.1:8050/**）。

> **远程服务器访问**：DeepCAVE 默认监听 `127.0.0.1:8050`。在本地机器执行
> `ssh -L 8050:127.0.0.1:8050 user@server` 建立端口转发后，即可在本机浏览器打开
> http://127.0.0.1:8050 。也可以自定义 `configs/*.py` 中的 `DASH_ADDRESS`/`DASH_PORT`，
> 再用 `--config` 传入。

### 4.2 API 版（脚本化分析，无需启动 GUI）

DeepCAVE 检测到非 `server.py/worker.py` 执行时自动进入 API 模式，evaluator 算法层可直接调用：

```python
from deepcave.runs.converters.deepcave import Run  # 或 smac3v2/optuna 等转换器
from deepcave.evaluators.fanova import fANOVA
from deepcave.evaluators.weighted_sobol import calculate as weighted_sobol
from deepcave.evaluators.footprint_replay import build_cloud, subset

run = Run.from_path("logs/DeepCAVE/minimal/run_1")

# 原有：fANOVA 全局重要性
ev = fANOVA(run)
ev.calculate(run.get_objective(0), budget=run.get_highest_budget())
print(ev.get_importances())

# 新增 P1：加权 Sobol（经验采样分布）
result = weighted_sobol(run, run.get_objective(0), budget=run.get_highest_budget())

# 新增 P3：Footprint 回放数据
cloud = build_cloud(run, run.get_objective(0), budget=run.get_highest_budget())
frame = subset(cloud, trial=10)
```

更多示例见 `examples/api/`（importances、ablation_paths、pdp、parallel_coordinates）与 `examples/record/`（minimal、digits_sklearn、mnist_pytorch）。

## 5. 插件（Plugin）功能详解

插件按网页侧边栏分为四类，共 **16 个**。基类有两种：

- **StaticPlugin**（`plugins/static.py`）：重计算任务通过 RQ 队列投递给 worker 异步执行并缓存（如训练随机森林），页面不卡顿；
- **DynamicPlugin**（`plugins/dynamic.py`）：轻量计算，直接在 Dash 回调中同步执行（如概览统计）。

两者都通过 `Config.PLUGINS`（`deepcave/config.py`）注册；均支持单 run 与 Group（多 run 合并）分析。

### 5.1 Summary（运行概览类）

| 插件 | 基类 | 功能 |
|------|------|------|
| **Overview** | Dynamic | 所选 run 的总览：元数据（优化器、配置空间、随机种子）、目标定义、trial 统计（成功/失败/超时数量）等最核心信息。 |
| **Configurations** | Dynamic | 逐条查看配置特征：以表格+滑块方式浏览 run 中的每个配置，展示各超参取值、目标代价，并支持跳转到其它插件的对应视图。 |
| **Configuration Footprint** | Static | 配置足迹：将高维配置空间经 **MDS（多维缩放）** 压缩到 2D，叠加性能热力表面，展示已评估配置、随机支撑配置、边界配置与 incumbent 的分布——"搜索到底覆盖了哪些区域"。 |
| **Footprint Replay** ★新增 | Static | 足迹时序回放：在 Footprint 点云上加 **trial 序号滑块**，逐 trial 回放配置点如何铺开、如何向最优区域收缩；MDS 投影只算一次、坐标全程冻结，红线为截至当前 trial 的 incumbent 序列，未出现的配置可显示为浅灰。回放顺序按所选预算下成功 trial 的 `end_time` 排序，配置在首个成功 trial 处视为"已出现"。 |

### 5.2 Objective Analysis（目标分析类）

| 插件 | 基类 | 功能 |
|------|------|------|
| **Cost over Time** | Dynamic | 目标值随"时间/评估次数"的收敛曲线：跟踪 incumbent（当前最优）与各 trial 代价随时间的变化，支持多个 run/Group 叠加对比。 |
| **Pareto Front** | Dynamic | 帕累托前沿（多目标）：在两个目标构成的平面中画出非支配解集合（Pareto front），标出理想点，直观呈现多目标权衡关系。 |
| **Hypervolume Convergence** ★新增 | Dynamic | 超体积收敛曲线（多目标）：沿 trial 提交顺序增量维护 Pareto 前沿，在最多 400 个检查点处计算**支配超体积随评估进度/壁钟时间**的收敛曲线，支持多个 run/Group 叠加对比。默认将各目标归一化到 [0,1]（保证不同 run 可比），参考点取最差值 × 因子（默认 1.1）；d≤3 走精确算法，d≥4 且前沿超过阈值时切换 Monte Carlo 近似（算法档位标注在图中与汇总表首行）。汇总表输出各 run 的最终 HV、收敛曲线 AUC 与到达 90% 最终 HV 的时刻。选同一目标两次等价于"使用全部目标"；目标数 <2 时降级为警告提示。 |

### 5.3 Budget Analysis（预算分析类）

| 插件 | 基类 | 功能 |
|------|------|------|
| **Budget Correlation** | Dynamic | 预算相关性：计算不同预算（多保真档位）下配置代价的 Spearman 相关系数并绘制热力图，判断"低保真结果能否预测高保真表现"，辅助决定保真度设置是否合理。 |
| **Fidelity Replay** ★新增 | Dynamic | 保真度回放：把多保真 run 回放为 **Successive Halving 式保真阶梯**——以 run 的各预算为 rung，逐级展示评估配置数、最优代价、SH 模拟晋升数（每级仅保留 top 1/η）、被晋升配置在下一级的**真实评估覆盖率**，以及相邻级之间的**秩相关性（Spearman）**；η 可自动从预算阶梯推断（非几何预算回退 3）。缺失策略可选 `carry_last`（默认，沿用上次代价）/ `skip` / `worst`。图中最优评估代价（虚线）与最优活跃代价（实线）随保真度变化（对数横轴）；Group 时叠加各成员曲线，并输出成员 KPI 表（最高 rung 最优代价、覆盖率、秩相关）+ Mean 行。 |

### 5.4 Hyperparameter Analysis（超参分析类）

| 插件 | 基类 | 功能 |
|------|------|------|
| **Importances** | Static | 超参重要性柱状图，三种方法可选：**LPI**（局部：在 incumbent 邻域扰动单个超参引起的方差占比）、**fANOVA**（全局：随机森林方差分解）、**Weighted Sobol (empirical)** ★新增（以优化器实际采样分布为基准的一阶 Sobol 敏感性 S1，可选总效应 ST 与"均匀分布对照"开关，用于检验均匀测度结论是否被未探索区域扭曲）。支持双目标加权标量化（多目标重要性）。 |
| **Ablation Paths** | Static | 消融路径：从默认配置出发，按贪婪顺序逐个把超参替换成 incumbent 的取值，展示每步目标值变化，找出"从默认到最优的关键路径"。多目标时可做加权消融。 |
| **Configuration Cube** | Dynamic | 配置立方体：挑三个超参构成 3D 立方体，按目标值着色，直观展示配置在所选三维子空间中的分布与好坏。 |
| **Parallel Coordinates** | Static | 平行坐标图：每个配置是一条贯穿所有超参轴的折线，按目标值着色/筛选，快速发现好配置聚集的超参区域；结合 fANOVA 可只保留最重要的超参轴。 |
| **Partial Dependencies** | Static | 部分依赖图（PDP）：基于随机森林代理模型（pyPDP），展示单个/成对超参取值对预测性能的平均影响曲线/曲面。 |
| **Symbolic Explanations** | Static | 符号解释：用 gplearn 符号回归从代理模型中蒸馏出**可读的数学公式**并绘图，让"性能-超参"关系有显式表达。 |
| **Sampling Density** ★新增 | Dynamic | 采样密度：选 1 个超参画**一维边际密度**（叠加均匀空间基线、样本 rug 刻度、incumbent 位置），选 2 个超参画**二维联合密度直方图**（叠加已评估点与 incumbent）。类别/整数超参显示为离散频次质量，连续超参用有界直方图（1D 最多 200 bins，2D 每轴 80 bins）。用于诊断"搜索行为"：哪些维度覆盖不足、哪些区域采样过密。切换显示开关只重绘不重算。 |

### 5.5 新增插件原理详解（面向机器学习研究员）

本节面向不需要读源码的使用者，解释三个新增插件的算法思想、参数含义与代码落点。三个插件都遵循同一分层：**算法在 `deepcave/evaluators/`（纯函数、可脚本复用），交互与绘图在 `deepcave/plugins/`（Web 与 API 共用同一实现）**。

#### 5.5.1 Bottleneck Diagnostics（瓶颈诊断）

**算法原理**：把 run 的成功 trial 按时间顺序整理成记录流（成本统一为越小越好，多 seed 取平均），得到 incumbent 曲线，再用滑窗检测"停滞区间"。每个窗口综合三个信号：① 窗口内 incumbent 的相对改进率；② 新成本分布与历史分布的 Mann-Whitney U 检验（判断改进是否只是噪声）；③ 新颖率——用新配置在高维空间的最近邻距离衡量"是否还在尝试新区域"（阈值自适应，只保留最近 200 个配置的距离）。状态机据此把曲线切分为带类型的区间：预热中 / 局部最优停滞 / 无效探索，并按严重度着色；最后按规则库给出中英文调整建议（横幅展示），run 仍在评估时随页面自动刷新。

**核心参数**：`window_k`（滑窗长度，默认 20）；`eps_noise`（相对改进低于此值视为无进展，默认 0.001）；`theta_e`（新颖率阈值，默认 0.5）；`tau_q`（检验确认分位，默认 0.75）；`min_trials`（最少样本数，默认 20，不足时只报"预热中"）；`xaxis` 支持 Time / Time(log) / Trials，切换只重绘不重算。

**代码构建**：算法层 `deepcave/evaluators/stagnation.py`（`build_records` / `detect_full` / `annotate_advice`，另有 `StagnationDetector` 流式检测器）；插件层 `deepcave/plugins/objective/bottleneck_diagnosis.py` 继承 `DynamicPlugin`，`process` 一次性产出记录、区间与建议，渲染层用 `add_vrect` 画背景色带叠加在 incumbent 曲线上。

#### 5.5.2 Hypervolume Convergence（超体积收敛）

**算法原理**：对多目标 run 按提交顺序逐个加入成功 trial，增量维护 Pareto 前沿（统一到最小化空间，最大化目标取负），在至多 400 个检查点上计算支配超体积（HV），得到随试验数/时间增长的收敛曲线。多个 run 画在同一张图中直接对比；汇总表给出最终 HV、收敛曲线的梯形面积 AUC（越大越快越稳）、以及达到最终 HV 90% 所用的时间（t@90%），最优 run 置顶并标注。

**核心参数**：检查点数上限 400，超过则等距抽样；参照点由所有 run 的联合最差值自动构造；HV 算法按目标维数自动选择——1D 闭式解、2D O(n log n) 扫描、3D 切片精确解、4D 及以上用 WFG 精确算法；当前沿点数超过阈值时自动退化为 Monte Carlo 近似（10 万采样点，固定种子）以控制耗时。

**代码构建**：算法层 `deepcave/evaluators/hypervolume.py`（纯 numpy：`hypervolume_series` / `auc_trapezoid` / `time_to_threshold` 等 12 个函数）；插件层 `deepcave/plugins/objective/hypervolume_convergence.py` 继承 `DynamicPlugin`，`process` 为每个 run 产出 HV 序列，`load_outputs` 渲染曲线与汇总表；x 轴切换仅影响渲染。

#### 5.5.3 Fidelity Replay（保真度回放）

**算法原理**：把多保真 run 回放成 Successive Halving / Hyperband 式的"保真阶梯"。预算级映射为逐级 rung（用 run 自身的预算档位，或按 eta 构成几何阶梯）。逐 rung 统计：实际评估的配置数、最优成本、模拟 successive halving 会晋级多少配置，以及两个关键 KPI——覆盖率（被晋级的配置中真正在下一 rung 被评估的比例）与排名一致性（相邻 rung 之间成本的 Spearman 相关）。某配置被晋级但下一 rung 缺评估时按缺失策略处理。输入为 Group 时，每个成员单独一行 KPI，并附 Mean 汇总行，便于横向比较多保真算法"是否把钱花在了对的配置上"。

**核心参数**：`eta`（默认 auto，自动从预算阶梯推断几何比；也可手动指定，控制每级只保留 ⌈n/eta⌉ 个配置）；`missing_policy`（缺失策略：`carry_last` 沿用上一 rung 成本 / `skip` 剔除出阶梯 / `worst` 记为当前最差）；`objective_id` 选择目标。

**代码构建**：算法层 `deepcave/evaluators/fidelity_replay.py`（`infer_eta` / `collect_rung_costs` / `replay`，纯函数无 UI 依赖）；插件层 `deepcave/plugins/budget/fidelity_replay.py` 继承 `DynamicPlugin`，`process` 调 `replay` 得到逐 rung 指标，`load_outputs` 渲染阶梯图与两张表（rung KPI 表、组员 KPI 表）。

## 6. Evaluator（算法层）功能详解

Evaluator 是与 UI 解耦的纯算法层，**Web 插件与 API 脚本共用同一实现**。插件负责交互与绘图，evaluator 负责训练代理模型、计算指标。

### 6.1 重要性类

| Evaluator | 功能 | 被哪个插件使用 |
|-----------|------|--------------|
| `evaluators/fanova.py` — **fANOVA** | 全局超参重要性：基于 pyrfr 的 fANOVA 森林（`epm/fanova_forest.py`），把性能方差分解到各超参（及交互项）上，输出方差占比。 | Importances（"fANOVA (global)"） |
| `evaluators/lpi.py` — **LPI** | 局部参数重要性：以默认/incumbent 配置为参考点，在邻域内单独扰动一个超参，用方差占比衡量该超参在**当前最优点附近**的敏感度。 | Importances（"Local Parameter Importance (local)"） |
| `evaluators/weighted_sobol.py` ★新增 — **Weighted Sobol** | 加权 Sobol 敏感性：用 sklearn `RandomForestRegressor` 作代理，**以经验采样分布（各维独立边际）替代均匀分布**做 Sobol 一阶指数 S1 与总效应 ST 估计；可输出"均匀分布对照"以对比两种测度的差异。参数：`n_samples=2048, n_trees=64, seed=0`。 | Importances（"Weighted Sobol (empirical)"） |
| `evaluators/mo_fanova.py` / `mo_lpi.py` / `mo_ablation.py` | 多目标变体：基于帕累托前沿上的点生成目标加权方案，把两个目标标量化后分别跑 fANOVA/LPI/Ablation，展示重要性随权重变化的曲线。 | Importances / Ablation Paths（选择两个目标时） |

### 6.2 消融与行为类

| Evaluator | 功能 | 被哪个插件使用 |
|-----------|------|--------------|
| `evaluators/ablation.py` — **Ablation** | 消融路径：默认配置 → incumbent 的贪婪替换序列，量化每个超参"翻正"贡献多少性能。 | Ablation Paths |
| `evaluators/footprint.py` — **Footprint** | 配置足迹：对已评估配置做 **MDS** 二维嵌入，训练最近邻/代理模型生成性能表面，输出 configs/borders/supports/incumbents 四类点的 2D 坐标。 | Configuration Footprint、Footprint Replay |
| `evaluators/footprint_replay.py` ★新增 — **build_cloud / subset** | 时序足迹：冻结一次 MDS 投影，附上每个配置的 `first_seen`（首个成功 trial 序号）、逐 trial incumbent 与最优代价序列；`subset(cloud, trial=n)` 取截至第 n 个 trial 的快照。 | Footprint Replay（API 模式亦可用） |
| `evaluators/sampling_density.py` ★新增 | 采样密度：对编码后的配置矩阵按超参计算 1D 边际直方图/频次与 2D 联合直方图，重复评估的配置先折叠（按 `config_id` 取均值），支持均匀基线对照。 | Sampling Density |
| `evaluators/hypervolume.py` ★新增 — **Hypervolume** | 多目标超体积：`compute_hypervolume` 按维度分派算法档位（1D/2D/3D 精确 → WFG 精确 → Monte Carlo 近似），`hypervolume_series` 沿提交顺序增量维护 Pareto 前沿并在最多 `max_checkpoints=400` 个检查点计算 HV 序列，`auc_trapezoid` / `time_to_threshold` 派生收敛统计；另提供 `to_minimization`、`make_ref_point`、`normalize_points` 等工具。 | Hypervolume Convergence |
| `evaluators/fidelity_replay.py` ★新增 — **Fidelity Replay** | 保真度回放：`infer_eta` 从预算阶梯推断 η 比率；`collect_rung_costs` 按预算档收集配置级（种子平均）代价；`replay` 逐 rung 模拟 SH 晋升并输出每级 KPI（评估数/活跃数/晋升数、最优代价、覆盖率、级间秩相关）与最终 incumbent，支持 `carry_last` / `skip` / `worst` 三种缺失策略。 | Fidelity Replay |

### 6.3 代理模型（`evaluators/epm/`）

| 模块 | 功能 |
|------|------|
| `epm/random_forest.py` — **RandomForest** | 对 pyrfr 随机森林的封装：只需传配置空间即可自动处理超参类型/边界，支持 instance 特征，供 fANOVA/LPI/Ablation 等使用。 |
| `epm/fanova_forest.py` — **FanovaForest** | fANOVA 专用森林：在随机森林之上收集每棵树的分裂值中点（midpoints）与区间尺寸（sizes），用于计算边际方差。 |
| `epm/random_forest_surrogate.py` — **RandomForestSurrogate** | 适配 pyPDP 包的代理模型接口（fit/predict），供 Partial Dependencies 与 Symbolic Explanations 使用。 |
| `epm/utils.py` | EPM 相关工具函数（如类型/边界计算）。 |

## 7. 架构说明

```
                 ┌────────────────────────────────────────────┐
                 │                用户入口                     │
                 └──────────────┬─────────────┬───────────────┘
                                │             │
                     Web 模式   ▼             ▼   API 模式
              `deepcave` CLI（start.sh）   `import deepcave`
              ├─ redis-server (6379)             │
              ├─ worker.py × N（RQ 队列）        │
              └─ server.py（Dash @ 8050）        │
                                │             │
                                ▼             ▼
                 ┌────────────────────────────────────────────┐
                 │  插件层 plugins/（Dash 回调，仅 Web 生效）  │
                 │  StaticPlugin → RQ 异步 + 缓存             │
                 │  DynamicPlugin → 回调内同步计算             │
                 └──────────────────┬─────────────────────────┘
                                    ▼
                 ┌────────────────────────────────────────────┐
                 │  算法层 evaluators/（Web 与 API 通用）      │
                 │  fanova / lpi / ablation / footprint /     │
                 │  weighted_sobol★ / sampling_density★ /     │
                 │  footprint_replay★ + epm 代理模型          │
                 └──────────────────┬─────────────────────────┘
                                    ▼
                 ┌────────────────────────────────────────────┐
                 │  数据层 runs/（Run/Group/Trial/Recorder）   │
                 │  converters: smac3v1/v2, bohb, optuna,     │
                 │  amltk, raytune, dataframe, deepcave 原生  │
                 └────────────────────────────────────────────┘
```

关键机制：

1. **双模式判定**：`deepcave/__init__.py` 检查启动脚本名。以 `server.py`/`worker.py` 启动为 Web 模式；普通 `import` 则 `_api_mode=True`，插件层 `@interactive` 装饰的 GUI 方法自动空转，evaluator 层照常工作。
2. **重活走队列**：StaticPlugin 的 `process` 经 `_process` 投递到 Redis Queue，由 worker 异步执行、结果缓存（`cache/`），回调只读缓存不重算。
3. **转换器机制**：各优化器输出统一转换为内部 `Run` 抽象（配置空间、trial 历史、目标、预算），插件与 evaluator 只面向该抽象，与优化器解耦。
4. **Recorder 原生记录**：`deepcave.runs.recorder.Recorder` 提供上下文管理器，在训练脚本中边跑边写 DeepCAVE 原生格式，支持运行中实时分析。
5. **Group 分析**：`runs/group.py` 可把多个 run 合并为一个 Group（要求目标/配置空间一致），实现多 run 联合诊断。

## 8. 示例数据与示例代码

### 8.1 预置运行数据（`logs/`）

仓库自带多种优化器/场景的预评估运行，启动 Web 版后在左侧 Run Selection 中即可加载（点击目录旁的 `+`）：

| 目录 | 内容 |
|------|------|
| `logs/DeepCAVE/` | 原生 Recorder 格式：`minimal`（最小示例）、`mnist_pytorch`（MNIST 调参） |
| `logs/SMAC3v1/`、`logs/SMAC3v2/` | SMAC v1/v2 输出 |
| `logs/BOHB/` | BOHB（HpBandSter）输出 |
| `logs/Optuna/` | Optuna 输出 |
| `logs/AMLTK/` | AMLTK 的 SMAC/Optuna optimizer 输出 |
| `logs/RayTune/` | RayTune 输出 |
| `logs/DataFrame/` | 由 Pandas DataFrame 转换的运行 |
| `logs/NAS/nb301_cifar10` | NASBench-301 神经架构搜索 |
| `logs/LLM/lm1b_2048` | 基于 PD1 的大语言模型调参（lm1b） |

### 8.2 示例代码（`examples/`）

- `examples/record/minimal.py` — 用 Recorder 记录最小运行（含双目标、多预算）；
- `examples/record/digits_sklearn.py`、`examples/record/mnist_pytorch.py` — 真实训练任务记录；
- `examples/api/importances.py` 等 — API 模式直接调用 evaluator 出图（需先运行 record 示例生成数据）。

### 8.3 测试（`tests/`）

```bash
conda activate DeepCAVE
pip install -e '/root/deepcave-experiments[dev]'
pytest tests/ -x          # 含 test_fanova / test_ablation / test_footprint_replay★ / test_sampling_density★ 等
```

## 9. 安装与启动验证记录

以下步骤于 2026-09-29 在本机（Ubuntu，conda `DeepCAVE` 环境，Python 3.9.25，系统已装 redis-server/swig）实测通过：

| 步骤 | 命令 | 结果 |
|------|------|------|
| 1. 安装 | `conda activate DeepCAVE && pip install -e /root/deepcave-experiments` | ✅ 成功，`deepcave-1.4.1`（editable）及全部依赖安装完成 |
| 2. CLI 注册 | `which deepcave` | ✅ `/root/miniconda3/envs/DeepCAVE/bin/deepcave` |
| 3. 启动 Web | `deepcave` | ✅ redis 检测通过（PONG）→ worker 启动 → `Dash is running on http://127.0.0.1:8050/` |
| 4. 网页可达 | `curl http://127.0.0.1:8050/` | ✅ HTTP 200，页面标题 `<title>DeepCAVE</title>` |
| 5. API 导入 | `import deepcave` 及 14 个插件/评估器模块 | ✅ `deepcave version: 1.4.1`，全部导入成功 |

> 备注：首次启动会自动拉起 redis-server（若未运行）；在远程服务器上部署时，用 SSH 端口转发访问 `http://127.0.0.1:8050`。

## 10. 插件开发规范（插件实现必须遵守）

> 本节沉淀自 bottleneck 插件（P4 瓶颈检测）上线初期的一轮线上问题复盘：插件在「切换 run / 刷新页面」后永久空白，且服务端无任何报错。后续实现任何新插件（如 Fidelity Replay、Hypervolume Convergence）都必须达到本节标准。

### 10.1 核心验收标准（缺一不可）

1. **切换 run 立即出图**：在插件下拉框切换到任意 run（包括之前看过的 run），输出必须立即正确渲染；
2. **改配置即时更新**：任意输入（objective / budget / 滑块等）变化后，图像即时更新；
3. **刷新页面出图**：F5 硬刷新后插件自动恢复渲染，不依赖用户手动点 Update；
4. **运行中 run 实时刷新**：`global-update` interval（500ms）tick 下，仍在评估的 run 数据变化能自动反映；
5. **全程无 traceback**：并发操作（快速切换 run / 连续改配置 / 页面刷新叠加 interval tick）下服务端日志零异常。

### 10.2 缓存规范（per-run 缓存 + hash 自动失效）

- 使用 `rc = RunCaches` 提供的 per-run 缓存：`rc.get(run, plugin_id, inputs_key)` 读、`rc.set(run, plugin_id, inputs_key, value)` 写。缓存 key 必须包含影响输出的全部输入。
- **绝不在 interval 回调中调用 `rc.clear()`**。`rc.clear()` 会对整个 `run_cache` 目录执行 `rmtree`，引发连锁反应：所有插件缓存被清 → index.json 丢失 → `RunHandler` 重建缓存并并发修改 `self.runs` → 请求线程迭代快照时 `del self.runs[run_path]` 抛 `KeyError`，且每个 tick 都全量重算（卡顿 + 数据丢失）。
- 运行中 run 的实时更新依赖 run 内容 hash：`RunHandler.update()` 会检测 run 文件变化并调用 `RunCaches.update` 自动失效该 run 的缓存，因此插件只需「tick 时 `rc.get` 未命中就重算」，无需自己管理失效。

### 10.3 线程安全规范

- `RunHandler.update()/update_runs()/update_groups()` 持 `threading.RLock`（`self._lock`），内部删除状态时用 `dict.pop(run_path, None)` / `.get()`，避免与并发请求线程迭代快照竞争（`KeyError: .../run_N` 的根因）。
- 插件回调内不要自行读写 `run_handler.runs` 内部结构，一律通过 `get_run / get_runs` 公开接口。
- `StaticPlugin.process` 的输出会经 RQ 队列序列化缓存，**必须是 JSON 可序列化的纯数据**（dict/list/float/str，plotly figure 在 `load_outputs` 里再组装）。

### 10.4 实时更新 / PreventUpdate 规范（本轮复盘的核心教训）

- 允许的去重方式：**仅当回调的唯一触发源是 `global-update` interval、且 (inputs_key, run 内容签名) 均未变化时，才可 `raise PreventUpdate()`**：

  ```python
  triggered = callback_context.triggered
  if triggered:
      triggered_ids = [...]
      interval_only = all("global-update" in p for p in triggered_ids)
      if interval_only and signature == self._realtime_signature:
          raise PreventUpdate()
  self._realtime_signature = signature
  ```

- **禁止**用「签名未变就 PreventUpdate」覆盖所有触发源。签名是服务端实例状态，会在页面刷新后保留、会在切回旧 run 时命中旧值，一旦命中，之后每个 interval tick 都持续拦截 → 插件永久空白，且 terminal 零输出（Dash 静默返回 204），极难排查。
- 页面加载（`triggered` 为空）、run 切换、配置变更、Update 按钮点击，一律照常渲染。

### 10.5 失效 run id 与降级处理

- 恢复 `last_inputs` 时必须校验缓存的 run id 是否仍存在于当前 options，无效则重置为 `None`（防止「Run not found」RuntimeError 或静默空白）。
- 输出回调内 `get_run / get_runs` 抛 `RuntimeError` 时（run 在解析与渲染之间被删除/切换），降级为提示性 `dbc.Alert`，而不是抛 500。
- `load_dependency_inputs` 在 run 切换后必须返回与**新 run** 匹配的 options/value（objective/budget/seed 等），确保下游 `process` 收到的输入组合有效。

### 10.6 上线前自测清单

- [ ] 浏览器真实触发链路测试（`changedPropIds=run-input.value` 等），而非只测手动 Update 按钮；
- [ ] run A → run B → run A 往返切换；
- [ ] F5 刷新后自动渲染；
- [ ] 运行中 run 的 500ms tick 更新 + 改配置并发；
- [ ] `pytest tests/` 全绿，服务端日志 0 traceback。

---

## 11. 致谢与引用

本仓库基于 [automl/DeepCAVE](https://github.com/automl/deepcave) v1.4.1 二次开发，新增功能的设计动机详见仓库根目录的 `development-plan.md` 与 `proposal.md`。

若使用 DeepCAVE，请引用其 [ReALML@ICML'22 workshop 论文](https://arxiv.org/abs/2206.03493)：

```bibtex
@misc{sass-realml2022,
    title = {DeepCAVE: An Interactive Analysis Tool for Automated Machine Learning},
    author = {Sass, René and Bergman, Eddie and Biedenkapp, André and Hutter, Frank and Lindauer, Marius},
    doi = {10.48550/ARXIV.2206.03493},
    url = {https://arxiv.org/abs/2206.03493},
    publisher = {arXiv},
    year = {2022},
    copyright = {arXiv perpetual, non-exclusive license}
}
```

Copyright (C) 2021-2024 The DeepCAVE Authors · Apache License 2.0




