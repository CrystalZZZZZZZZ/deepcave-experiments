DeepCAVE 本质上不是“帮你搜索超参数”的优化器，而是 AutoML / HPO（超参数优化）过程的**可视化和分析平台**。它围绕一次 HPO 产生的数据（DeepCAVE 中叫 Run）提供交互式探索、诊断和论文级图表。下面按架构和功能两部分介绍，均基于 v1.4.1。

## 架构

可以理解成五层：

1. **数据接入层（Runs & Converters）**

   核心概念是 Run——一次超参数优化过程，包含一组 trial，每个 trial 是“配置 + 目标值 + budget + seed + 状态”。DeepCAVE 不锁死某个优化器，而是通过转换器读取各种框架落盘的数据：

   - 原生格式：用 `Recorder` 在搜索循环里直接记录，保存 `history.jsonl`、`configspace.json`、`configs.json`、`origins.json`、`meta.json`；
   - 第三方：SMAC v1/v2、AMLTK、Optuna、RayTune、BOHB；
   - 通用表格：Pandas DataFrame（只需 `configspace.csv` + `trials.csv`）。

   这一层支持**观察文件系统**，所以既能看到已完成 run，也能实时监控仍在写入磁盘的搜索过程；多个 run 可以组成 Group 做联合分析。

2. **队列与缓存层（Redis + Worker）**

   耗时的分析任务（如训练代理模型算超参数重要性）被设计成“静态插件”。交互模式下，任务先进入 Redis 队列，由本地 worker 进程取走计算，结果写回 run 目录缓存；下次看同一个 run 直接读缓存。轻量插件则是“动态插件”，输入一改立即重算。API 模式完全绕过这一层，直接在当前 Python 进程里同步计算。

3. **插件层（Plugins）**

   这是 DeepCAVE 的功能主体。插件统一按三块组织：**Input block**（触发计算）、**Filter block**（只影响显示）、**Output block**（图表/组件）。插件分静态（排队、可缓存）和动态（即时），并且可以自定义。

4. **展示/交互层（Dash Web UI）**

   交互面板基于 Dash：主页面负责选择 run/group，左侧选择插件，每个插件包含参数面板和图表。服务由 `deepcave` CLI 启动，可配置 IP、端口、worker 数量、刷新率等。API 模式则不启动 Web 服务，直接 import 插件类调用 `generate_inputs → generate_outputs → load_outputs`。

5. **扩展层**

   官方提供“如何添加自定义 Converter”和“如何添加自定义 Plugin”教程。要接入你自己的搜索框架或新增专属分析图，不需要改核心代码，按插件/转换器接口扩展即可。

## 主要功能

### Run 管理与监控

- 目录式加载 run、自动识别优化器类型；
- 多 run 选择与 **Group 联合分析**（比如多次重复实验看均值和方差）；
- 实时监测正在运行的 HPO 过程，自动感知 run 文件变化；
- 结果缓存与异步计算队列，避免重计算。

### 分析插件（按用途分）

| 类别 | 插件 | 回答的问题 |
|---|---|---|
| 总览 | Overview | run 的元信息、目标、状态统计、配置空间是否正确 |
| 配置分析 | Configurations | 查看某个具体配置的来源、各目标表现、在不同 budget 下表现 |
| 性能分析 | Cost over Time | 目标值随时间/试验次数如何提升，多 run 谁更好 |
| 多目标 | Pareto Front | 两个目标间的 Pareto 前沿，比如精度和训练成本 |
| 预算分析 | Budget Correlation | 低预算结果和全预算结果相关吗，多 fidelity 是否可靠 |
| 重要性 | Importances | 每个超参数对目标影响多大（fANOVA / LPI） |
| 路径分析 | Ablation Paths | 从默认配置到最优配置，哪些超参数改动带来提升 |
| 配置空间 | Configuration Cube | 用 2D/3D 切片观察配置在空间中的分布和性能 |
| 相关性 | Parallel Coordinates | 各超参数取值与最终目标之间的趋势 |
| 依赖关系 | Partial Dependencies | 单个/两个超参数与目标函数的关系（PDP） |
| 可解释 | Symbolic Explanations | 用符号回归给出超参数与性能的显式公式 |
| 探索覆盖 | Configuration Footprint | 优化器探索空间的充分性和倾向性 |

这些插件基本覆盖了超参数搜索方案最关心的几个问题：**搜得够不够广、哪里值得重点搜、哪些参数重要、多预算是否一致、多目标如何权衡**。

### 优化器支持与 API

- 支持 SMAC、AMLTK、Optuna、RayTune、BOHB，以及 DataFrame 通用导入；
- 自己跑搜索时用 `Recorder` 逐条记录，格式统一；
- API 模式支持 Python 直接调用插件生成 Plotly 图并保存，适合做自动化报告；
- 多目标、multi-fidelity budget、多 seed 都是数据模型中的一等概念，几乎所有插件都能感知这些维度。

### 易用性

- 集成帮助按钮、内置文档；
- 插件 UI 有统一的 Input/Filter/Output 交互范式；
- 通过自定义配置可以改服务地址、端口、缓存目录、是否保存图片、刷新率等。

## 一句话总结

DeepCAVE 的架构可以理解为：**转换器统一数据格式 → Run 数据模型承载 trial 结构 → 插件把数据变成分析视图 → Redis/worker 处理重计算 → Dash/API 两种方式消费结果**。对你的超参数搜索方案来说，它的定位是“搜索结束或进行中时的分析诊断层”，搜索本身仍需要 SMAC3、Optuna 这类优化器或自己的搜索循环来驱动。